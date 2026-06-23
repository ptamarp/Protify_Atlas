import entrypoint_setup

import os
import math
import queue
import threading
import time
import torch
import torch.multiprocessing as mp
import warnings
import sqlite3
import gzip
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from huggingface_hub import hf_hub_download

try:
    from seed_utils import seed_worker, dataloader_generator, get_global_seed
    from data.dataset_classes import SimpleProteinDataset
    from base_models.atlas import (
        ATLAS_EMBEDDING_KINDS,
        atlas_embedding_kind,
        atlas_kind_is_matrix,
        is_atlas_ppi_model_name,
    )
    from base_models.get_base_models import get_base_model
    from pooler import Pooler
    from utils import torch_load, print_message, maybe_compile, tensor_to_embedding_blob, batch_tensor_to_blobs, _SQLWriter
except ImportError:
    from .seed_utils import seed_worker, dataloader_generator, get_global_seed
    from .data.dataset_classes import SimpleProteinDataset
    from .base_models.atlas import (
        ATLAS_EMBEDDING_KINDS,
        atlas_embedding_kind,
        atlas_kind_is_matrix,
        is_atlas_ppi_model_name,
    )
    from .base_models.get_base_models import get_base_model
    from .pooler import Pooler
    from .utils import (
        torch_load, print_message, maybe_compile,
        tensor_to_embedding_blob, batch_tensor_to_blobs, _SQLWriter,
    )


def build_collator(tokenizer: object, padding: str = 'max_length', max_length: int = 2048) -> Callable[[List[str]], Dict[str, torch.Tensor]]:
    def _collate_fn(sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Collate function for batching sequences."""
        kwargs: Dict[str, Any] = dict(
            return_tensors="pt", padding=padding, truncation=True, max_length=max_length,
        )
        if padding != 'max_length':
            kwargs['pad_to_multiple_of'] = 8
        return tokenizer(sequences, **kwargs)
    return _collate_fn


def get_embedding_filename(
        model_name: str,
        matrix_embed: bool,
        pooling_types: List[str],
        extension: str = 'pth',
        hidden_state_index: int = -1,
) -> str:
    """
    Generate embedding filename with pooling types for vector embeddings.
    
    Args:
        model_name: Name of the model
        matrix_embed: Whether embeddings are matrices (True) or vectors (False)
        pooling_types: List of pooling types used (only relevant for vector embeddings)
        extension: File extension ('pth' or 'db')
        hidden_state_index: Hidden-state tuple index used for embeddings. -1 uses the final hidden state.
    
    Returns:
        Filename string in format: {model_name}_{matrix_embed}[_hs{hidden_state_index}][_{pooling_types}].{extension}
    """
    assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
    base_name = f'{model_name}_{matrix_embed}'
    if hidden_state_index != -1:
        base_name = f'{base_name}_hs{hidden_state_index}'
    if is_atlas_ppi_model_name(model_name):
        base_name = f'{base_name}_{atlas_embedding_kind(matrix_embed, pooling_types)}'
        return f'{base_name}.{extension}'
    if not matrix_embed and pooling_types:
        # For vector embeddings, include pooling types in filename
        pooling_str = '_'.join(sorted(pooling_types))  # Sort for consistency
        base_name = f'{base_name}_{pooling_str}'
    return f'{base_name}.{extension}'


def get_atlas_embedding_filename(
        model_name: str,
        embedding_kind: str,
        extension: str = 'pth',
        hidden_state_index: int = -1,
) -> str:
    return get_embedding_filename(
        model_name,
        atlas_kind_is_matrix(embedding_kind),
        [embedding_kind],
        extension,
        hidden_state_index,
    )


def _make_embedding_progress(
    dataloader: DataLoader,
    padding: str,
    n_warmup: int = 3,
    n_calibration: int = 5,
) -> Iterator[Tuple[int, Any]]:
    """Progress-bar wrapper for embedding loops. Drop-in replacement for enumerate(dataloader).

    When padding='max_length', all batches have uniform cost so plain tqdm works.
    When padding='longest' (sorted longest-first), batch times vary dramatically.
    In that case: yield warmup batches first (compiler warmup + OOM check on longest
    sequences), then time mid-length calibration batches to estimate total ETA.

    Keep in sync with fastplms/embedding_mixin.py. Canonical source: core/embed/progress.py.
    """
    total = len(dataloader)
    if padding == 'max_length' or total <= n_warmup + n_calibration:
        for i, batch in tqdm(enumerate(dataloader), total=total, desc='Embedding batches'):
            yield i, batch
        return

    dl_iter = iter(dataloader)

    # Phase 1: warmup on longest batches (first n_warmup, since sorted longest-first)
    warmup_bar = tqdm(range(n_warmup), desc='Warmup (longest batches)', leave=False)
    for i in warmup_bar:
        batch = next(dl_iter)
        yield i, batch
    warmup_bar.close()

    # Phase 2: skip to middle of dataset for calibration timing
    # We need to yield all intermediate batches too (they contain real data)
    mid_start = total // 2
    intermediate_bar = tqdm(
        range(n_warmup, mid_start), desc='Embedding batches', leave=False,
    )
    for i in intermediate_bar:
        batch = next(dl_iter)
        yield i, batch
    intermediate_bar.close()

    # Phase 3: time calibration batches from the middle
    calibration_times: List[float] = []
    cal_bar = tqdm(range(n_calibration), desc='Calibrating ETA', leave=False)
    for j in cal_bar:
        t0 = time.perf_counter()
        batch = next(dl_iter)
        yield mid_start + j, batch
        calibration_times.append(time.perf_counter() - t0)
    cal_bar.close()

    avg_time = sum(calibration_times) / len(calibration_times)
    remaining_start = mid_start + n_calibration
    remaining_count = total - remaining_start
    estimated_total_seconds = avg_time * remaining_count

    # Phase 4: remaining batches with calibrated ETA
    main_bar = tqdm(
        range(remaining_count),
        desc='Embedding batches',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )
    main_bar.set_postfix_str(f'ETA ~{estimated_total_seconds:.0f}s (calibrated)')
    for k in main_bar:
        batch = next(dl_iter)
        yield remaining_start + k, batch
    main_bar.close()


@dataclass
class EmbeddingArguments:
    def __init__(
            self,
            embedding_batch_size: int = 4,
            embedding_num_workers: int = 0,
            download_embeddings: bool = False,
            download_dir: str = 'Synthyra/vector_embeddings',
            matrix_embed: bool = False,
            embedding_pooling_types: List[str] = ['mean'],
            save_embeddings: bool = False,
            embed_dtype: torch.dtype = torch.float32,
            model_dtype: torch.dtype = None,
            sql: bool = False,
            embedding_save_dir: str = 'embeddings',
            padding: str = 'max_length',
            max_length: int = 2048,
            embedding_hidden_state_index: int = -1,
            multi_gpu: bool = False,
            autocast: bool = False,
            embedding_scaler: bool = True,
            **kwargs
    ):
        self.batch_size = embedding_batch_size
        self.num_workers = embedding_num_workers
        self.download_embeddings = download_embeddings
        self.download_dir = download_dir
        self.matrix_embed = matrix_embed
        self.pooling_types = embedding_pooling_types
        self.save_embeddings = save_embeddings
        self.embed_dtype = embed_dtype
        self.model_dtype = model_dtype
        self.sql = sql
        self.embedding_save_dir = embedding_save_dir
        self.padding = padding
        self.max_length = max_length
        self.hidden_state_index = embedding_hidden_state_index
        self.multi_gpu = multi_gpu
        self.autocast = autocast
        assert isinstance(embedding_scaler, bool), f"Invalid embedding_scaler: {embedding_scaler}"
        self.embedding_scaler = embedding_scaler


class Embedder:
    def __init__(self, args: EmbeddingArguments, all_seqs: List[str]):
        self.args = args
        self.all_seqs = all_seqs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.matrix_embed = args.matrix_embed
        self.pooling_types = args.pooling_types
        self.download_embeddings = args.download_embeddings
        self.download_dir = args.download_dir
        self.save_embeddings = args.save_embeddings
        self.embed_dtype = args.embed_dtype
        self.model_dtype = args.model_dtype
        self.sql = args.sql
        self.embedding_save_dir = args.embedding_save_dir
        self.padding = args.padding
        self.max_length = args.max_length
        self.hidden_state_index = args.hidden_state_index
        self.multi_gpu = args.multi_gpu
        self.autocast = args.autocast

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print_message(f'Device {self.device} found ({n_gpus} GPU{"s" if n_gpus != 1 else ""})')

    def _download_embeddings(self, model_name: str):
        # download from download_dir
        # unzip
        # move to embedding_save_dir
        filename = get_embedding_filename(
            model_name,
            self.matrix_embed,
            self.pooling_types,
            'pth',
            self.hidden_state_index,
        )
        try:
            local_path = hf_hub_download(
                repo_id=self.download_dir,
                filename=f'embeddings/{filename}.gz',
                repo_type='dataset'
            )
        except Exception:
            print(f'No embeddings found for {model_name} in {self.download_dir}')
            return

        # unzip
        print_message(f'Unzipping {local_path}')
        with gzip.open(local_path, 'rb') as f_in:
            with open(local_path.replace('.gz', ''), 'wb') as f_out:
                f_out.write(f_in.read())
        # move to embedding_save_dir
        unzipped_path = local_path.replace('.gz', '')
        final_path = os.path.join(self.embedding_save_dir, filename)
        
        if os.path.exists(final_path):
            print_message(f'Found existing embeddings in {final_path}')
            # Load downloaded embeddings
            downloaded_embeddings = torch_load(unzipped_path)
            existing_embeddings = torch_load(final_path)

            download_dtype = torch.float16
            if self.embed_dtype != download_dtype:
                print_message(f"Warning:\nDownloaded embeddings are {download_dtype} but the current setting is {self.embed_dtype}\nWhen combining with existing embeddings, this could result in unintended biases or reductions in performance")

            # Combine with existing embeddings
            print_message('Combining and casting')
            downloaded_embeddings.update(existing_embeddings)

            # Cast all embeddings to the correct dtype
            for seq in downloaded_embeddings:
                downloaded_embeddings[seq] = downloaded_embeddings[seq].to(self.embed_dtype)

            # Save the combined embeddings
            print_message(f'Saving combined embeddings to {final_path}')
            torch.save(downloaded_embeddings, final_path)
        else:
            print_message(f'Downloading embeddings from {self.download_dir}, no previous embeddings found')
            downloaded_embeddings = torch.load(unzipped_path)
            torch.save(downloaded_embeddings, final_path)
        return final_path

    def _read_sequences_from_db(self, db_path: str) -> Set[str]:
        """Read all embedded sequences from SQLite database."""
        import sqlite3
        with sqlite3.connect(db_path, timeout=30) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            return {row[0] for row in c.fetchall()}

    def _read_embeddings_from_disk(self, model_name: str):
        if is_atlas_ppi_model_name(model_name):
            return self._read_atlas_embeddings_from_disk(model_name)

        if self.sql:
            filename = get_embedding_filename(
                model_name,
                self.matrix_embed,
                self.pooling_types,
                'db',
                self.hidden_state_index,
            )
            save_path = os.path.join(self.embedding_save_dir, filename)
            if os.path.exists(save_path):
                conn = sqlite3.connect(save_path, timeout=30)
                c = conn.cursor()
                c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
                conn.close()
                already_embedded = self._read_sequences_from_db(save_path)
                to_embed = [seq for seq in self.all_seqs if seq not in already_embedded]
                print_message(f"Loaded {len(already_embedded)} already embedded sequences from {save_path}\nEmbedding {len(to_embed)} new sequences")
                return to_embed, save_path, {}
            else:
                print_message(f"No embeddings found in {save_path}")
                return self.all_seqs, save_path, {}

        else:
            embeddings_dict = {}
            filename = get_embedding_filename(
                model_name,
                self.matrix_embed,
                self.pooling_types,
                'pth',
                self.hidden_state_index,
            )
            save_path = os.path.join(self.embedding_save_dir, filename)
            if os.path.exists(save_path):
                print_message(f"Loading embeddings from {save_path}")
                embeddings_dict = torch_load(save_path)
                print_message(f"Loaded {len(embeddings_dict)} embeddings from {save_path}")
                # Cast existing embeddings to the specified dtype
                #for seq in embeddings_dict:
                #    embeddings_dict[seq] = embeddings_dict[seq].to(self.embed_dtype)
                to_embed = [seq for seq in self.all_seqs if seq not in embeddings_dict]
                return to_embed, save_path, embeddings_dict
            else:
                print_message(f"No embeddings found in {save_path}")
                return self.all_seqs, save_path, {}

    def _native_atlas_embedding_kind(self) -> str:
        try:
            return atlas_embedding_kind(self.matrix_embed, self.pooling_types)
        except AssertionError:
            # Preserve the historical Protify default when Atlas is selected but
            # no Atlas-specific embedding kind was supplied.
            if not self.matrix_embed and self.pooling_types in (['mean'], ['mean', 'var'], ['var', 'mean']):
                return 'pooled_concat'
            raise

    @staticmethod
    def _uses_native_atlas_embeddings(model: Any) -> bool:
        return bool(
            getattr(model, 'atlas_native_embedding', False)
            or getattr(getattr(model, '_orig_mod', None), 'atlas_native_embedding', False)
        )

    def _atlas_paths(self, model_name: str, extension: str) -> Dict[str, str]:
        return {
            embedding_kind: os.path.join(
                self.embedding_save_dir,
                get_atlas_embedding_filename(
                    model_name,
                    embedding_kind,
                    extension,
                    self.hidden_state_index,
                ),
            )
            for embedding_kind in ATLAS_EMBEDDING_KINDS
        }

    def _read_atlas_sequences_from_db_paths(self, paths_by_kind: Dict[str, str]) -> Dict[str, Set[str]]:
        embedded_by_kind = {}
        for embedding_kind, path in paths_by_kind.items():
            if os.path.exists(path):
                conn = sqlite3.connect(path, timeout=30)
                c = conn.cursor()
                c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
                conn.close()
                embedded_by_kind[embedding_kind] = self._read_sequences_from_db(path)
            else:
                embedded_by_kind[embedding_kind] = set()
        return embedded_by_kind

    def _read_atlas_embeddings_from_disk(self, model_name: str):
        requested_kind = self._native_atlas_embedding_kind()
        extension = 'db' if self.sql else 'pth'
        paths_by_kind = self._atlas_paths(model_name, extension)
        requested_path = paths_by_kind[requested_kind]

        if self.sql:
            embedded_by_kind = self._read_atlas_sequences_from_db_paths(paths_by_kind)
            to_embed = [
                seq for seq in self.all_seqs
                if any(seq not in embedded_by_kind[embedding_kind] for embedding_kind in ATLAS_EMBEDDING_KINDS)
            ]
            already_requested = len(embedded_by_kind[requested_kind])
            print_message(
                f"Loaded {already_requested} already embedded {requested_kind} Atlas sequences from {requested_path}\n"
                f"Embedding {len(to_embed)} sequences missing from at least one Atlas cache"
            )
            return to_embed, requested_path, {}

        embeddings_dict = {}
        embeddings_by_kind: Dict[str, Dict[str, torch.Tensor]] = {}
        for embedding_kind, path in paths_by_kind.items():
            if os.path.exists(path):
                print_message(f"Loading Atlas {embedding_kind} embeddings from {path}")
                embeddings_by_kind[embedding_kind] = torch_load(path)
                print_message(f"Loaded {len(embeddings_by_kind[embedding_kind])} Atlas {embedding_kind} embeddings")
            else:
                print_message(f"No Atlas {embedding_kind} embeddings found in {path}")
                embeddings_by_kind[embedding_kind] = {}

        to_embed = [
            seq for seq in self.all_seqs
            if any(seq not in embeddings_by_kind[embedding_kind] for embedding_kind in ATLAS_EMBEDDING_KINDS)
        ]
        embeddings_dict = embeddings_by_kind[requested_kind]
        self._atlas_existing_embeddings_by_kind = embeddings_by_kind
        return to_embed, requested_path, embeddings_dict

    def _split_native_embeddings(
            self,
            batch_embeddings: Any,
            seqs: List[str],
            embedding_kind: str = "unknown",
    ) -> List[torch.Tensor]:
        if isinstance(batch_embeddings, torch.Tensor):
            if batch_embeddings.ndim > 0 and batch_embeddings.shape[0] == len(seqs):
                split = [batch_embeddings[i] for i in range(len(seqs))]
            elif len(seqs) == 1:
                split = [batch_embeddings]
            else:
                raise AssertionError(
                    f"Atlas {embedding_kind} returned shape {tuple(batch_embeddings.shape)} "
                    f"for {len(seqs)} sequences; first dimension must match batch size."
                )
        elif isinstance(batch_embeddings, (list, tuple)):
            assert len(batch_embeddings) == len(seqs), (
                f"Atlas {embedding_kind} returned {len(batch_embeddings)} embeddings for {len(seqs)} sequences"
            )
            split = [torch.as_tensor(embedding) for embedding in batch_embeddings]
        else:
            raise TypeError(f"Unsupported Atlas {embedding_kind} embedding return type: {type(batch_embeddings)!r}")

        normalized = []
        for seq, emb in zip(seqs, split):
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            normalized.append(emb.detach().cpu())
        return normalized

    def _derive_atlas_embeddings_by_kind(
            self,
            raw_embeddings_by_kind: Dict[str, Any],
            seqs: List[str],
            model: Any,
    ) -> Dict[str, List[torch.Tensor]]:
        assert isinstance(raw_embeddings_by_kind, dict), (
            "Atlas embed_sequences(sequences) must return a dictionary of all embedding views."
        )
        missing = set(ATLAS_EMBEDDING_KINDS) - set(raw_embeddings_by_kind)
        assert not missing, f"Atlas embed_sequences did not return required views: {sorted(missing)}"

        normalized: Dict[str, List[torch.Tensor]] = {}
        for embedding_kind in ATLAS_EMBEDDING_KINDS:
            normalized[embedding_kind] = self._split_native_embeddings(
                raw_embeddings_by_kind[embedding_kind],
                seqs,
                embedding_kind=embedding_kind,
            )
        return normalized

    def _open_atlas_sql_writers(self, paths_by_kind: Dict[str, str]) -> Tuple[Dict[str, sqlite3.Connection], Dict[str, _SQLWriter]]:
        conns = {}
        writers = {}
        for embedding_kind, path in paths_by_kind.items():
            conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
            c = conn.cursor()
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('PRAGMA busy_timeout=30000')
            c.execute('PRAGMA synchronous=OFF')
            c.execute('PRAGMA cache_size=-64000')
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            writer = _SQLWriter(conn)
            writer.__enter__()
            conns[embedding_kind] = conn
            writers[embedding_kind] = writer
        return conns, writers

    def _write_atlas_sql_batch(
            self,
            writers_by_kind: Dict[str, _SQLWriter],
            seqs: List[str],
            embeddings_by_kind: Dict[str, List[torch.Tensor]],
    ) -> None:
        for embedding_kind, embeddings in embeddings_by_kind.items():
            embeddings = [emb.to(self.embed_dtype) for emb in embeddings]
            first_shape = tuple(embeddings[0].shape)
            same_shape = all(tuple(emb.shape) == first_shape for emb in embeddings)
            if same_shape:
                stacked = torch.stack(embeddings)
                blobs = batch_tensor_to_blobs(stacked)
                batch_rows = list(zip(seqs, blobs))
            else:
                batch_rows = [
                    (seq, tensor_to_embedding_blob(emb))
                    for seq, emb in zip(seqs, embeddings)
                ]
            writers_by_kind[embedding_kind].write_batch(batch_rows)

    @torch.inference_mode()
    def _embed_sequences_with_native_atlas(
            self,
            to_embed: List[str],
            save_path: str,
            embedding_model: Any,
            embeddings_dict: Dict[str, torch.Tensor],
            model_name: str = 'Atlas-PPI-auto',
    ) -> Optional[Dict[str, torch.Tensor]]:
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        model = embedding_model.to(self.device).eval()
        requested_kind = self._native_atlas_embedding_kind()
        print_message(f'Atlas requested embedding kind: {requested_kind}')

        to_embed = sorted(to_embed, key=len, reverse=True)
        extension = 'db' if self.sql else 'pth'
        paths_by_kind = self._atlas_paths(model_name, extension)
        conns_by_kind = {}
        writers_by_kind = {}
        embeddings_by_kind_cache = getattr(self, '_atlas_existing_embeddings_by_kind', None)
        if not self.sql and embeddings_by_kind_cache is None:
            embeddings_by_kind_cache = {}
            for embedding_kind, path in paths_by_kind.items():
                embeddings_by_kind_cache[embedding_kind] = torch_load(path) if os.path.exists(path) else {}

        if self.sql:
            conns_by_kind, writers_by_kind = self._open_atlas_sql_writers(paths_by_kind)

        total_batches = math.ceil(len(to_embed) / self.batch_size)
        try:
            for batch_start in tqdm(range(0, len(to_embed), self.batch_size), total=total_batches, desc='Embedding batches'):
                seqs = to_embed[batch_start:batch_start + self.batch_size]
                with torch.autocast(self.device.type, dtype=self.embed_dtype, enabled=self.autocast):
                    raw_embeddings_by_kind = model.embed_sequences(seqs)
                embeddings_by_kind = self._derive_atlas_embeddings_by_kind(raw_embeddings_by_kind, seqs, model)

                if self.sql:
                    self._write_atlas_sql_batch(writers_by_kind, seqs, embeddings_by_kind)
                else:
                    for embedding_kind, embeddings in embeddings_by_kind.items():
                        kind_cache = embeddings_by_kind_cache.setdefault(embedding_kind, {})
                        for seq, emb in zip(seqs, embeddings):
                            kind_cache[seq] = emb.to(self.embed_dtype)
                    embeddings_dict = embeddings_by_kind_cache[requested_kind]
        finally:
            if self.sql:
                for writer in writers_by_kind.values():
                    writer.__exit__(None, None, None)
                for conn in conns_by_kind.values():
                    conn.close()

        if not self.sql and self.save_embeddings:
            for embedding_kind, path in paths_by_kind.items():
                print_message(f"Saving Atlas {embedding_kind} embeddings to {path}")
                torch.save(embeddings_by_kind_cache[embedding_kind], path)

        return embeddings_dict

    @torch.inference_mode()
    def _embed_sequences(
            self,
            to_embed: List[str],
            save_path: str,
            embedding_model: Any,
            tokenizer: Any,
            embeddings_dict: Dict[str, torch.Tensor],
            model_name: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        if self._uses_native_atlas_embeddings(embedding_model):
            return self._embed_sequences_with_native_atlas(
                to_embed,
                save_path,
                embedding_model,
                embeddings_dict,
                model_name=model_name or 'Atlas-PPI-auto',
            )

        model = embedding_model.to(self.device).eval()
        dynamic = self.padding == 'longest'
        model = maybe_compile(model, dynamic=dynamic)
        device = self.device
        collate_fn = build_collator(tokenizer, padding=self.padding, max_length=self.max_length)
        # Models that self-pool (e.g. Vec2VecForEmbedding: base+pooler+translator
        # bundled) set already_pooled=True; forward() returns (B, D) directly and
        # the Protify embedder must skip its own Pooler to avoid double-pooling.
        already_pooled = bool(
            getattr(model, 'already_pooled', False)
            or getattr(getattr(model, '_orig_mod', None), 'already_pooled', False)
        )
        if self.matrix_embed or already_pooled:
            pooler = None
        else:
            print_message(f'Pooling types: {self.pooling_types}')
            pooler = Pooler(self.pooling_types)

        def _get_embeddings(
                residue_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                attentions: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
            if residue_embeddings.ndim == 2 or self.matrix_embed or already_pooled:
                return residue_embeddings
            else:
                return pooler(emb=residue_embeddings, attention_mask=attention_mask, attentions=attentions)

        to_embed = sorted(to_embed, key=len, reverse=True)
        dataset = SimpleProteinDataset(to_embed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=dataloader_generator(get_global_seed())
        )

        sql_queue = None
        sql_writer_thread = None
        if self.sql:
            conn = sqlite3.connect(save_path, timeout=30, check_same_thread=False)
            c = conn.cursor()
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('PRAGMA busy_timeout=30000')
            c.execute('PRAGMA synchronous=OFF')
            c.execute('PRAGMA cache_size=-64000')
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')

            sql_writer = _SQLWriter(conn)
            sql_writer.__enter__()

        for i, batch in _make_embedding_progress(dataloader, self.padding):
            seqs = to_embed[i * self.batch_size:(i + 1) * self.batch_size]
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
            elif 'sequence_ids' in batch:
                attention_mask = (batch['sequence_ids'] != -1).long().to(device)
            else:
                attention_mask = torch.ones_like(batch['input_ids'], device=device)
            hidden_kwargs = {}
            if self.hidden_state_index != -1:
                hidden_kwargs['output_hidden_states'] = True
                hidden_kwargs['hidden_state_index'] = self.hidden_state_index

            with torch.autocast(device.type, dtype=self.embed_dtype, enabled=self.autocast):
                if 'parti' in self.pooling_types and not already_pooled:
                    try:
                        residue_embeddings, attentions = model(
                            **batch,
                            output_attentions=True,
                            **hidden_kwargs,
                        )
                        embeddings = _get_embeddings(residue_embeddings, attention_mask=attention_mask, attentions=attentions).cpu()
                    except Exception as e:
                        print_message(f"Error in parti pooling: {e}\nDefaulting to mean pooling")
                        self.pooling_types = ['mean']
                        pooler = Pooler(self.pooling_types)
                        residue_embeddings = model(**batch, **hidden_kwargs)
                        embeddings = _get_embeddings(residue_embeddings, attention_mask=attention_mask).cpu()
                else:
                    residue_embeddings = model(**batch, **hidden_kwargs)
                    embeddings = _get_embeddings(residue_embeddings, attention_mask=attention_mask).cpu()

            if self.sql:
                embeddings = embeddings.to(self.embed_dtype)
                if self.matrix_embed:
                    batch_rows = []
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                        batch_rows.append((seq, tensor_to_embedding_blob(emb[mask.bool()])))
                else:
                    blobs = batch_tensor_to_blobs(embeddings)
                    batch_rows = list(zip(seqs, blobs))
                sql_writer.write_batch(batch_rows)
            else:
                for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                    if self.matrix_embed:
                        emb = emb[mask.bool()]
                    embeddings_dict[seq] = emb.to(self.embed_dtype)

        if self.sql:
            sql_writer.__exit__(None, None, None)
            conn.close()
            return embeddings_dict
        
        if self.save_embeddings:
            print_message(f"Saving embeddings to {save_path}")
            torch.save(embeddings_dict, save_path)
            
        return embeddings_dict

    def _embed_sequences_multi_gpu(
            self,
            to_embed: List[str],
            save_path: str,
            dispatch_name: str,
            model_path: Optional[str],
            embeddings_dict: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        n_gpus = torch.cuda.device_count()
        to_embed = sorted(to_embed, key=len, reverse=True)
        chunk_size = math.ceil(len(to_embed) / n_gpus)
        shards = [to_embed[i * chunk_size:(i + 1) * chunk_size] for i in range(n_gpus)]
        shards = [s for s in shards if len(s) > 0]
        actual_gpus = len(shards)
        print_message(f"Multi-GPU: splitting {len(to_embed)} sequences across {actual_gpus} GPUs")

        if self.sql:
            os.makedirs(self.embedding_save_dir, exist_ok=True)
            conn = sqlite3.connect(save_path, timeout=30)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            conn.close()

        result_dicts: Dict[int, Dict[str, torch.Tensor]] = mp.Manager().dict()

        def _worker(rank: int, shard: List[str]) -> None:
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
            model_obj, tokenizer = get_base_model(dispatch_name, dtype=self.model_dtype, model_path=model_path)
            model_obj = model_obj.to(device).eval()
            dynamic = self.padding == 'longest'
            model_obj = maybe_compile(model_obj, dynamic=dynamic)
            collate_fn = build_collator(tokenizer, padding=self.padding, max_length=self.max_length)

            if self.matrix_embed:
                pooler = None
            else:
                pooler = Pooler(self.pooling_types)

            def _get_embeddings(residue_embeddings, attention_mask=None, attentions=None):
                if residue_embeddings.ndim == 2 or self.matrix_embed:
                    return residue_embeddings
                return pooler(emb=residue_embeddings, attention_mask=attention_mask, attentions=attentions)

            dataset = SimpleProteinDataset(shard)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=2 if self.num_workers > 0 else None,
                collate_fn=collate_fn,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=dataloader_generator(get_global_seed()),
            )

            local_dict = {}
            worker_sql_writer = None
            sql_conn = None
            if self.sql:
                sql_conn = sqlite3.connect(save_path, timeout=30, check_same_thread=False)
                sql_c = sql_conn.cursor()
                sql_c.execute('PRAGMA journal_mode=WAL')
                sql_c.execute('PRAGMA busy_timeout=30000')
                sql_c.execute('PRAGMA synchronous=OFF')
                sql_c.execute('PRAGMA cache_size=-64000')
                worker_sql_writer = _SQLWriter(sql_conn)
                worker_sql_writer.__enter__()

            with torch.inference_mode():
                for i, batch_data in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    desc=f'GPU {rank}',
                    position=rank,
                ):
                    seqs = shard[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_data = {k: v.to(device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                    if 'attention_mask' in batch_data:
                        attention_mask = batch_data['attention_mask']
                    elif 'sequence_ids' in batch_data:
                        attention_mask = (batch_data['sequence_ids'] != -1).long().to(device)
                    else:
                        attention_mask = torch.ones_like(batch_data['input_ids'], device=device)
                    hidden_kwargs = {}
                    if self.hidden_state_index != -1:
                        hidden_kwargs['output_hidden_states'] = True
                        hidden_kwargs['hidden_state_index'] = self.hidden_state_index

                    with torch.autocast(device.type, dtype=self.embed_dtype, enabled=self.autocast):
                        residue_embeddings = model_obj(**batch_data, **hidden_kwargs)
                        embeddings = _get_embeddings(residue_embeddings, attention_mask=attention_mask).cpu()

                    if self.sql:
                        embeddings = embeddings.to(self.embed_dtype)
                        if self.matrix_embed:
                            batch_rows = []
                            for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                                batch_rows.append((seq, tensor_to_embedding_blob(emb[mask.bool()])))
                        else:
                            blobs = batch_tensor_to_blobs(embeddings)
                            batch_rows = list(zip(seqs, blobs))
                        worker_sql_writer.write_batch(batch_rows)
                    else:
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                            if self.matrix_embed:
                                emb = emb[mask.bool()]
                            local_dict[seq] = emb.to(self.embed_dtype)

            if self.sql:
                worker_sql_writer.__exit__(None, None, None)
                sql_conn.close()
            else:
                result_dicts[rank] = local_dict

        processes = []
        mp.set_start_method('spawn', force=True)
        for rank, shard in enumerate(shards):
            p = mp.Process(target=_worker, args=(rank, shard))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if not self.sql:
            for rank in range(actual_gpus):
                embeddings_dict.update(result_dicts[rank])
            if self.save_embeddings:
                print_message(f"Saving embeddings to {save_path}")
                torch.save(embeddings_dict, save_path)

        return embeddings_dict

    def __call__(self, model_name: str, model_type: str = None, model_path: str = None):
        dispatch_name = model_type or model_name
        if is_atlas_ppi_model_name(dispatch_name):
            requested_kind = self._native_atlas_embedding_kind()
            self.pooling_types = [requested_kind]
            self.args.pooling_types = self.pooling_types

        if self.download_embeddings:
            self._download_embeddings(model_name)

        if self.device.type == 'cpu':
            warnings.warn("Downloading embeddings is recommended for CPU usage - Embedding on CPU will be extremely slow!")
        to_embed, save_path, embeddings_dict = self._read_embeddings_from_disk(model_name)

        if len(to_embed) > 0:
            print_message(f"Embedding {len(to_embed)} sequences with {model_name}")

            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if self.multi_gpu and n_gpus > 1 and not is_atlas_ppi_model_name(dispatch_name):
                return self._embed_sequences_multi_gpu(
                    to_embed, save_path, dispatch_name, model_path, embeddings_dict,
                )
            else:
                model, tokenizer = get_base_model(dispatch_name, dtype=self.model_dtype, model_path=model_path)
                return self._embed_sequences(
                    to_embed,
                    save_path,
                    model,
                    tokenizer,
                    embeddings_dict,
                    model_name=model_name,
                )
        else:
            print_message(f"No sequences to embed with {model_name}")
            return embeddings_dict


if __name__ == '__main__':
    ### Embed all supported datasets with all supported models
    # py -m embedder
    import argparse
    from huggingface_hub import upload_file, login
    from data.supported_datasets import vector_benchmark
    from data.data_mixin import DataArguments, DataMixin
    from base_models.get_base_models import BaseModelArguments, get_base_model
    from seed_utils import set_global_seed

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default=None, help='Huggingface token')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embed_dtype', type=str, default='float16')
    parser.add_argument('--model_names', nargs='+', default=['standard'])
    parser.add_argument('--models_to_skip', nargs='+', default=[], help='When checking for existing embeddings, skip these models.')
    parser.add_argument('--embedding_save_dir', type=str, default='embeddings')
    parser.add_argument('--download_dir', type=str, default='Synthyra/vector_embeddings')
    parser.add_argument('--embedding_pooling_types', nargs='+', default=['mean', 'var'], help='Pooling types for embeddings.')
    parser.add_argument('--embedding_hidden_state_index', type=int, default=-1, help='Hidden-state tuple index for embeddings. -1 uses the final hidden state.')
    args = parser.parse_args()

    chosen_seed = set_global_seed()

    if args.token is not None:
        login(args.token)

    if args.embed_dtype == 'float16':
        dtype = torch.float16
    elif args.embed_dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.embed_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid embedding dtype: {args.embed_dtype}")

    # Get data    
    data_args = DataArguments(
        data_names=vector_benchmark,
        max_length=1024,
        trim=False
    )
    all_seqs = DataMixin(data_args).get_data()[1]

    # Embed for each model
    model_args = BaseModelArguments(model_names=args.model_names)
    for model_name in model_args.model_names:

        embedder_args = EmbeddingArguments(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download_embeddings=model_name not in args.models_to_skip,
            matrix_embed=False,
            embedding_pooling_types=args.embedding_pooling_types,
            save_embeddings=True,
            embed_dtype=dtype,
            sql=False,
            embedding_save_dir='embeddings',
            embedding_hidden_state_index=args.embedding_hidden_state_index,
        )
        embedder = Embedder(embedder_args, all_seqs)

        _ = embedder(model_name)
        filename = get_embedding_filename(
            model_name,
            False,
            embedder_args.pooling_types,
            'pth',
            embedder_args.hidden_state_index,
        )
        save_path = os.path.join(args.embedding_save_dir, filename)
        
        compressed_path = f"{save_path}.gz"
        print(f"Compressing {save_path} to {compressed_path}")
        with open(save_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        upload_path = compressed_path
        path_in_repo = f'embeddings/{filename}.gz'
            
        upload_file(
             path_or_fileobj=upload_path,
            path_in_repo=path_in_repo,
            repo_id=args.download_dir,
            repo_type='dataset'
        )

    print('Done')

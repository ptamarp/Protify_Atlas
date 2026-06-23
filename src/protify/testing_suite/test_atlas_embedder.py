import os
import sys
import sqlite3
from types import SimpleNamespace

import torch
import torch.nn as nn


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROTIFY_ROOT, SRC_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from src.protify.base_models.atlas import ATLAS_EMBEDDING_KINDS, atlas_kind_is_matrix
from src.protify.embedder import (
    Embedder,
    EmbeddingArguments,
    get_atlas_embedding_filename,
    get_embedding_filename,
)


class FakeAtlasModel(nn.Module):
    atlas_native_embedding = True

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(a_input_size=2, b_input_size=2)
        self.embed_all_calls = 0

    def embed_all_sequences(self, sequences):
        self.embed_all_calls += 1
        concat = []
        for sequence in sequences:
            emb = torch.zeros(len(sequence) + 2, 4)
            emb[:, :2] = float(len(sequence))
            emb[:, 2:] = float(len(sequence) + 10)
            concat.append(emb)
        return {"concat": concat}


def test_atlas_embedding_filename_uses_exact_native_kind():
    filename = get_embedding_filename("Atlas-PPI-auto", False, ["pooled_concat"])

    assert filename == "Atlas-PPI-auto_False_pooled_concat.pth"


def test_atlas_embedding_filename_rejects_invalid_kind():
    try:
        get_embedding_filename("Atlas-PPI-auto", False, ["mean", "var"])
    except AssertionError as exc:
        assert "exactly one embedding kind" in str(exc)
    else:
        raise AssertionError("Expected invalid Atlas embedding kind to raise")


def test_atlas_native_embedder_writes_all_pth_kinds_from_one_call(tmp_path):
    sequences = ["MKT", "GHHH"]
    args = EmbeddingArguments(
        embedding_batch_size=2,
        matrix_embed=False,
        embedding_pooling_types=["pooled_concat"],
        save_embeddings=True,
        embed_dtype=torch.float32,
        embedding_save_dir=str(tmp_path),
    )
    embedder = Embedder(args, sequences)
    model = FakeAtlasModel()
    save_path = tmp_path / get_embedding_filename("Atlas-PPI-auto", False, ["pooled_concat"])

    embeddings = embedder._embed_sequences(
        sequences,
        str(save_path),
        model,
        tokenizer=None,
        embeddings_dict={},
        model_name="Atlas-PPI-auto",
    )

    assert model.embed_all_calls == 1
    assert set(embeddings) == set(sequences)
    assert torch.equal(embeddings["MKT"], torch.tensor([3.0, 3.0, 13.0, 13.0]))
    for embedding_kind in ATLAS_EMBEDDING_KINDS:
        path = tmp_path / get_atlas_embedding_filename("Atlas-PPI-auto", embedding_kind)
        assert path.exists(), f"Missing Atlas cache for {embedding_kind}"
        cached = torch.load(path, map_location="cpu", weights_only=True)
        assert set(cached) == set(sequences)
        if atlas_kind_is_matrix(embedding_kind):
            assert cached["MKT"].shape == (3, 4 if embedding_kind == "concat" else 2)
        else:
            assert cached["MKT"].ndim == 1


def test_atlas_sql_embedder_writes_all_kind_databases_from_one_call(tmp_path):
    sequences = ["MKT"]
    args = EmbeddingArguments(
        embedding_batch_size=1,
        matrix_embed=False,
        embedding_pooling_types=["pooled_concat"],
        sql=True,
        embed_dtype=torch.float32,
        embedding_save_dir=str(tmp_path),
    )
    embedder = Embedder(args, sequences)
    model = FakeAtlasModel()

    embedder._embed_sequences(
        sequences,
        str(tmp_path / get_embedding_filename("Atlas-PPI-auto", False, ["pooled_concat"], extension="db")),
        model,
        tokenizer=None,
        embeddings_dict={},
        model_name="Atlas-PPI-auto",
    )

    assert model.embed_all_calls == 1
    for embedding_kind in ATLAS_EMBEDDING_KINDS:
        path = tmp_path / get_atlas_embedding_filename("Atlas-PPI-auto", embedding_kind, extension="db")
        assert path.exists(), f"Missing Atlas database for {embedding_kind}"
        with sqlite3.connect(path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert count == 1

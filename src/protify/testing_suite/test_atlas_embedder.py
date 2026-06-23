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
        self.embed_sequences_calls = []

    def embed_sequences(self, sequences, embedding_kind=None):
        self.embed_sequences_calls.append(embedding_kind)
        assert embedding_kind is None
        a, b, concat = [], [], []
        for sequence in sequences:
            a_emb = torch.full((len(sequence), 2), float(len(sequence)))
            b_emb = torch.full((len(sequence), 2), float(len(sequence) + 10))
            a.append(a_emb)
            b.append(b_emb)
            concat.append(torch.cat([a_emb, b_emb], dim=-1))
        return {
            "a": a,
            "b": b,
            "concat": concat,
            "pooled_a": torch.stack([embedding.mean(dim=0) for embedding in a]),
            "pooled_b": torch.stack([embedding.mean(dim=0) for embedding in b]),
            "pooled_concat": torch.stack([embedding.mean(dim=0) for embedding in concat]),
        }


class FakeAtlasVectorProjectionModel(nn.Module):
    atlas_native_embedding = True

    def __init__(self):
        super().__init__()
        self.embed_sequences_calls = []

    def embed_sequences(self, sequences, embedding_kind=None):
        self.embed_sequences_calls.append(embedding_kind)
        assert embedding_kind is None
        a = torch.stack([
            torch.full((2,), float(len(sequence)))
            for sequence in sequences
        ])
        b = torch.stack([
            torch.full((2,), float(len(sequence) + 10))
            for sequence in sequences
        ])
        concat = torch.cat([a, b], dim=-1)
        return {
            "a": a,
            "b": b,
            "concat": concat,
            "pooled_a": a,
            "pooled_b": b,
            "pooled_concat": concat,
        }


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

    assert model.embed_sequences_calls == [None]
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

    assert model.embed_sequences_calls == [None]
    for embedding_kind in ATLAS_EMBEDDING_KINDS:
        path = tmp_path / get_atlas_embedding_filename("Atlas-PPI-auto", embedding_kind, extension="db")
        assert path.exists(), f"Missing Atlas database for {embedding_kind}"
        with sqlite3.connect(path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert count == 1


def test_atlas_matrix_views_accept_vector_projections(tmp_path):
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
    model = FakeAtlasVectorProjectionModel()

    embeddings = embedder._embed_sequences(
        sequences,
        str(tmp_path / get_embedding_filename("Atlas-PPI-auto", False, ["pooled_concat"])),
        model,
        tokenizer=None,
        embeddings_dict={},
        model_name="Atlas-PPI-auto",
    )

    assert model.embed_sequences_calls == [None]
    assert torch.equal(embeddings["MKT"], torch.tensor([3.0, 3.0, 13.0, 13.0]))
    for embedding_kind in ("a", "b", "concat"):
        path = tmp_path / get_atlas_embedding_filename("Atlas-PPI-auto", embedding_kind)
        cached = torch.load(path, map_location="cpu", weights_only=True)
        expected_width = 4 if embedding_kind == "concat" else 2
        assert cached["MKT"].shape == (1, expected_width)

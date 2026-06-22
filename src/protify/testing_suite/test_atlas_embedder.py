import os
import sys

import torch
import torch.nn as nn


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROTIFY_ROOT, SRC_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from src.protify.embedder import Embedder, EmbeddingArguments, get_embedding_filename


class FakeAtlasModel(nn.Module):
    atlas_native_embedding = True

    def __init__(self, matrix: bool = False):
        super().__init__()
        self.matrix = matrix
        self.embedding_kinds = []

    def embed_sequences(self, sequences, embedding_kind="pooled_concat"):
        self.embedding_kinds.append(embedding_kind)
        if self.matrix:
            return [
                torch.full((len(sequence) + 2, 4), float(index + 1))
                for index, sequence in enumerate(sequences)
            ]
        return torch.stack([
            torch.full((4,), float(len(sequence)))
            for sequence in sequences
        ])


def test_atlas_embedding_filename_uses_native_kind_for_default_pooling():
    filename = get_embedding_filename("Atlas-PPI-auto", False, ["mean", "var"])

    assert filename == "Atlas-PPI-auto_False_pooled_concat.pth"


def test_atlas_native_embedder_uses_pooled_concat(tmp_path):
    sequences = ["MKT", "GHHH"]
    args = EmbeddingArguments(
        embedding_batch_size=2,
        matrix_embed=False,
        embedding_pooling_types=["mean", "var"],
        save_embeddings=True,
        embed_dtype=torch.float32,
        embedding_save_dir=str(tmp_path),
    )
    embedder = Embedder(args, sequences)
    model = FakeAtlasModel(matrix=False)
    save_path = tmp_path / get_embedding_filename("Atlas-PPI-auto", False, ["mean", "var"])

    embeddings = embedder._embed_sequences(sequences, str(save_path), model, tokenizer=None, embeddings_dict={})

    assert model.embedding_kinds == ["pooled_concat"]
    assert save_path.exists()
    assert set(embeddings) == set(sequences)
    assert torch.equal(embeddings["MKT"], torch.full((4,), 3.0))
    assert torch.equal(embeddings["GHHH"], torch.full((4,), 4.0))


def test_atlas_matrix_embedder_uses_concat_and_trims_to_sequence_length(tmp_path):
    sequences = ["MKT"]
    args = EmbeddingArguments(
        embedding_batch_size=1,
        matrix_embed=True,
        embedding_pooling_types=["mean"],
        save_embeddings=False,
        embed_dtype=torch.float32,
        embedding_save_dir=str(tmp_path),
    )
    embedder = Embedder(args, sequences)
    model = FakeAtlasModel(matrix=True)

    embeddings = embedder._embed_sequences(
        sequences,
        str(tmp_path / get_embedding_filename("Atlas-PPI-auto", True, ["mean"])),
        model,
        tokenizer=None,
        embeddings_dict={},
    )

    assert model.embedding_kinds == ["concat"]
    assert embeddings["MKT"].shape == (3, 4)

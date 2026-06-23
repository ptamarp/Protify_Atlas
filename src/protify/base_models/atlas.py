import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


ATLAS_PPI_AUTO_PRESET = "Atlas-PPI-auto"
ATLAS_PPI_AUTO_PATH = "GleghornLab/Atlas-PPI-auto"
ATLAS_MATRIX_EMBEDDING_KINDS = ("a", "b", "concat")
ATLAS_POOLED_EMBEDDING_KINDS = ("pooled_a", "pooled_b", "pooled_concat")
ATLAS_EMBEDDING_KINDS = ATLAS_MATRIX_EMBEDDING_KINDS + ATLAS_POOLED_EMBEDDING_KINDS

presets = {
    ATLAS_PPI_AUTO_PRESET: ATLAS_PPI_AUTO_PATH,
}


def is_atlas_ppi_model_name(model_name: str) -> bool:
    return model_name.lower() in {
        ATLAS_PPI_AUTO_PRESET.lower(),
        ATLAS_PPI_AUTO_PATH.lower(),
        "atlas",
        "atlas-ppi",
        "atlas_ppi",
    }


def atlas_embedding_kind(matrix_embed: bool, pooling_types: Optional[List[str]] = None) -> str:
    """Resolve Protify embedding settings to Atlas' native embedding kinds."""
    if not pooling_types:
        return "concat" if matrix_embed else "pooled_concat"
    assert len(pooling_types) == 1, (
        "Atlas-PPI-auto expects exactly one embedding kind from "
        f"{', '.join(ATLAS_EMBEDDING_KINDS)}."
    )
    requested = pooling_types[0]
    assert requested in ATLAS_EMBEDDING_KINDS, (
        f"Invalid Atlas-PPI-auto embedding kind: {requested}. "
        f"Expected one of {', '.join(ATLAS_EMBEDDING_KINDS)}."
    )
    if matrix_embed:
        assert requested in ATLAS_MATRIX_EMBEDDING_KINDS, (
            f"Atlas embedding kind {requested} is pooled; use one of "
            f"{', '.join(ATLAS_MATRIX_EMBEDDING_KINDS)} with --matrix_embed."
        )
    else:
        assert requested in ATLAS_POOLED_EMBEDDING_KINDS, (
            f"Atlas embedding kind {requested} is token-level; use --matrix_embed "
            f"or choose one of {', '.join(ATLAS_POOLED_EMBEDDING_KINDS)}."
        )
    return requested


def atlas_kind_is_matrix(embedding_kind: str) -> bool:
    assert embedding_kind in ATLAS_EMBEDDING_KINDS, f"Invalid Atlas embedding kind: {embedding_kind}"
    return embedding_kind in ATLAS_MATRIX_EMBEDDING_KINDS


class AtlasPPITokenizer:
    """Placeholder tokenizer for code paths that carry a tokenizer object.

    Atlas handles token preparation internally through ``embed_sequences`` and
    does not expose the standard tokenizer contract used by other Protify PLMs.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Atlas-PPI-auto embeddings are generated with model.embed_sequences(); "
            "standard tokenizer-based training is not supported for this model."
        )


class AtlasPPIForEmbedding(nn.Module):
    """Thin adapter around GleghornLab/Atlas-PPI-auto for Protify embedding."""

    atlas_native_embedding = True

    def __init__(self, model_path: str = ATLAS_PPI_AUTO_PATH, dtype: torch.dtype = None):
        super().__init__()
        kwargs = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            kwargs["token"] = token
        self.model = AutoModel.from_pretrained(model_path, **kwargs)
        self.model_path = model_path

    @property
    def config(self):
        return self.model.config

    def embed_sequences(self, sequences: List[str], embedding_kind: Optional[str] = None):
        if embedding_kind is None:
            return self.model.embed_sequences(sequences)
        return self.model.embed_sequences(sequences, embedding_kind=embedding_kind)

    def embed_all_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Return all Atlas embedding kinds from one native sequence-embedding pass."""
        embeddings = self.model.embed_sequences(sequences)
        assert isinstance(embeddings, dict), (
            "Atlas-PPI-auto embed_sequences(sequences) must return a dictionary of embedding views."
        )
        missing = set(ATLAS_EMBEDDING_KINDS) - set(embeddings)
        assert not missing, f"Atlas embed_sequences did not return required views: {sorted(missing)}"
        return embeddings

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def build_atlas_ppi_model(
        preset: str,
        masked_lm: bool = False,
        dtype: torch.dtype = None,
        model_path: str = None,
        **kwargs,
):
    assert not masked_lm, "Atlas-PPI-auto does not provide a masked-language-model embedding path."
    model_path = model_path or presets.get(preset, ATLAS_PPI_AUTO_PATH)
    model = AtlasPPIForEmbedding(model_path, dtype=dtype).eval()
    return model, AtlasPPITokenizer()


def get_atlas_ppi_tokenizer(preset: str, model_path: str = None):
    return AtlasPPITokenizer()

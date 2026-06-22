import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


ATLAS_PPI_AUTO_PRESET = "Atlas-PPI-auto"
ATLAS_PPI_AUTO_PATH = "GleghornLab/Atlas-PPI-auto"

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
    pooling_types = pooling_types or []
    if len(pooling_types) == 1:
        requested = pooling_types[0]
        if matrix_embed and requested in {"a", "b", "concat"}:
            return requested
        if not matrix_embed and requested in {"pooled_a", "pooled_b", "pooled_concat"}:
            return requested
        if not matrix_embed and requested in {"a", "b", "concat"}:
            return f"pooled_{requested}"
    return "concat" if matrix_embed else "pooled_concat"


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

    def embed_sequences(self, sequences: List[str], embedding_kind: str = "pooled_concat"):
        return self.model.embed_sequences(sequences, embedding_kind=embedding_kind)

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

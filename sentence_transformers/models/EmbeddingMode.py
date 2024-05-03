from torch import Tensor
from torch import nn
from typing import Dict


class EmbeddingMode(nn.Module):
    """
    This layer selects which of the doc or query embedding is set as the sentence_embedding.
    """

    def __init__(self, mode: str = "doc"):
        self.mode = mode
        super(EmbeddingMode, self).__init__()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ("doc", "query"), "EmbeddingMode only supports `doc` or `query` values."
        self._mode = value

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": features[f"{self.mode}_embedding"]})
        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(input_path):
        return EmbeddingMode()

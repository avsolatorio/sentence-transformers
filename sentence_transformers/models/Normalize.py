from torch import Tensor
from torch import nn
from typing import Dict
import torch.nn.functional as F


class Normalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]):
        for key in ("sentence_embedding", "query_embedding"):
            if key in features:
                features.update({key: F.normalize(features[key], p=2, dim=1)})

        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(input_path):
        return Normalize()

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import Tensor, nn


class RandomProjection(nn.Module):
    """
    Feed-forward function with activation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer. Can be used to generate deep averaging networks (DAN).

    Args:
        in_features: Size of the input dimension
        out_features: Output size
        bias: Add a bias vector
        activation_function: Pytorch activation function applied on
            output
        init_weight: Initial value for the matrix of the linear layer
        init_bias: Initial value for the bias of the linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int = 42,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        torch.manual_seed(self.seed)
        # Yes, the order of the dimensions is correct. The in_features is the number of columns in the matrix, the out_features the number of rows. This is to prevent transposing the matrix in the forward pass.
        self.projection = F.normalize(
            torch.randn(self.out_features, self.in_features), p=2, dim=-1
        )
        # Reset the seed
        torch.seed()

    def forward(self, features: dict[str, Tensor]):
        # We use tensordot to compute the cosine similarity between the token embeddings and the projection matrix.
        # This is faster than the cosine similarity computation in the forward method.
        # We then take the mean over the num_tokens dimension, resulting in a tensor of shape [bsz, out_features]

        # We normalize the token embeddings and the projection matrix to have a norm of 1. This is necessary to compute the cosine similarity.
        # Since the token embeddings has shape [bsz, num_tokens, embedding_dim], we normalize it along the last dimension. The projection is already normalized.
        # We then compute the tensordot between the normalized token embeddings and the projection matrix along the last dimension of the token embeddings (dim=2) and the last dimension of the projection matrix (dim=1).
        # This results in a tensor of shape [bsz, num_tokens, out_features]
        # We take the mean over the num_tokens dimension, resulting in a tensor of shape [bsz, out_features]

        features.update(
            {
                "sentence_embedding": torch.tensordot(
                    F.normalize(features["token_embeddings"], p=2, dim=-1),
                    self.projection,
                    dims=([2], [1]),
                ).mean(dim=1)
            }
        )

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "seed": self.seed,
        }

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(
                self.state_dict(), os.path.join(output_path, "pytorch_model.bin")
            )

    def __repr__(self):
        return f"RandomProjection({self.get_config_dict()})"

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = RandomProjection(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"),
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
        return model

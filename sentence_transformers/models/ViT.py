from transformers import ViTConfig, ViTModel
import torch
from torch import nn
from typing import List, Union, Tuple
import os
import json


class ViT(nn.Module):
    """ViT model for embeddings"""

    def __init__(
        self,
        in_word_embedding_dimension: int,
        image_height: int ,
        patch_width: int,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        add_pooling_layer: bool = False,
        num_channels: int = 1,
    ):
        nn.Module.__init__(self)
        self.config_keys = ["in_word_embedding_dimension", "image_height", "patch_width", "intermediate_size", "num_hidden_layers", "num_attention_heads", "hidden_size", "add_pooling_layer", "num_channels"]

        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.image_height = image_height
        self.patch_width = patch_width
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.add_pooling_layer = add_pooling_layer
        self.num_channels = num_channels

        self.config = self.get_vit_config()
        self.vit = ViTModel(self.config, add_pooling_layer=add_pooling_layer)

        self.embeddings_dimension = hidden_size

    def get_vit_config(self):
        assert self.in_word_embedding_dimension % self.image_height == 0, "in_word_embedding_dimension must be divisible by image_height"
        image_size = (self.image_height, self.in_word_embedding_dimension // self.image_height)
        self.image_size = image_size

        assert image_size[1] % self.patch_width == 0, "image_width must be divisible by patch_width"
        patch_size = (self.image_height, self.patch_width)
        self.patch_size = patch_size

        # TODO: consider implementing support for num_channels > 1 by permuting the input tensor

        self.vit_config = ViTConfig(
            image_size=image_size , # (4, 192),  # 768 -> 4x192 image
            patch_size=patch_size,  # (4, 8),  # Generates (CLS + 24) 32-dim embeddings as input to the transformer
            num_channels=self.num_channels,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=0.1,
        )

        return self.vit_config

    def forward(self, features):
        embedding = self.vit(features["sentence_embedding"].view(-1, self.num_channels, self.image_size[0], self.image_size[1]))

        features.update({"vit_token_embeddings": embedding.last_hidden_state})

        if self.add_pooling_layer:
            features.update({"sentence_embedding": embedding.pooler_output})

        else:
            features.update({"sentence_embedding": embedding.last_hidden_state[:, 0]})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str, **kwargs) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, "vit_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "vit_config.json"), "r") as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        model = ViT(**config)
        model.load_state_dict(weights)
        return model

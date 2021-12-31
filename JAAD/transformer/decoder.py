# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .layer_norm import LayerNorm
from .functional import clones


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x,memory,embed_ego,embed_ped,src_mask,spd_mask,ped_mask):
        for layer in self.layers:
            x= layer(x, memory,embed_ego,embed_ped,src_mask,spd_mask,ped_mask)

        return self.norm(x)

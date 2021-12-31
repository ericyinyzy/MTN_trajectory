# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .functional import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
    def forward(self, x,embed_spd,src_mask,obd_enc_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        t=0
        m=embed_spd
        for layer in self.layers:
            x,m= layer(x,m,src_mask,obd_enc_mask,t)
            t+=1
        return self.norm(x),self.norm(m)#,self.norm(m1)

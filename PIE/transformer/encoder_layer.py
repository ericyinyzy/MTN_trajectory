# -*- coding: utf-8 -*-
# date: 2018-11-30 15:27
import torch.nn as nn

from .functional import clones
from .sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn,src_attn,cross_attn,feed_forward, dropout):
        #d_model, c(attn), c(ff), dropout
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn    #512
        self.src_attn=src_attn
        self.cross_attn=cross_attn
        self.feed_forward = feed_forward   #
        self.sublayer = clones(SublayerConnection(size, dropout), 5)
        self.size = size   #512

    def forward(self, x,embed,src_mask,obd_enc_mask,st):
        """
        Follow Figure 1 (left) for connections.

        """
        m=embed
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask),st=st)
        pre_x = x
        x = self.sublayer[2](x, lambda x: self.src_attn(x, m, m, obd_enc_mask))
        m=self.sublayer[3](m, lambda m: self.cross_attn(m, pre_x, pre_x, src_mask))

        return self.sublayer[1](x, self.feed_forward),self.sublayer[4](m,self.feed_forward)

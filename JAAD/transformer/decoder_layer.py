# -*- coding: utf-8 -*-
# date: 2018-11-30 15:41
import torch.nn as nn
import torch
from .functional import clones
from .sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, ped_attn, src_attn,spd_attn,feed_forward,dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.ped_attn = ped_attn
        self.src_attn = src_attn
        self.spd_attn=spd_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)

    def forward(self, x,memory,embed_ego,embed_ped,src_mask,spd_mask,ped_mask):
        """
        Follow Figure 1 (right) for connections.
        """
        m = memory
        spd=embed_ego
        ped=embed_ped
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), st=st)        #
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[3](x,lambda x: self.spd_attn(x,spd, spd, spd_mask), sublayer1 =lambda x: self.ped_attn(x, ped, ped, ped_mask), sublayer2 = lambda x: self.src_attn(x, m, m, src_mask))
        x= self.sublayer[2](x, self.feed_forward)
        return x
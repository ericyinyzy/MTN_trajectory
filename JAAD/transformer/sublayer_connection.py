# -*- coding: utf-8 -*-
# date: 2018-11-30 15:17
import torch.nn as nn

from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer,st=1,sublayer1=None,sublayer2=None):
        """
        Apply residual connection to any sublayer with the same size.
        """
        if st==0:
            return x + self.dropout(sublayer(x))
        if sublayer1==None:
            return self.self_attn(x,sublayer)
        else:
            return self.src_attn(x,sublayer,sublayer1,sublayer2)
    def self_attn(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    def src_attn(self,x,sublayer,sublayer1,sublayer2):
        b=sublayer(self.norm(x))
        b1=sublayer1(self.norm(x))
        b2=sublayer2(self.norm(x))



        # exit()
        return x + self.dropout(b)+self.dropout(b1)+self.dropout(b2)

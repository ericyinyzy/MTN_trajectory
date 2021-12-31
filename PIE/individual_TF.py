import torch.nn as nn
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
import copy
import math


class IndividualTF(nn.Module):
    def __init__(self, inp_l,enc_inp_size, dec_inp_size, dec_out_size, N=1,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        generator=Generator(d_model, dec_out_size)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn),c(attn),c(attn),c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model),c(position)),
            nn.Sequential(LinearEmbedding_sp(inp_l,d_model)),
            nn.Sequential(LinearEmbedding_sp(2*(inp_l-1),d_model)),
            nn.Sequential(LinearEmbedding_sp(2*(inp_l-1),d_model)),
            generator)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,residual,inp,obd_spd,ego_flow,ped_flow,tgt,src_mask,obd_enc_mask,spd_mask,ped_mask):
        output=self.model(inp,obd_spd,ego_flow,ped_flow,tgt,src_mask,obd_enc_mask,spd_mask,ped_mask)
        out_lane=self.model.generator(output)
        output=out_lane+residual
        return output

class LinearEmbedding_sp(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding_sp, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj=nn.Linear(d_model,out_size)

    def forward(self, x):
        return self.proj(x)



# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn
import torch
from torch.nn.functional import relu

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed,enc_extra_embed_ego,enc_extra_embed_ped,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.enc_extra_embed_II=enc_extra_embed_ego
        self.enc_extra_embed_III = enc_extra_embed_ped
        self.generator = generator

    def forward(self, src,enc_extra_ego,enc_extra_ped,tgt,src_mask,spd_mask,ped_mask):
        """
        Take in and process masked src and target sequences.
        """
        encode=self.encode(src,src_mask)
        embed_ego = self.enc_extra_embed_II(enc_extra_ego.permute(0, 2, 1))  #embedding for the optical flow of ego-vehicle
        embed_ped=self.enc_extra_embed_III(enc_extra_ped.permute(0, 2, 1)) #embedding for the optical flow of ped
        decode=self.decode(encode,embed_ego,embed_ped,src_mask, tgt,spd_mask,ped_mask)
        return decode
    def decode_speed(self,src,tgt,src_mask):
        return self.decoder_speed(src,tgt,src_mask)

    def encode(self, src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self, memory,embed_ego,embed_ped,src_mask, tgt, spd_mask,ped_mask):
        t=self.tgt_embed(tgt)
        return self.decoder(t, memory,embed_ego,embed_ped,src_mask,spd_mask,ped_mask)

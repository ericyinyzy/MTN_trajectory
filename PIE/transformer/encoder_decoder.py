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

    def __init__(self, encoder, decoder, src_embed, tgt_embed,enc_extra_embed,enc_extra_embed_II,enc_extra_embed_III,generator):#generator_II):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.enc_extra_embed_I=enc_extra_embed
        self.enc_extra_embed_II=enc_extra_embed_II
        self.enc_extra_embed_III = enc_extra_embed_III
        self.generator = generator

    def forward(self, src,obd_spd,ego_op_flow,ped_op_flow, tgt,src_mask,obd_enc_mask,spd_mask,ped_mask):
        """
        Take in and process masked src and target sequences.
        """
        embed_spd=self.enc_extra_embed_I(obd_spd.permute(0,2,1))
        encode,mix=self.encode(src,embed_spd,src_mask,obd_enc_mask)
        embed_ego_flow = self.enc_extra_embed_II(ego_op_flow.permute(0, 2, 1))
        embed_ped_flow=self.enc_extra_embed_III(ped_op_flow.permute(0, 2, 1))
        decode=self.decode(tgt,encode,mix,embed_ego_flow,embed_ped_flow,src_mask,spd_mask,ped_mask)
        return decode

    def decode_speed(self,src,tgt,src_mask):
        return self.decoder_speed(src,tgt,src_mask)

    def encode(self, src,embed_spd,src_mask,obd_enc_mask):
        return self.encoder(self.src_embed(src),embed_spd,src_mask,obd_enc_mask)

    def decode(self, tgt,memory,embed,embed_ego_flow,embed_ped_flow,enc_mask, spd_mask,ped_mask):
        memory=embed+memory
        t=self.tgt_embed(tgt)
        return self.decoder(t, memory,embed_ego_flow,embed_ped_flow,enc_mask,spd_mask,ped_mask)

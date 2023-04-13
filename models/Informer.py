import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from models.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from utils.masking import TriangularCausalMask, ProbMask
from models.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(configs['enc_in'], configs['d_model'], configs['embed'], configs['freq'], configs['dropout'])
        self.dec_embedding = DataEmbedding(configs['dec_in'], configs['d_model'], configs['embed'], configs['freq'], configs['dropout'])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=configs['output_attention']),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            [
                ConvLayer(configs['d_model'] ) for l in range(configs['e_layers'] - 1)
            ] if configs['distil'] else None,
            norm_layer=torch.nn.LayerNorm(configs['d_model'])
        )

        factor = configs['factor']
        dropout = configs['dropout']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        c_out = configs['c_out']
        activation = configs['activation']
        d_layers = configs['d_layers']
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

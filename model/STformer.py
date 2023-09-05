import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import plot_multivariate_time_series, plot_multivariate_timeseries, oringe_time_series
from .Attentions import SingleStageAttentionLayer, TwoStageAttentionLayer, CrissCrossAtten, CrissCrossAttentionLayer
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding, PatchEmbedding, FlattenHead
from math import ceil
from einops import rearrange, repeat

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn, mask, sigma = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return self.norm2(x + y), attn, mask, sigma

class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()

        # if win_size > 1:
        #     self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        # else:
        #     self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(CrissCrossAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def forward(self, x, x_o,attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape    # x.shape->[128, 55(MSL),9,d_model=128]
        # print('x_Size:',x.shape)

        # if self.merge_layer is not None:
        #     x = self.merge_layer(x)
        series_lis = []
        prior_lis = []
        sigma_lis = []
        for layer in self.encode_layers:
            x, attn, prior, sigma = layer(x,x_o)
            series_lis.append(attn)
            prior_lis.append(prior)
            # sigma_lis.append(sigma)

        return x,  attn, prior, sigma

# class Encoder(nn.Module):
#     def __init__(self, attn_layers, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.norm = norm_layer
#
#     def forward(self, x, attn_mask=None):
#         # x [B, L, D]
#         series_list = []
#         prior_list = []
#         sigma_list = []
#         for attn_layer in self.attn_layers:
#             x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
#             series_list.append(series)
#             prior_list.append(prior)
#             sigma_list.append(sigma)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         return x, series_list, prior_list, sigma_list

class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x,x_o):
        series_list = []
        prior_list = []
        # sigma_list = []
        encode_x = []     # x.shape-> [128,55=dim,9,128]
        encode_x.append(x)

        for block in self.encode_blocks:
            x, attns , prior, sigma= block(x,x_o)
            encode_x.append(x)
            series_list.append(attns)
            prior_list.append(prior)
            # sigma_list.append(sigma)

        return encode_x, series_list, prior_list, sigma #  encode_x.shape->[128, 55(MSL), 9, 128]



class STformer(nn.Module):
    # def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    #              dropout=0.0, activation='gelu', output_attention=True):
    #     super(AnomalyTransformer, self).__init__()
    #     self.output_attention = output_attention
    #
    #     # Encoding
    #     self.embedding = DataEmbedding(enc_in, d_model, dropout)
    #
    #     # Encoder
    #     self.encoder = Encoder(
    #         [
    #             EncoderLayer(
    #                 AttentionLayer(
    #                     AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
    #                     d_model, n_heads),
    #                 d_model,
    #                 d_ff,
    #                 dropout=dropout,
    #                 activation=activation
    #             ) for l in range(e_layers)
    #         ],
    #         norm_layer=torch.nn.LayerNorm(d_model)
    #     )
    #
    #     self.projection = nn.Linear(d_model, c_out, bias=True)
    def __init__(self, configs):
        super(STformer, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        # self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        # self.head_nf = configs.d_model * self.out_seg_num
        self.head_nf = configs.d_model * self.in_seg_num
        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)
        self.embedding = DataEmbedding(self.enc_in, configs.d_model, 0.0)

        # Encoder
        # self.encoder = Encoder(
        #     [
        #         scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,   # heads=8
        #                     1, configs.dropout,
        #                     self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
        #                     ) for l in range(configs.e_layers)
        #     ]
        # )

        self.encoder = Encoder(
            [
                scale_block(configs, 1 , configs.d_model, configs.n_heads, configs.d_ff,   # heads=8
                            1, configs.dropout,
                            self.in_seg_num , configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )



        # print('heads',configs.n_heads)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
        #                                    configs.d_ff, configs.dropout),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             self.seg_len,
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             # activation=configs.activation,
        #         )
        #         for l in range(configs.e_layers + 1)
        #     ],
        # )
        # if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                head_dropout=configs.dropout)
        self.crtir=nn.MSELoss()

    # def forward(self, x):
    #     enc_out = self.embedding(x)
    #     enc_out, series, prior, sigmas = self.encoder(enc_out)
    #     enc_out = self.projection(enc_out)
    #
    #     if self.output_attention:
    #         return enc_out, series, prior, sigmas
    #     else:
    #         return enc_out  # [B, L, D]


    def forward(self, x_enc):  #  x_enc:[B,L,Dim]

        # ser=x_enc.detach().cpu().numpy()
        # plot_multivariate_time_series(np.array(ser[20,:,0:4]))

        x_origin=self.embedding(x_enc)
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))   #x_enc.shape->[7040,9,128] n_vars=55=dim
        # print('n_vars_Size:',n_vars)
        x_enc = rearrange(x_enc,'(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)

        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        # plot_multivariate_timeseries(x_enc.detach().cpu().numpy()[20, 0:4, 0:4, :])


        enc_out, series, prior, sigmas = self.encoder(x_enc,x_origin)  # x_enc.shape->[128,55(MSL),9,128]
        ser=np.array(enc_out)
        # print('ser',ser.shape)
        # print('ser0',ser[0].shape)
        # plot_multivariate_time_series(np.array(ser[0].detach().cpu().numpy()[20,:,:,:]))
        loss_emb=self.crtir(x_enc,enc_out[-1])
        # plot_multivariate_timeseries(enc_out[-1].detach().cpu().numpy()[20, 0:8, :, :])
        # series= [item for sublist in series for item in sublist]
        # prior= [item for sublist in prior for item in sublist]
        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out, series, prior, sigmas ,loss_emb #  dec_out.shape -> [128,len=50,dim=55]


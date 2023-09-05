import torch
import torch.nn as nn
import numpy as np
from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import random
import math

from matplotlib import pyplot as plt
from torchvision import transforms

from tools import multi_heatplot,mahalanobis_distance, plot_heatmap_surface, pearson_similarity, plot_multivariate_time_series, \
    plot_heatmap, With_labels
from sklearn.preprocessing import MinMaxScaler, StandardScaler
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_boolean_matrix(M, N):
    K = M * N
    row_indices = np.arange(K) // N
    col_indices = np.arange(K) % N
    boolean_matrix = np.logical_or(
        np.eye(K),
        np.logical_or(
            np.equal.outer(row_indices, row_indices),
            np.equal.outer(col_indices, col_indices)
        )
    ).astype(int)
    return ~boolean_matrix

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        series = self.dropout(torch.softmax(attn, dim=-1))
        # plot_heatmap(series[0,0,:,:].cpu().detach().numpy())
        # plot_heatmap(prior[0,0,:,:].cpu().detach().numpy())
        # plot_heatmap(scores[0, 0, :, :].cpu().detach().numpy())

        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AnomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AnomAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class CrossAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(CrossAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.split=1
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, keysT, values,valuesT, attn_mask):
        B, L, H, E = queries.shape    # (B, L, H, E==64=dk)
        _, S, _, D = values.shape     # S==L D==E==64
        _, C, _, _ = valuesT.shape  # S==L D==E==64
        # 生成0到7的列表并将列表随机打乱
        nums = list(range(8))
        random.shuffle(nums)
        # 将头数分成两份
        series_queries = torch.index_select(queries, dim=2, index=torch.tensor(nums[self.split:]).to(device))
        cross_queries = torch.index_select(queries, dim=2, index=torch.tensor(nums[:self.split]).to(device))
        series_keys=torch.index_select(keys, dim=2, index=torch.tensor(nums[self.split:]).to(device))
        cross_keysT = torch.index_select(keysT, dim=2, index=torch.tensor(nums[:self.split]).to(device))
        values=torch.index_select(values, dim=2, index=torch.tensor(nums[self.split:]).to(device))
        valuesT= torch.index_select(valuesT, dim=2, index=torch.tensor(nums[:self.split]).to(device))
        # sigma=torch.index_select(sigma, dim=2, index=torch.tensor(nums[self.split:]).to(device))

        # print('series_queries:',series_queries.shape)
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blne,bsne->bnls", series_queries, series_keys)
        cross_scores = torch.einsum("blme,bcme->bmlc", cross_queries, cross_keysT)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            cross_scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        cross_attn = scale * cross_scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        # sigma = torch.sigmoid(sigma * 5) + 1e-5
        # sigma = torch.pow(3, sigma) - 1
        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        crosses = self.dropout(torch.softmax(cross_attn, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", series, values)
        V_cross=torch.einsum("bhlc,bchd->blhd", crosses, valuesT)

        V_total=torch.cat([V, V_cross], dim=2)

        if self.output_attention:
            return (V_total.contiguous(), series)
        else:
            return (V_total.contiguous(), None)



class CrissCrossAtten(nn.Module):
    def __init__(self,seg_num,Dim,mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(CrissCrossAtten, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.split=1
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.M = Dim
        self.N = seg_num


    def forward(self, queries, keys, values, map,attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # plot_heatmap_surface(scores[12,2,:,:].cpu().detach().numpy())
        # print('Attention:',scores.shape)
        # print('Spatio-Temporal Dependency:',map.shape)
        scores = torch.einsum('...ij,...jk->...ik', scores, map)  #linear kernel function
        # scores = torch.einsum('...ij, ij->...ij', scores, torch.tensor(create_boolean_matrix(self.M,self.N)).to(device))

        # plot_heatmap_surface(np.array(map).reshape(x.shape[1],x.shape[2]))

        # plt.imshow(map[:,:].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Heatmap Example")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.show()

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            # cross_scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        # cross_attn = scale * cross_scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        # sigma = torch.sigmoid(sigma * 5) + 1e-5
        # sigma = torch.pow(3, sigma) - 1
        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        # crosses = self.dropout(torch.softmax(cross_attn, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", series, values)
        # V_cross=torch.einsum("bhlc,bchd->blhd", crosses, valuesT)

        # V_total=torch.cat([V, V_cross], dim=2)

        if self.output_attention:
            return (V.contiguous(), series)
        else:
            return (V.contiguous(), None)


# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#
#     def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
#         # Q [B, H, L, D]
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape
#
#         # calculate the sampled Q_K
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         # real U = U_part(factor*ln(L_k))*L_q
#         index_sample = torch.randint(L_K, (L_Q, sample_k))
#         K_sample = K_expand[:, :, torch.arange(
#             L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(
#             Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
#
#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]
#
#         # use the reduced Q to calculate Q_K
#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                    torch.arange(H)[None, :, None],
#                    M_top, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
#
#         return Q_K, M_top
#
#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             # V_sum = V.sum(dim=-2)
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H,
#                                                 L_Q, V_sum.shape[-1]).clone()
#         else:  # use mask
#             # requires that L_Q == L_V, i.e. for self-attention only
#             assert (L_Q == L_V)
#             contex = V.cumsum(dim=-2)
#         return contex
#
#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         B, H, L_V, D = V.shape
#
#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#
#         attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
#
#         context_in[torch.arange(B)[:, None, None],
#         torch.arange(H)[None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#         if self.output_attention:
#             attns = (torch.ones([B, H, L_V, L_V]) /
#                      L_V).type_as(attn).to(attn.device)
#             attns[torch.arange(B)[:, None, None], torch.arange(H)[
#                                                   None, :, None], index, :] = attn
#             return (context_in, attns)
#         else:
#             return (context_in, None)
#
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape
#
#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)
#
#         U_part = self.factor * \
#                  np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#         u = self.factor * \
#             np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
#
#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q
#
#         scores_top, index = self._prob_QK(
#             queries, keys, sample_k=U_part, n_top=u)
#
#         # add scale factor
#         scale = self.scale or 1. / sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         # get the context
#         context = self._get_initial_context(values, L_Q)
#         # update the context with selected top_k queries
#         context, attn = self._update_context(
#             context, values, scores_top, index, L_Q, attn_mask)
#
#         return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)   # q.shape->[7040=(128*55),7(9),8,16]
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class CrissAttenLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(CrissAttenLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values,map, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        _, C,_, _ = map.shape
        H = self.n_heads
        # map=map.view(B,C,C,1)
        queries = self.query_projection(queries).view(B, L, H, -1)   # q.shape->[7040=(128*55),7(9),8,16]
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        map=map.view(B, H, C, C)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            map,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayerFusion(nn.Module):
    def __init__(self, attention, d_model, n_heads,seq_len,enc_in, d_keys=None,
                 d_values=None):
        super(AttentionLayerFusion, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.a_value_projection = nn.Linear(seq_len+enc_in, seq_len)
        self.n_heads = n_heads

    def forward(self, queries, keys, values,valuest, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        values_total=torch.cat([values,valuest],dim=1).transpose(1,2)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)

        values = self.a_value_projection(values_total).transpose(1,2).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# class ReformerLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None, causal=False, bucket_size=4, n_hashes=4):
#         super().__init__()
#         self.bucket_size = bucket_size
#         self.attn = LSHSelfAttention(
#             dim=d_model,
#             heads=n_heads,
#             bucket_size=bucket_size,
#             n_hashes=n_hashes,
#             causal=causal
#         )
#
#     def fit_length(self, queries):
#         # inside reformer: assert N % (bucket_size * 2) == 0
#         B, N, C = queries.shape
#         if N % (self.bucket_size * 2) == 0:
#             return queries
#         else:
#             # fill the time series
#             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
#             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)
#
#     def forward(self, queries, keys, values, attn_mask, tau, delta):
#         # in Reformer: defalut queries=keys
#         B, N, C = queries.shape
#         queries = self.attn(self.fit_length(queries))[:, :N, :]
#         return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.anomaly_attention = AnomAttentionLayer(AnomalyAttention(configs.seq_len,False, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x,x_o, attn_mask=None, tau=None, delta=None):
        # print('x1:',x.shape)
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0] # x->[128,55,9,128]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')   # x->[128,55,7(4,2),128] -> [7040=(128*55),7(4,2),128]
        time_enc,attn= self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        x_or, series, prior, sigma = self.anomaly_attention(
            x_o, x_o, x_o, attn_mask=None
        )
        # dim_in = time_in + self.dropout(time_enc)
        # dim_in = self.norm1(dim_in)
        # dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        # dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        # final_out = rearrange(series, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        # final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        # final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out, series, prior, sigma


class HeadSeparateLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(HeadSeparateLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        # self.sigma_projection = nn.Linear(d_model,
        #                                   n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, valuesT, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape   # (B, L, d_model)
        _, S, _ = keys.shape        # (B, L, d_model) S==L==100
        _, C, _ = valuesT.shape              # (B, L, d_model) C==25
        H = self.n_heads
        x = queries
        # queries = self.query_projection(queries).view(B, L, H, -1)
        queries = self.query_projection(queries).view(B, L, H, -1)      # (B, L, Head, 64)
        keys = self.key_projection(keys).view(B, S, H, -1)
        keysT = self.key_projection(valuesT).view(B, C, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        valuest = self.value_projection(valuesT).view(B, C, H, -1)
        # sigma = self.sigma_projection(x).view(B, L, H)

        out, series = self.inner_attention(
            queries,
            keys,
            keysT,
            values,
            valuest,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series


class SingleStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(SingleStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.dim=configs.enc_in
        self.seg_num=seg_num
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        # self.time_attention = HeadSeparateLayer(CrossAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                                    output_attention=configs.output_attention), d_model, n_heads)
        self.anomaly_attention = AnomAttentionLayer(AnomalyAttention(configs.win_size,False, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.Merge = nn.Linear(configs.enc_in-1, 1)

    def forward(self, x, x_o,attn_mask=None, tau=None, delta=None):
        # print('x1:',x.shape)
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0] # x->[128,55,9,128]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> b (ts_d seg_num) d_model')
        # tensor_list = []
        # time_in=x.permute(1,0,2,3)[index]
        # indices = torch.tensor([i for i in range(x.size(1)) if i != index]).to(device)
        # new_tensor = torch.index_select(x, dim=1, index=indices).to(device)
        # time_others=self.Merge(new_tensor.permute(0,3,2,1))[...,-1].permute(0,2,1).to(device)
        # time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')   # x->[128,55,7(4,2),128] -> [7040=(128*55),7(4,2),128]

        # print('time_in:',time_in.shape)
        time_enc, _ = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        x_or, series, prior, sigma = self.anomaly_attention(
            x_o, x_o, x_o,attn_mask=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # tensor_list.append(dim_in)
        # dim_send = torch.stack(tensor_list, dim=0).permute(1,0,2,3)
        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, 'b (ts_d seg_num) d_model -> b ts_d seg_num d_model', ts_d=self.dim)
        # batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        # dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        # dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        # dim_enc = dim_send + self.dropout(dim_receive)
        # dim_enc = self.norm3(dim_enc)
        # dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        # dim_enc = self.norm4(dim_enc)

        # final_out = rearrange(dim_send, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return dim_send, series, prior, sigma



class CrissCrossAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(CrissCrossAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.dim=configs.enc_in
        self.seg_num=seg_num
        self.n_heads=n_heads
        # self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                                    output_attention=configs.output_attention), d_model, n_heads)
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.criss_attention =CrissAttenLayer(CrissCrossAtten(self.seg_num,self.dim,False, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        # self.time_attention = HeadSeparateLayer(CrossAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                                    output_attention=configs.output_attention), d_model, n_heads)
        self.anomaly_attention = AnomAttentionLayer(AnomalyAttention(configs.win_size,False, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.lamda_projection = nn.Linear(d_model,
                                          n_heads)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.Merge = nn.Linear(configs.enc_in-1, 1)

    def forward(self, x, x_o,attn_mask=None, tau=None, delta=None):
        # print('x1:',x.shape)
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0] # x->[128,55,9,128]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> b (ts_d seg_num) d_model')
        B,L,_=time_in.shape
        H = self.n_heads
        # tensor_list = []
        # time_in=x.permute(1,0,2,3)[index]
        # indices = torch.tensor([i for i in range(x.size(1)) if i != index]).to(device)
        # new_tensor = torch.index_select(x, dim=1, index=indices).to(device)
        # time_others=self.Merge(new_tensor.permute(0,3,2,1))[...,-1].permute(0,2,1).to(device)
        # time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')   # x->[128,55,7(4,2),128] -> [7040=(128*55),7(4,2),128]
        # ser=time_in.detach().cpu().numpy()
        # plot_multivariate_time_series(np.array(ser[20,:,:]))
        # map=torch.tensor(pearson_similarity(time_in[20,:, :].cpu().detach().numpy()))
        # print('map00:',map)
        # plot_heatmap(np.array(map).reshape(x.shape[1],x.shape[2]))

        lamda = self.lamda_projection(time_in).view(B,L,H) 

        # print('lamda heatmap!')

        # da1=time_in.cpu().detach().numpy()[20,:,0].reshape(x.shape[1],x.shape[2])
        # da2=lamda.cpu().detach().numpy()[20,:,0].reshape(x.shape[1],x.shape[2])

        # multi_heatplot(da1,da2)

        lamda = lamda.unsqueeze(1).repeat(1, L, 1, 1)  # lamda <- [B L L H]
        
        # plot_heatmap(lamda.cpu().detach().numpy()[20,:,:,0])

        mean_tensor = np.mean(time_in.cpu().detach().numpy(), axis=1)
        centered_tensor = time_in.cpu().detach().numpy() - mean_tensor[:, np.newaxis, :]
        std_tensor = np.std(time_in.cpu().detach().numpy(), axis=1)
        normalized_tensor = centered_tensor / std_tensor[:, np.newaxis, :]
        correlation_map = np.einsum('ijk,ik->ij', normalized_tensor, mean_tensor)# correlation_map<-[B,dim*segNum]
        map = torch.tensor(np.einsum('ij,ik->ijk', correlation_map, correlation_map)).to(device)
        map = map.unsqueeze(-1) * lamda # map <-[B,dim*segNum,dim*segNum,H] 
        
        # map<-[B,dim*segNum,dim*segNum]
        # linear kernel function
        # scaler = StandardScaler()
        # normalized_data = scaler.fit_transform(map.reshape(-1,1))
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # map = scaler.fit_transform(normalized_data)
        # plot_heatmap_surface(np.array(map).reshape(x.shape[1],x.shape[2]))
        # map=torch.softmax(torch.tensor(map),dim=0).to(device)
        # plot_heatmap_surface(map.cpu().detach().numpy().reshape(x.shape[1],x.shape[2]))
        # map=torch.matmul(map.reshape(-1,1).float(), map.reshape(1,-1).float()).to(device)
        # plot_heatmap(map[0:81,0:81].cpu().detach().numpy())



        time_enc, _ = self.criss_attention(
            time_in, time_in, time_in,map, attn_mask=None, tau=None, delta=None
        )


        x_or, series, prior, sigma = self.anomaly_attention(
            x_o, x_o, x_o,attn_mask=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # tensor_list.append(dim_in)
        # dim_send = torch.stack(tensor_list, dim=0).permute(1,0,2,3)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, 'b (ts_d seg_num) d_model -> b ts_d seg_num d_model', ts_d=self.dim)
        # batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        # dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        # dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        # dim_enc = dim_send + self.dropout(dim_receive)
        # dim_enc = self.norm3(dim_enc)
        # dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        # dim_enc = self.norm4(dim_enc)
        # final_out = rearrange(dim_send, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return dim_send, series, prior, sigma
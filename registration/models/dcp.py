import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.pcn import PCN_encoder,PCN_encoder22
from train_utils import rotation_error, translation_error, rmse_loss, rt_to_transformation, rotation_geodesic_error


# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
def knn_point(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_global, tgt_global):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, src_global), src_mask,
                           tgt, tgt_mask, tgt_global)

    def encode(self, src, src_mask, src_global):
        return self.encoder(self.src_embed(src), src_mask, src_global)

    def decode(self, memory, src_mask, tgt, tgt_mask, tgt_global):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, tgt_global))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, x_global):
        for layer in self.layers:
            x = layer(x, mask, x_global)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, tgt_global):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, tgt_global)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, src_global):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, src_global))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, tgt_global):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, src_global = None, tgt_global = tgt_global))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear = nn.Linear(d_model + 1024, d_model)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None, src_global = None, tgt_global = None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        
        x = self.linears[-1](x)
        if (src_global is not None):
            x = torch.cat([x, src_global.view(x.shape[0], 1, -1).repeat(1, x.shape[1], 1)], 2)
            x = self.linear(x)
        if (tgt_global is not None):
            x = torch.cat([x, tgt_global.view(x.shape[0], 1, -1).repeat(1, x.shape[1], 1)], 2)
            x = self.linear(x)
        return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = 256
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = 1024
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_global = input[2]
        tgt_global = input[3]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None, src_global, tgt_global).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None, tgt_global, src_global).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = 512
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.afff = nn.ReLU()

    def forward(self, feature_input):
        a = self.bn2(self.conv_res(feature_input))
        
        b = self.afff(feature_input)
        b = self.bn1(self.conv1(b))
        b = self.afff(b)
        b = self.bn3(self.conv2(b))
        res = a+b
        return res
class Model(nn.Module):
    def __init__(self, args, type_ = None):
        super(Model, self).__init__()
        self.emb_dims = 512
        self.cycle = False
        self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        self.pointer = Transformer(args=args)
        self.head = SVDHead(args=args)
        self.encoder = PCN_encoder()
        self.encoder22 = PCN_encoder22()
        size_z = 128
        self.size_z_global = size_z
        size_z = 32
        self.size_z_local = size_z
        self.pcn_output_global = 1024
        self.pcn_output_local = 256
        self.posterior_infer2 = Linear_ResBlock(input_size=self.pcn_output_local, output_size= self.size_z_local * 2)
        self.posterior_infer4 = Linear_ResBlock(input_size=self.pcn_output_global, output_size= self.size_z_global * 2)
        
        self.prior_infer2 = Linear_ResBlock(input_size=self.pcn_output_local, output_size=self.size_z_local * 2)
        self.prior_infer4 = Linear_ResBlock(input_size=self.pcn_output_global, output_size=self.size_z_global * 2)
        self.generator = Linear_ResBlock(input_size=self.size_z_local, output_size = self.pcn_output_local)
        self.generator2 = Linear_ResBlock(input_size=self.size_z_global, output_size = self.pcn_output_global)
    def forward(self, src, tgt, complete_src=None, complete_tgt=None, T_gt=None, pose_src = False, pose_tgt = False, prefix="train", epoch = 0, gamma = None):
        src_point = src
        complete_src_point = complete_src
        tgt_point = tgt

        src = src.transpose(1,2).contiguous()
        tgt = tgt.transpose(1,2).contiguous()
        complete_src = complete_src.transpose(1,2).contiguous()
        complete_tgt = complete_tgt.transpose(1,2).contiguous()
        batch_size, _, _ = src.size()
        
        src_idx = knn(src.contiguous(),8)
        src_local_ = index_points(src.transpose(1, 2).contiguous(), src_idx)
        src_local_feat = src_local_.reshape((src_local_.shape[0], src_local_.shape[1], -1)).transpose(1, 2).contiguous()
        
        tgt_idx = knn(tgt.contiguous(),8)
        tgt_local_ = index_points(tgt.transpose(1, 2).contiguous(), tgt_idx)
        tgt_local_feat = tgt_local_.reshape((tgt_local_.shape[0], tgt_local_.shape[1], -1)).transpose(1, 2).contiguous()
        
        src_global = src
        tgt_global = tgt
        
        if(prefix == 'train'):
            complete_src_global = complete_src
            complete_tgt_global = complete_tgt
        
        src = src.transpose(1, 2).contiguous()
        tgt = tgt.transpose(1, 2).contiguous()
        if (prefix == 'train'): 
            complete_src = complete_src.transpose(1, 2).contiguous()
            complete_tgt = complete_tgt.transpose(1, 2).contiguous() 
            _, gt_idx = knn_point(1, complete_tgt, tgt)
            tgt_local__ = index_points(complete_tgt, gt_idx).squeeze()
            _, idx = knn_point(8, complete_tgt, tgt_local__)
            tgt_local_ = index_points(complete_tgt, idx)
            tgt_local_idx_feat = tgt_local_.reshape((tgt_local_.shape[0], tgt_local_.shape[1], -1)).transpose(1, 2).contiguous()
            

            _, gt_idx = knn_point(1, complete_src, src)
            tgt_local__ = index_points(complete_src, gt_idx).squeeze()
            _, idx = knn_point(8, complete_src, tgt_local__)
            tgt_local_ = index_points(complete_src, idx)
            src_local_idx_feat = tgt_local_.reshape((tgt_local_.shape[0], tgt_local_.shape[1], -1)).transpose(1, 2).contiguous()
            
            
            local_ori = torch.cat([src_local_feat, tgt_local_feat, src_local_idx_feat, tgt_local_idx_feat], dim=0)
            global_ori = torch.cat([src_global, tgt_global, complete_src_global, complete_tgt_global], dim=0)
        else:
            local_ori = torch.cat([src_local_feat, tgt_local_feat], dim=0)
            global_ori = torch.cat([src_global, tgt_global], dim=0)
        

        global_feat = self.encoder22(global_ori)
        local_feat = self.encoder(local_ori)

#         if (prefix == 'train'):
#             ori_local, idx_local = local_feat.chunk(2)
#             ori_global, idx_global = global_feat.chunk(2)
#         else:
#             ori_local = local_feat
#             ori_global = global_feat
#         try:    
#             o_x = self.posterior_infer2((ori_local))
#             q_mu, q_std = torch.split(o_x, self.size_z_local, dim=1)
#             q_std = F.softplus(q_std)
#             ori_distribution_local = torch.distributions.Normal(q_mu, q_std)
#             standard_distribution_local = torch.distributions.normal.Normal(torch.zeros_like(q_mu), torch.ones_like(q_std))
#             z_local = ori_distribution_local.rsample()


#             q_x = self.posterior_infer4((ori_global))
#             o_mu, o_std = torch.split(q_x, self.size_z_global, dim=1)
#             o_std = F.softplus(o_std)
#             ori_distribution_global = torch.distributions.Normal(o_mu, o_std)
#             standard_distribution_global = torch.distributions.normal.Normal(torch.zeros_like(o_mu), torch.ones_like(o_std))
#             z_global = ori_distribution_global.rsample() 

#             if (prefix == 'train'):
#                 t_x = self.prior_infer2((idx_local))
#                 t_mu, t_std = torch.split(t_x, self.size_z_local, dim=1)
#                 t_std = F.softplus(t_std)
#                 idx_distribution_local = torch.distributions.Normal(t_mu, t_std)
#                 idx_distribution_local_fix = torch.distributions.Normal(t_mu.detach(), t_std.detach())
#                 z_local_idx = idx_distribution_local.rsample()

#                 t_x = self.prior_infer4(idx_global)
#                 t_mu, t_std = torch.split(t_x, self.size_z_global, dim=1)
#                 t_std = F.softplus(t_std)
#                 idx_distribution_global = torch.distributions.Normal(t_mu, t_std)
#                 idx_distribution_global_fix = torch.distributions.Normal(t_mu.detach(), t_std.detach())
#                 z_global_idx = idx_distribution_global.rsample()


#     #             local_feat = torch.cat([ori_local, ori_local], 0)
#     #             global_feat = torch.cat([ori_global, ori_global], 0)
#                 z_local = torch.cat([z_local, z_local_idx], 0)
#                 z_global = torch.cat([z_global, z_global_idx], 0)
                
#         except Exception as e:
#             print(e)
#             print(o_mu)
#             print(o_std)
#             print("dsadsadsad")
#             import time
#             time.sleep(10000000)
            
        
        src = src.transpose(1, 2).contiguous()
        tgt = tgt.transpose(1, 2).contiguous() 
#         z_local = self.generator(z_local)
#         z_global = self.generator2(z_global)
        
        if (prefix == 'train'):
            complete_src = complete_src.transpose(1, 2).contiguous()
            complete_tgt = complete_tgt.transpose(1, 2).contiguous()
#             feat_z = z_local.reshape((src.shape[0] * 4,src.shape[2],z_local.shape[1])).transpose(2,1)
            feat = local_feat.reshape((src.shape[0] * 4,src.shape[2],local_feat.shape[1])).transpose(2,1)
            
        else:
#             feat_z = z_local.reshape((src.shape[0] * 2,src.shape[2],z_local.shape[1])).transpose(2,1)
            feat = local_feat.reshape((src.shape[0] * 2,src.shape[2],local_feat.shape[1])).transpose(2,1)
            
        feat = feat
        
        feat_global = global_feat
        
        if (prefix == 'train'):
            src_embedding, tgt_embedding = torch.cat([feat.chunk(4)[0],feat.chunk(4)[2]],0), torch.cat([feat.chunk(4)[1],feat.chunk(4)[3]],0)
            
            src_embedding_p, tgt_embedding_p = torch.cat([feat.chunk(4)[0],feat.chunk(4)[2]],0), torch.cat([feat.chunk(4)[1],feat.chunk(4)[3]],0)
            src_global, tgt_global = torch.cat([feat_global.chunk(4)[0],feat_global.chunk(4)[2]],0), torch.cat([feat_global.chunk(4)[1],feat_global.chunk(4)[3]],0)
            src = torch.cat([src, complete_src], 0)
            tgt = torch.cat([tgt, complete_tgt], 0)
            T_gt = torch.cat([T_gt, T_gt], 0)
            
            src_point = torch.cat([src_point, complete_src_point], 0)
        else:
            src_embedding, tgt_embedding = feat.chunk(2)
            src_embedding_p, tgt_embedding_p = feat.chunk(2)
            src_global, tgt_global = feat_global.chunk(2)
        
        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding_p, tgt_embedding_p, src_global, tgt_global)
        
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p
        
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt)
        if self.cycle:
            rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)

        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        
        T_12 = rt_to_transformation(rotation_ab, translation_ab.unsqueeze(2))
        
        if T_gt == None:
            return T_12
        else:
            r_err = rotation_error(T_12[:, :3, :3], T_gt[:, :3, :3])
            t_err = translation_error(T_12[:, :3, 3], T_gt[:, :3, 3])
            rmse = rmse_loss(src_point, T_12, T_gt)
            eye = torch.eye(4).expand_as(T_gt).to(T_gt.device)
            mse = F.mse_loss(T_12 @ torch.inverse(T_gt), eye)
            loss = mse
            rt_mse = (rotation_geodesic_error(T_12[:, :3, :3], T_gt[:, :3, :3]) + translation_error(T_12[:, :3, 3], T_gt[:, :3, 3]))
#             loss = rt_mse.mean()
#             if (prefix == 'train'):
#                 dl_local = torch.distributions.kl_divergence(standard_distribution_local, idx_distribution_local) + torch.distributions.kl_divergence(idx_distribution_local_fix, ori_distribution_local)
#                 dl_global = torch.distributions.kl_divergence(standard_distribution_global, idx_distribution_global) + torch.distributions.kl_divergence(idx_distribution_global_fix, ori_distribution_global)
#                 dl_loss = (dl_local.mean() + dl_global.mean())
#                 loss = loss + dl_loss * 0.01
            
            if (prefix == 'train'):
                return loss, r_err, t_err, rmse, rt_mse
            if (prefix == 'val'):
                return r_err, t_err, rmse, rt_mse, loss
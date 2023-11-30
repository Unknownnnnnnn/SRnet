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
# from util import knn, batch_choice
import open3d as o3d
# from open3d.open3d.geometry import estimate_normals
from train_utils import rotation_error, translation_error, rmse_loss, rt_to_transformation, rotation_geodesic_error
# from model_utils import knn, batch_choice, FPFH, Conv1DBlock, Conv2DBlock, SVDHead
from models.pcn import PCN_encoder,PCN_encoder22
import random

def knn_point(pk, point_input, point_output):
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx
def batch_choice(data, k, p=None, replace=False):
    # data is [B, N]
    out = []
    for i in range(len(data)):
        out.append(np.random.choice(data[i], size=k, p=p[i], replace=replace))
    out = np.stack(out, 0)
    return out

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
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


    
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

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = 33 if args.use_fpfh else args.descriptor_size
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights, src_corr_centered.transpose(2, 1).contiguous())

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
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights * src).sum(dim=2, keepdim=True)) + (weights * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)


class Model(nn.Module):
    def __init__(self, args, type_ = None):
        super(Model, self).__init__()
        self.head = SVDHead(args=args)
        self.num_iter = args.num_iters
        self.encoder = PCN_encoder()
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
        self.encoder2 = PCN_encoder22()
        self.fuse = Conv1DBlock((256, 256, 64, 64), 1)
        self.significance_fc = Conv1DBlock((64, 64, 32, 1), 1)
        
        self.sim_mat_conv1 = nn.ModuleList([Conv2DBlock((64*2+4, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv2 = nn.ModuleList([Conv2DBlock((32, 32, 1), 1) for _ in range(self.num_iter)])
        self.weight_fc = nn.ModuleList([Conv1DBlock((32, 32, 1), 1) for _ in range(self.num_iter)])
    def forward(self, src, tgt, complete_src=None, complete_tgt=None, T_gt=None, pose_src = False, pose_tgt = False, prefix="train", epoch = 0, gamma = None):
        self.pts = src
        
        if T_gt is not None:
            R_gt = T_gt[:, :3, :3]
            t_gt = T_gt[:, :3, 3]

        src = src.transpose(1,2).contiguous()
        tgt = tgt.transpose(1,2).contiguous()
        complete_src = complete_src.transpose(1,2).contiguous()
        complete_tgt = complete_tgt.transpose(1,2).contiguous()
        ##### getting ground truth correspondences #####
        if prefix == "train":
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
            min_dist, min_idx = (dist ** 2).sum(1).min(-1) # [B, npoint], [B, npoint]
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy() # drop to cpu for numpy
            match_labels = (min_dist < 0.05).float()
            indicator = match_labels.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)
        ##### getting ground truth correspondences #####

        ##### get embedding and significance score #####
        
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
        
        
        global_feat = self.encoder2(global_ori)  #batch,2048,5*3
        local_feat = self.encoder(local_ori)
        
        if (prefix == 'train'):
            ori_local, idx_local = local_feat.chunk(2)
            ori_global, idx_global = global_feat.chunk(2)
        else:
            ori_local = local_feat
            ori_global = global_feat
        
        try:
            o_x = self.posterior_infer2((ori_local))
            q_mu, q_std = torch.split(o_x, self.size_z_local, dim=1)
            q_std = F.softplus(q_std)
            ori_distribution_local = torch.distributions.Normal(q_mu, q_std)
            standard_distribution_local = torch.distributions.normal.Normal(torch.zeros_like(q_mu), torch.ones_like(q_std))
            z_local = ori_distribution_local.rsample()


            q_x = self.posterior_infer4((ori_global))
            o_mu, o_std = torch.split(q_x, self.size_z_global, dim=1)
            o_std = F.softplus(o_std)
            ori_distribution_global = torch.distributions.Normal(o_mu, o_std)
            standard_distribution_global = torch.distributions.normal.Normal(torch.zeros_like(o_mu), torch.ones_like(o_std))
            z_global = ori_distribution_global.rsample() 

            if (prefix == 'train'):
                t_x = self.prior_infer2((idx_local))
                t_mu, t_std = torch.split(t_x, self.size_z_local, dim=1)
                t_std = F.softplus(t_std)
                idx_distribution_local = torch.distributions.Normal(t_mu, t_std)
                idx_distribution_local_fix = torch.distributions.Normal(t_mu.detach(), t_std.detach())
                z_local_idx = idx_distribution_local.rsample()

                t_x = self.prior_infer4(idx_global)
                t_mu, t_std = torch.split(t_x, self.size_z_global, dim=1)
                t_std = F.softplus(t_std)
                idx_distribution_global = torch.distributions.Normal(t_mu, t_std)
                idx_distribution_global_fix = torch.distributions.Normal(t_mu.detach(), t_std.detach())
                z_global_idx = idx_distribution_global.rsample()


                local_feat = torch.cat([ori_local, ori_local], 0)
                global_feat = torch.cat([ori_global, ori_global], 0)
                z_local = torch.cat([z_local, z_local_idx], 0)
                z_global = torch.cat([z_global, z_global_idx], 0)
                
        except Exception as e:
            print(e)
            print(o_mu)
            print(o_std)
            print("dsadsadsad")
            import time
            time.sleep(10000000)
        src = src.transpose(1, 2).contiguous()
        tgt = tgt.transpose(1, 2).contiguous() 
        z_local = self.generator(z_local)
        z_global = self.generator2(z_global)
        
        if (prefix == 'train'):
            feat_z = z_local.reshape((src.shape[0] * 4,src.shape[2],z_local.shape[1])).transpose(2,1)
            feat = local_feat.reshape((src.shape[0] * 4,src.shape[2],local_feat.shape[1])).transpose(2,1)
        else:
            feat_z = z_local.reshape((src.shape[0] * 2,src.shape[2],z_local.shape[1])).transpose(2,1)
            feat = local_feat.reshape((src.shape[0] * 2,src.shape[2],local_feat.shape[1])).transpose(2,1)
        
        feat = feat + feat_z.contiguous()
        feat_global = global_feat + z_global.contiguous()
        if (prefix == 'train'):
            feat_all = torch.cat([feat, feat_global.view(src.shape[0] * 4, -1, 1).repeat(1, 1, src.shape[2])], 1)
        else:
            feat_all = torch.cat([feat, feat_global.view(src.shape[0] * 2, -1, 1).repeat(1, 1, src.shape[2])], 1)
            
        feat_all = self.fuse(feat)
        src_embedding, tgt_embedding = feat_all.chunk(2)
        if (prefix == 'train'):
            src_embedding_ = torch.cat([src_embedding.chunk(2)[0],tgt_embedding.chunk(2)[0]],0)
            tgt_embedding_ = torch.cat([src_embedding.chunk(2)[1],tgt_embedding.chunk(2)[1]],0)
            src_embedding, tgt_embedding = src_embedding_, tgt_embedding_
            
            
            
        src_sig_score = self.significance_fc(src_embedding).squeeze(1)
        tgt_sig_score = self.significance_fc(tgt_embedding).squeeze(1)
        ##### get embedding and significance score #####

        ##### hard point elimination #####
        num_point_preserved = src.size(-1) // 8
        if prefix == "train":
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved//2, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved-num_point_preserved//2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        if prefix == "train":
            src_idx = np.concatenate([src_idx, src_idx], 0)
            tgt_idx = np.concatenate([tgt_idx, tgt_idx], 0)
            batch_idx = np.concatenate([batch_idx, batch_idx], 0)
            src = torch.cat([src, src], 0)
            tgt = torch.cat([tgt, tgt], 0)
            match_labels = match_labels[batch_idx, src_idx]
            R_gt = torch.cat([R_gt, R_gt], 0)
            t_gt = torch.cat([t_gt, t_gt], 0)
            self.pts = torch.cat([self.pts, self.pts], 0)
            T_gt = torch.cat([T_gt, T_gt], 0)
            
        src = src[batch_idx, :, src_idx].transpose(1, 2)
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)
        src_sig_score = src_sig_score[batch_idx, src_idx]
        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]
        ##### hard point elimination #####
        
        ##### initialize #####
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        loss = 0.
        ##### initialize #####

        for i in range(self.num_iter):

            ##### stack features #####
            batch_size, num_dims, num_points = src_embedding.size()
            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix = torch.cat([_src_emb, _tgt_emb], 1)
            ##### stack features #####

            ##### compute distances #####
            diff = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            dist = (diff ** 2).sum(1, keepdim=True)
            dist = torch.sqrt(dist)
            diff = diff / (dist + 1e-8)
            ##### compute distances #####

            ##### similarity matrix convolution to get features #####
            similarity_matrix = torch.cat([similarity_matrix, dist, diff], 1)
            similarity_matrix = self.sim_mat_conv1[i](similarity_matrix)
            ##### similarity matrix convolution to get features #####

            ##### soft point elimination #####
            weights = similarity_matrix.max(-1)[0]
            weights = self.weight_fc[i](weights).squeeze(1)
            ##### soft point elimination #####

            ##### similarity matrix convolution to get similarities #####
            similarity_matrix = self.sim_mat_conv2[i](similarity_matrix)
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)
            ##### similarity matrix convolution to get similarities #####
            
            ##### negative entropy loss #####
            if prefix == "train" and i == 0:
                src_neg_ent = torch.softmax(similarity_matrix, dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix, dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                loss = loss + F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach())
            ##### negative entropy loss #####

            ##### matching loss #####
            if prefix == "train":
                temp = torch.softmax(similarity_matrix, dim=-1)
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss + match_loss
            ##### matching loss #####

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)
            ##### finding correspondences #####

            ##### soft point elimination loss #####
            if prefix == "train":
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)
                loss = loss + weight_loss
            ##### soft point elimination loss #####
            
            if (prefix == 'train' and i == 0 ):
                dl_local = torch.distributions.kl_divergence(standard_distribution_local, idx_distribution_local) + torch.distributions.kl_divergence(idx_distribution_local_fix, ori_distribution_local)
                dl_global = torch.distributions.kl_divergence(standard_distribution_global, idx_distribution_global) + torch.distributions.kl_divergence(idx_distribution_global_fix, ori_distribution_global)
                dl_loss = (dl_local.mean() + dl_global.mean())
                loss = loss + dl_loss * 0.1
            ##### hybrid point elimination #####
            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
            ##### normalize weights #####

            ##### get R and t #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights.unsqueeze(1))
            rotation_ab = rotation_ab.detach() # prevent backprop through svd
            translation_ab = translation_ab.detach() # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab
            ##### get R and t #####

        self.T = rt_to_transformation(R, t.unsqueeze(-1))
        if T_gt is None:
            return self.T
        else:
            mse = (rotation_geodesic_error(R, R_gt) + translation_error(t, t_gt)) #.mean()
            r_err = rotation_error(R, R_gt)
            t_err = translation_error(t, t_gt)
            
            rmse = rmse_loss(self.pts, self.T, T_gt)

            # return R, t, loss
            if (prefix == 'train'):
                return loss, r_err, t_err, rmse, mse
            else:
                return r_err, t_err, rmse, mse, loss
    def get_transform(self):
        return self.T #, self.scores12
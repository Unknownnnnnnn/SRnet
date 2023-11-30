from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from vis_utils import plot_single_pcd
from torch.nn import init
from torch.autograd import Variable
import numpy as np
# from utils.model_utils import gen_grid_up, calc_emd, calc_cd
from model_utils import *
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points


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
    
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c    
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
#         torch.Size([16, 256, 256])
#         torch.Size([16, 256, 64])
#         torch.Size([16, 64, 256])
#         torch.Size([16, 256, 256])
        
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = (x_v @ attention) # b, c, n 
        x_r = (self.act(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
# class PCN_encoder(nn.Module):
#     def __init__(self, output_size=1024):
#         super(PCN_encoder, self).__init__()
#         self.conv1 = nn.Conv1d(24, 32, 1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(32, 64, 1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.conv4 = nn.Conv1d(256, 256, 1)
#         self.att1 = SA_Layer(64)
#         self.att2 = SA_Layer(256)
# #         self.conv5 = nn.Conv1d(256, 256, 1)
#     def forward(self, x):
#         batch_size, _, num_points = x.size()
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
# #         x = self.att1(x)
#         global_feature, _ = torch.max(x, 2)
#         x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
        
#         return x
class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=True)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Conv1DBlock(nn.Module):
    def __init__(self, channels,ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x

    
class Linear1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Linear1DBNReLU, self).__init__()
        self.conv = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Linear1DBlock(nn.Module):
    def __init__(self, channels,):
        super(Linear1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Linear1DBNReLU(channels[i], channels[i+1]))
        self.conv.append(nn.Linear(channels[-2], channels[-1]))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x
    
class PCN_encoder(nn.Module):
    def __init__(self, ):
        super(PCN_encoder, self).__init__()
        additional_channel = 21
        
        self.sa1 = PointNetSetAbstraction(1024, 0.035, 16, 27, [64, 128, 128], False)
        self.sa2 = PointNetSetAbstraction(512, 0.0176, 32, 128 + 3, [128, 256], False)
        
        self.fp2 = PointNetFeaturePropagation(384, [256, 256])
        self.fp1 = PointNetFeaturePropagation(283, [256, 256])
        
    def forward(self, xyz):
#         xyz = xyz[:, :3, :]
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
#         label = F.one_hot(label, num_classes=16) 
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         cls_label_one_hot = label.view(B,16,1).repeat(1,1,N)
        feat = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)

        
        feat = feat.transpose(2,1)
        feat = feat.reshape((feat.shape[0]*feat.shape[1],-1))
        return feat
    
class PCN_encoder22(nn.Module):
    def __init__(self, ):
        super(PCN_encoder22, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.0176, nsample=32, in_channel=3 , mlp=[64, 128, 256], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=2.3466, nsample=64, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3, mlp=[1024, 1024], group_all=True)

        self.conv1 = nn.Linear(1024, 2048)
        self.conv2 = nn.Linear(2048, 2048)
        self.conv3 = nn.Linear(2048, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
    def forward(self, xyz):
        
        B,C,N = xyz.shape
#         label = F.one_hot(label, num_classes=16) 
#         cls_label_one_hot = label.view(B,16,1).repeat(1,1,N)
        norm = None
        
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        feat = l3_points.view(B, -1)
#         feat = F.relu(self.conv1(feat))
#         feat = F.relu(self.conv2(feat))
#         feat = self.conv3(feat)
        
        return feat
        
        
# class PCN_encoder22(nn.Module):
#     def __init__(self, output_size=1024):
#         super(PCN_encoder22, self).__init__()
#         self.conv1 = nn.Conv1d(3, 128, 1)
#         self.conv2 = nn.Conv1d(128, 256, 1)
#         self.conv3 = nn.Conv1d(512, 1024, 1)
#         self.conv4 = nn.Conv1d(1024, 2048, 1)

#     def forward(self, x):
#         batch_size, _, num_points = x.size()
#         x = F.relu(self.conv1(x))
#         x = self.conv2(x)
#         global_feature, _ = torch.max(x, 2)
#         x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
#         x = F.relu(self.conv3(x))
#         x = self.conv4(x)
#         global_feature, _ = torch.max(x, 2)
#         return global_feature.view(batch_size, -1),x
    
    
# class PCN_encoder22(nn.Module):
#     def __init__(self, output_size=1024):
#         super(PCN_encoder22, self).__init__()                                                                  
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.conv1_ = nn.Conv1d(3, 64, 1)
#         self.conv1__ = nn.Conv1d(3, 64, 1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn1_ = nn.BatchNorm1d(32)
#         self.bn1__ = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(64, 256, 1)
#         self.conv2_ = nn.Conv1d(64, 256, 1)
#         self.conv2__ = nn.Conv1d(64, 256, 1)
#         self.conv4 = nn.Conv1d(1024, 2048, 1)
#         self.conv4_ = nn.Conv1d(128, 256, 1)
#         self.conv4__ = nn.Conv1d(128, 256, 1)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.conv6 = nn.Conv1d(2048, 2048, 1)
#         self.conv6_ = nn.Conv1d(256, 256, 1)
#         self.conv6__ = nn.Conv1d(256, 256, 1)
#         self.merge = nn.Linear(768, 256)
#         self.relu = nn.ReLU(inplace = False)
#     def forward(self, x):
#         batch_size, _, num_points = x.size()
        
#         x1 = self.relu(self.conv1(x))
#         x1 = self.conv2(x1)
#         global_feature, _ = torch.max(x1, 2)
#         x1 = torch.cat((x1, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        
        
#         p_idx = furthest_point_sample(x.transpose(2,1).contiguous(), num_points // 2 )
#         x2 = gather_points(x.contiguous(), p_idx.int().contiguous()) #0.571168
#         x2 = self.relu(self.conv1_(x2))
#         x2 = self.conv2_(x2)
#         global_feature_2, _ = torch.max(x2, 2)
#         x2 = torch.cat((x1, global_feature_2.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        
#         p_idx = furthest_point_sample(x.transpose(2,1).contiguous(), num_points // 4 )
#         x3 = gather_points(x.contiguous(), p_idx.int().contiguous()) #0.571168
#         x3 = self.relu(self.conv1__(x3))
#         x3 = self.conv2__(x3)
#         global_feature_3, _ = torch.max(x3, 2)
#         x3 = torch.cat((x2, global_feature_3.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)

#         x = self.relu(self.conv4(x3))
#         x = self.conv6(x)
#         global_feature, _ = torch.max(x, 2)
        
#         return global_feature.view(batch_size, -1), x


class PCN_encoder_emb(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder_emb, self).__init__()                                                                  
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv1_ = nn.Conv1d(3, 16, 1)
        self.conv1__ = nn.Conv1d(3, 16, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn1_ = nn.BatchNorm1d(32)
        self.bn1__ = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(16, 64, 1)
        self.conv2_ = nn.Conv1d(16, 64, 1)
        self.conv2__ = nn.Conv1d(16, 64, 1)
        self.conv4 = nn.Conv1d(256, 256, 1)
        self.conv4_ = nn.Conv1d(128, 256, 1)
        self.conv4__ = nn.Conv1d(128, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, 1)
        self.conv6_ = nn.Conv1d(256, 256, 1)
        self.conv6__ = nn.Conv1d(256, 256, 1)
        self.merge = nn.Linear(768, 256)
        self.relu = nn.ReLU(inplace = False)
    def forward(self, x):
        batch_size, _, num_points = x.size()
        
        x1 = self.relu(self.conv1(x))
        x1 = self.conv2(x1)
        global_feature, _ = torch.max(x1, 2)
        x1 = torch.cat((x1, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        
        
        p_idx = furthest_point_sample(x.transpose(2,1).contiguous(), num_points // 2 )
        x2 = gather_points(x.contiguous(), p_idx.int().contiguous()) #0.571168
        x2 = self.relu(self.conv1_(x2))
        x2 = self.conv2_(x2)
        global_feature_2, _ = torch.max(x2, 2)
        x2 = torch.cat((x1, global_feature_2.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        
        p_idx = furthest_point_sample(x.transpose(2,1).contiguous(), num_points // 4 )
        x3 = gather_points(x.contiguous(), p_idx.int().contiguous()) #0.571168
        x3 = self.relu(self.conv1__(x3))
        x3 = self.conv2__(x3)
        global_feature_3, _ = torch.max(x3, 2)
        x3 = torch.cat((x2, global_feature_3.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)

        x = self.relu(self.conv4(x3))
        x = self.conv6(x)
        
        
        return x

class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Model(nn.Module):
    def __init__(self, args, num_coarse=1024):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 3 + 1024

        self.encoder = PCN_encoder()
        self.decoder = PCN_decoder(num_coarse, self.num_points, self.scale, self.cat_feature_num)

    def forward(self, x, gt=None, prefix="train", mean_feature=None, alpha=None):
        feat = self.encoder(x)
        out1, out2 = self.decoder(feat)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()

        if prefix=="train":
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + loss2.mean() * alpha
            return out2, loss2, total_train_loss
        elif prefix=="val":
            if self.eval_emd:
                emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
#             result = {}
#             result['out1'] = out1
#             result['out2'] = out2
#             result['emd'] = emd
#             result['cd_p'] = cd_p
#             result['cd_t'] = cd_t
#             result['f1'] = f1

            return (cd_p,cd_t,f1)
        else:
            return {'result': out2}
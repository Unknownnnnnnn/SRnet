import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from utils.model_utils import *
from model_utils import *
from models.pcn import PCN_encoder,PCN_encoder_2
from vis_utils import plot_single_pcd
import open3d as o3d
import random
from torch.autograd import Variable
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation, grouping_operation
from helper2 import PointNetSetAbstraction,PointNetFeaturePropagation
import voxelocc
def get_projection_loss(project):
    sigma = project.sigma()
    return sigma

def _axis_to_dim(axis):
    """Translate Tensorflow 'axis' to corresponding PyTorch 'dim'"""
    return {0: 0, 1: 2, 2: 3, 3: 1}.get(axis)
class SoftProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-4,
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(min_sigma, dtype=torch.float32)

    def forward(self, point_cloud, query_cloud, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        return self.project(point_cloud, query_cloud)

    def _group_points(self, point_cloud, query_cloud, point_features=None):
        group_size = self._group_size
        # find nearest group_size neighbours in point_cloud
        dist, idx = knn_point(group_size, point_cloud, query_cloud)
        
        point_cloud = point_cloud.transpose(1,2).contiguous()
        query_cloud = query_cloud.transpose(1,2).contiguous()
        # self._dist = dist.unsqueeze(1).permute(0, 1, 3, 2) ** 2

        idx = idx.type(
            torch.int32
        )  # index should be Batch x QueryPoints x K
        
        grouped_points = grouping_operation(point_cloud, idx.contiguous())  # B x 3 x QueryPoints x K
        grouped_features = (
            None if point_features is None else group_point(point_features, idx)
        )  # B x F x QueryPoints x K
        return grouped_points, grouped_features

    def _get_distances(self, grouped_points, query_cloud):
        query_cloud = query_cloud.transpose(1,2).contiguous()
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True) / (self.sigma()+1e-8)
        return dist

    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature ** 2, self._min_sigma.to(device))

    def project(self, point_cloud, query_cloud, hard=False):
        grouped_points, _ = self._group_points(point_cloud, query_cloud)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        if hard:
            raise NotImplementedError

        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return projected_points
class AttProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-4,
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(min_sigma, dtype=torch.float32)

    def forward(self, point_cloud, query_cloud, weight, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        return self.project(point_cloud, query_cloud, weight)

    def _get_distances(self, grouped_points, query_cloud):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True) / self.sigma()
        return dist

    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature ** 2, self._min_sigma.to(device))
        
    def project(self, point_cloud, query_cloud, weight, hard=False):
#         torch.Size([12, 2048, 25])
#         torch.Size([12, 3, 2048])
#         torch.Size([12, 3, 25, 2048])
        weight = weight.unsqueeze(1)
        grouped_points = query_cloud.transpose(3,2)
        dist = self._get_distances(grouped_points, point_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        weight = torch.softmax(weight, dim=_axis_to_dim(2))
        weight = weight.repeat(1, 3, 1, 1)
        
        projected_points = torch.sum(
            grouped_points * weights * weight, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        return projected_points
    
# class pointnet_encoder(nn.Module):
#     def __init__(self):
#         super(pointnet_encoder, self).__init__()
#         self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
#         self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
#         self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
#         self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
#         self.fp4 = PointNetFeaturePropagation(768, [256, 256])
#         self.fp3 = PointNetFeaturePropagation(384, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(320, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
#         self.conv1 = nn.Conv1d(128, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, 3, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)

#     def forward(self, xyz, num_samples):
#         xyz = xyz.transpose(1,2).contiguous()
#         l0_points = xyz
#         l0_xyz = xyz[:,:3,:]

#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

#         l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
#         x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.conv3(x)
#         p_idx = furthest_point_sample(x.transpose(1,2).contiguous(), num_samples)
#         point_output = gather_points(x, p_idx).transpose(1,2).contiguous()
#         return point_output

class pointnet_encoder(nn.Module):
    def __init__(self):
        super(pointnet_encoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(4096, 0.1, 0.2, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(2048, 0.2, 0.4, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(1024, 0.4, 0.8, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(256, 0.8, 1.0, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 512])
        self.fp3 = PointNetFeaturePropagation(640, [512, 1024])
        self.fp2 = PointNetFeaturePropagation(1088, [1024, 1024])
        self.fp1 = PointNetFeaturePropagation(1024, [1024, 1024, 1024])
        self.conv1 = nn.Conv1d(1024, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, xyz, num_samples):
        xyz = xyz.transpose(1,2).contiguous()
        l0_points = xyz
        l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        p_idx = furthest_point_sample(x.transpose(1,2).contiguous(), num_samples)
        point_output = gather_points(x, p_idx).transpose(1,2).contiguous()
        return point_output    
class SampleNetEncoder(nn.Module):
    def __init__(self, num_out_points = 2048):
        super(SampleNetEncoder, self).__init__()
        self.num_out_points = num_out_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 3 * self.num_out_points)
        
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.bn_fc3 = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = self.conv5(y)

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y)))
        y = F.relu(self.bn_fc1(self.fc2(y)))
        y = F.relu(self.bn_fc1(self.fc3(y)))
        y = self.fc4(y)

        y = y.view(-1, self.num_out_points, 3)
        return y
class value_weight(nn.Module):
    def __init__(self):
        super(value_weight, self).__init__()
#         self.encoder = PointNetEncoder(channel = 10, global_feat = True)
        self.encoder = pointnet_encoder()
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.bn = nn.Softmax()
        
        self.project = SoftProjection(
                group_size = 25, initial_temperature = 0.04, is_temperature_trainable = True, min_sigma = 1e-4)
    def forward(self, x):#B,N,5,3  /////  B,N,1  // B,N,3
#         local_mean = torch.mean(x_local, dim = 2)
#         local_std = torch.std(x_local, dim = 2)
#         local = torch.cat([x, local_mean, local_std], dim = 2)
#         local = torch.cat([local_mean, local_std, num, x], dim = 2).permute(0,2,1) #1,1,1,3
#         y, trans, trans_feat = self.encoder(local)  #8,1024
        
        sim = self.encoder(x, 2048)

        y = self.project(point_cloud = x, query_cloud = sim)
        return sim, y

class find_num(nn.Module):
    def __init__(self, one):
        super(find_num, self).__init__()
        self.one_view_num = one
    def forward(self, a):
#         help(torch.where)
        zero_matrix = torch.zeros_like(a)#.cuda(a.device)
        one_matrix = torch.ones_like(a)#.cuda(a.device)
        
        b = torch.where(a < self.one_view_num,one_matrix,zero_matrix) + 0
        
        d = torch.where(a < 2 * self.one_view_num,one_matrix,zero_matrix)
        c = torch.where(a > self.one_view_num,one_matrix,zero_matrix)
        e = (c == d) + 0
        
        f = torch.where(a < 3 * self.one_view_num,one_matrix,zero_matrix)
        g = torch.where(a > 2 * self.one_view_num,one_matrix,zero_matrix)
        h = (f == g) + 0

        i = torch.where(a > 3 * self.one_view_num,one_matrix,zero_matrix) + 0
        
        j = b.max(2)[0] + e.max(2)[0] + h.max(2)[0] + i.max(2)[0]
        j = j / 4.
        
        return j.unsqueeze(2)
    
class SA_Layer(nn.Module):
    def __init__(self, channels, is_half = False):
        super().__init__()
        if (is_half == False):
            self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
            self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        else:
            self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
            self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
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
        x_r = (self.act(self.after_norm(self.trans_conv(x - x_r))))
        x = x + x_r
        return x
    
class SA_module(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=16):
        super(SA_module, self).__init__()
        self.share_planes = share_planes
        self.k = k
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

        self.conv_w = nn.Sequential(nn.ReLU(inplace=False),
                                    nn.Conv2d(rel_planes * (k + 1), mid_planes // share_planes, kernel_size=1,
                                              bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(mid_planes // share_planes, k * mid_planes // share_planes,
                                              kernel_size=1))
        self.activation_fn = nn.ReLU(inplace=False)

        self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

    def forward(self, input):
        x, idx = input
        batch_size, _, _, num_points = x.size()
        identity = x  # B C 1 N
        x = self.activation_fn(x)
        xn = get_edge_features(x, idx)  # B C K N
        x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

        x2 = x2.view(batch_size, -1, 1, num_points).contiguous()  # B kC 1 N
        w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k, num_points)
        w = w.repeat(1, self.share_planes, 1, 1)
        out = w * x3
        out = torch.sum(out, dim=2, keepdim=True)

        out = self.activation_fn(out)
        out = self.conv_out(out)  # B C 1 N
        out = out + identity
        return [out, idx]


class Folding(nn.Module):
    def __init__(self, input_size, output_size, step_ratio, global_feature_size=1024, num_models=1):
        super(Folding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.step_ratio = step_ratio
        self.num_models = num_models

        self.conv = nn.Conv1d(input_size + global_feature_size + 2, output_size, 1, bias=True)

        sqrted = int(math.sqrt(step_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (step_ratio % i) == 0:
                num_x = i
                num_y = step_ratio // i
                break

        grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
        grid_y = torch.linspace(-0.2, 0.2, steps=num_y)

        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)

    def forward(self, point_feat, global_feat):
        batch_size, num_features, num_points = point_feat.size()
        point_feat = point_feat.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(
            batch_size,
            -1, num_features).transpose(1, 2).contiguous()
        global_feat = global_feat.unsqueeze(2).repeat(1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
        grid_feat = self.grid.unsqueeze(0).repeat(batch_size, num_points, 1).transpose(1, 2).contiguous().cuda(point_feat.device)
        features = torch.cat([global_feat, point_feat, grid_feat], axis=1)
        features = F.relu(self.conv(features))
        return features


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.afff = nn.ReLU()

    def forward(self, feature_input):
        a = self.conv_res(feature_input)
        
        b = self.afff(feature_input)
        b = self.conv1(b)
        b = self.afff(b)
        b = self.conv2(b)
        res = a+b
        return res


class SK_SA_module(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=[10, 20], r=2, L=32):
        super(SK_SA_module, self).__init__()

        self.num_kernels = len(k)
        d = max(int(out_planes / r), L)

        self.sams = nn.ModuleList([])

        for i in range(len(k)):
            self.sams.append(SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k[i]))

        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])

        for i in range(len(k)):
            self.fcs.append(nn.Linear(d, out_planes))

        self.softmax = nn.Softmax(dim=1)
        self.af = nn.ReLU(inplace=False)

    def forward(self, input):
        x, idxs = input
        assert (self.num_kernels == len(idxs))
        for i, sam in enumerate(self.sams):
            fea, _ = sam([x, idxs[i]])
            fea = self.af(fea)
            fea = fea.unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return [fea_v, idxs]


class SKN_Res_unit(nn.Module):
    def __init__(self, input_size, output_size, k=[10, 20], layers=1):
        super(SKN_Res_unit, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.sam = self._make_layer(output_size, output_size // 16, output_size // 4, output_size, int(layers), 8, k=k)
        self.conv2 = nn.Conv2d(output_size, output_size, 1, bias=False)
        self.conv_res = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.af = nn.ReLU(inplace=False)

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def forward(self, feat, idx):
        x, _ = self.sam([self.conv1(feat), idx])
        x = self.conv2(self.af(x))
        return x + self.conv_res(feat)


class SA_SKN_Res_encoder(nn.Module):
    def __init__(self, input_size=3, k=[10, 20], pk=16, output_size=64, layers=[2, 2, 2, 2],
                 pts_num=[3072, 1536, 768, 384]):
        super(SA_SKN_Res_encoder, self).__init__()
        self.init_channel = 32

        c1 = self.init_channel
        self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))

        c2 = c1 * 2
        self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))

        c3 = c2 * 2
        self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))

        c4 = c3 * 2
        self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))

        self.conv5 = nn.Conv2d(c4, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 256)

        self.conv6 = nn.Conv2d(c4 + 256, c4, 1)
        self.conv7 = nn.Conv2d(c3 + c4, c3, 1)
        self.conv8 = nn.Conv2d(c2 + c3, c2, 1)
        self.conv9 = nn.Conv2d(c1 + c2, c1, 1)

        self.conv_out = nn.Conv2d(c1, output_size, 1)
        self.dropout = nn.Dropout()
        self.af = nn.ReLU(inplace=False)
        self.k = k
        self.pk = pk
        self.rate = 1.2

        self.pts_num = pts_num

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def _edge_pooling(self, features, points, rate=1.2, k=16, sample_num=None):
        features = features.squeeze(2)

        if sample_num is None:
            input_points_num = int(features.size()[2])
            sample_num = int(input_points_num // rate)
        ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, k)
        ds_features = ds_features.unsqueeze(2)
        return ds_features, p_idx, pn_idx, ds_points

    def _edge_unpooling(self, features, src_pts, tgt_pts):
        features = features.squeeze(2)
        idx, weight = three_nn_upsampling(tgt_pts, src_pts)
        features = three_interpolate(features, idx, weight)
        features = features.unsqueeze(2)
        return features

    def forward(self, features):
        batch_size, _, num_points = features.size()
        pt1 = features[:, 0:3, :]

        idx1 = []
        for i in range(len(self.k)):
            idx = knn(pt1, self.k[i])
            idx1.append(idx)

        pt1 = pt1.transpose(1, 2).contiguous()

        x = features.unsqueeze(2)
        x = self.sam_res1(x, idx1)
        x1 = self.af(x)
        
        x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk, None)
        idx2 = []
        for i in range(len(self.k)):
            idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
            idx2.append(idx)

        x = self.sam_res2(x, idx2)
        x2 = self.af(x)

        x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk, None)
        idx3 = []
        for i in range(len(self.k)):
            idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
            idx3.append(idx)

        x = self.sam_res3(x, idx3)
        x3 = self.af(x)

        x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk, None)
        idx4 = []
        for i in range(len(self.k)):
            idx = knn(pt4.transpose(1, 2).contiguous(), self.k[i])
            idx4.append(idx)

        x = self.sam_res4(x, idx4)
        x4 = self.af(x)
        x = self.conv5(x4)
        x, _ = torch.max(x, -1)
        x = x.view(batch_size, -1)
        x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))
        x = x.unsqueeze(2).repeat(1, 1, 11).unsqueeze(2)
        
        x = self.af(self.conv6(torch.cat([x, x4], 1)))
        x = self._edge_unpooling(x, pt4, pt3)
        x = self.af(self.conv7(torch.cat([x, x3], 1)))
        x = self._edge_unpooling(x, pt3, pt2)
        x = self.af(self.conv8(torch.cat([x, x2], 1)))
        x = self._edge_unpooling(x, pt2, pt1)
        x = self.af(self.conv9(torch.cat([x, x1], 1)))
        x = self.conv_out(x)
        x = x.squeeze(2)
        return x


    
    
    
    
    
    
    
    
    
    
    
    
    
class MSAP_SKN_decoder(nn.Module):
    def __init__(self, num_coarse_raw, num_fps, num_coarse, num_fine, layers=[2, 2, 2, 2], knn_list=[10, 20], pk=10,
                 points_label=False, local_folding=False):
        super(MSAP_SKN_decoder, self).__init__()
        self.num_coarse_raw = num_coarse_raw
        self.num_fps = num_fps
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.points_label = points_label
        self.local_folding = local_folding

        self.conv1 = nn.Conv1d(512, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, 75, 1)
        self.conv4 = nn.Conv1d(78, 75, 1)
        self.conv5 = nn.Conv1d(150, 3, 1)
        self.att1 = SA_Layer(256)
        self.att2 = SA_Layer(64)
        self.att3 = SA_Layer(64)
        self.att4 = SA_Layer(3)
        self.att1__ = SA_Layer(150)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        
        self.dense_feature_size = 256
        self.expand_feature_size = 64
        self.bn_res = nn.BatchNorm1d(self.expand_feature_size)

        self.input_size = 3
        self.encoder = PCN_encoder_2(output_size = self.dense_feature_size)#SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk,
#                                           #output_size=self.dense_feature_size, layers=layers)
#         self.encoder = SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk,
#                                           output_size=self.dense_feature_size, layers=layers)
            
            
        self.up_scale = int(np.ceil(num_fine / (num_coarse_raw + 2048)))

        self.prior_infer_1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        
        self.prior_infer_2 = nn.Conv1d(self.expand_feature_size, 64, 1)
        
        self.prior_infer_3 = nn.Conv1d(64, 3, 1)
        
        self.prior_infer_4 = nn.Conv1d(64, 3, 1)
        
        self.bn_ = nn.BatchNorm1d(self.expand_feature_size)
        self.bn__ = nn.BatchNorm1d(64)
        self.bn___ = nn.BatchNorm1d(3)

        self.project = SoftProjection(
                group_size = 10, initial_temperature = 0.1, is_temperature_trainable = True, min_sigma = 1e-4)
        self.project_2 = SoftProjection(
                group_size = 5, initial_temperature = 0.2, is_temperature_trainable = True, min_sigma = 1e-4)
        self.af = nn.ReLU(inplace=False)
        
    def forward(self, global_feat, point_input, x, gt):
        batch_size = global_feat.size()[0]
        global_feat = self.af(self.bn1(self.conv1(global_feat)))
        coarse_raw = self.af(self.bn2(self.conv2(global_feat)))
        coarse_raw = self.att1(coarse_raw)
        coarse_raw = self.conv3(coarse_raw)
        points = torch.cat([coarse_raw, point_input[:,:75,:]], 1)
        
        coarse_raw = self.att1__(points)
        coarse_raw = self.conv5(coarse_raw)
        coarse_raw = coarse_raw + point_input[:,-3:,:]
        org_points_input = point_input
        coarse_input = coarse_raw

        coarse_mid = coarse_raw
        
        points = torch.cat([coarse_mid, org_points_input[:,-3:,:]], 1)
        
        dense_x, dense_feat = self.encoder(points)

        coarse = self.prior_infer_4(dense_x).transpose(2,1).contiguous()

        coarse = self.project_2(point_cloud = x, query_cloud = coarse)

        return coarse_raw.transpose(2,1).contiguous(), coarse_mid.transpose(2,1).contiguous(), coarse.transpose(2,1).contiguous() # 2,2048,2,3

    
class Model(nn.Module):
    def __init__(self, args, type_, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]
        self.type = type_
        self.size_z = size_z
        self.distribution_loss = args.distribution_loss
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.encoder = PCN_encoder(output_size = 128)
        self.posterior_infer1 = Linear_ResBlock(input_size=256, output_size= 256)
        self.posterior_infer2 = Linear_ResBlock(input_size=256, output_size=size_z * 2)
        
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        
        self.generator = Linear_ResBlock(input_size=size_z, output_size = 256)
        self.decoder = MSAP_SKN_decoder(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse,
                                        num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list,
                                        pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)
        self.find_num = find_num(one = 2048)
#         self.value_weight = value_weight()
    def compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]

        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / float(dim))

    def mmd_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x, random_x, gt=None, transforms=None, label = None,prefix="train", mean_feature=None, alpha=None):
        B,N,C = x.size()
#         x = x[:,:4096,:]
        x = x.transpose(1, 2).contiguous()
        gt = gt.contiguous()
        # x.shape = 6,3,8192
        # gt.shape = 6,3,2048
        
        
        p_idx = furthest_point_sample(x.contiguous(), 2048)   #0.607799
        pred_center = gather_points(x.transpose(2,1).contiguous(), p_idx).permute(0,2,1)
        _, gt_idx = knn_point(25, x, pred_center)
        x_local__ = index_points(x, gt_idx)
        x_local = torch.cat((x_local__, pred_center.unsqueeze(2)), dim = 2)
        
        x_local_ = x_local.reshape((x_local.shape[0], x_local.shape[1], -1)).transpose(1, 2).contiguous() #batch*2048,3,5
        local_x, global_feat = self.encoder(x_local_)  #batch*2048,5,3
        o_x = self.posterior_infer2(self.posterior_infer1(global_feat))
        q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
        q_std = F.softplus(q_std)
        ori_distribution = torch.distributions.Normal(q_mu, q_std)  # 输入不完整的正态分布
        
        standard_distribution = torch.distributions.Normal(torch.zeros_like(q_mu), torch.ones_like(q_std)) # GT的正态分布
        z = ori_distribution.rsample()
        z = z.reshape((local_x.shape[0],local_x.shape[2],z.shape[1]))
        z = self.generator(z).transpose(2,1)
        feat = torch.cat([local_x, z.contiguous()], dim = 1) # local feature
#         feat = local_x + z.contiguous()
        
        

        _, gt_idx = knn_point(1, gt, pred_center)
        gt_local_ = index_points(gt, gt_idx).squeeze()
        _, idx = knn_point(25, gt, gt_local_)
        gt_local = index_points(gt, idx).squeeze()
        gt_local = torch.cat((gt_local, gt_local_.unsqueeze(2)), dim = 2)
        gt_local = gt_local.reshape((gt_local.shape[0], gt_local.shape[1], -1)).transpose(1, 2).contiguous() #batch*2048,3,5
        local_gt, gt_feat = self.encoder(gt_local)
        gt_x = self.posterior_infer2(self.posterior_infer1(gt_feat))
        gt_mu, gt_std = torch.split(gt_x, self.size_z, dim=1)
        gt_std = F.softplus(gt_std)
        gt_distribution = torch.distributions.Normal(gt_mu, gt_std)  # 输入不完整的正态分布
        
        plus_raw, plus_mid, plus_local = self.decoder(feat, x_local_, x, gt)
#         plus_local = plus_local.reshape(x_local__.shape)
#         plus_raw = plus_raw.reshape(x_local__.shape) 
        
        mean_local_2 = torch.mean((x_local__), dim = 2)
        mean_local_3 = plus_raw
        mean_local = plus_local # batch,2048,3    0.956426    

        if prefix=="train":
            if self.distribution_loss == 'MMD':
                z_m = m_distribution.rsample()
                z_q = q_distribution.rsample()
                z_p = p_distribution.rsample()
                z_p_fix = p_distribution_fix.rsample()
                dl_rec = self.mmd_loss(z_m, z_p)
                dl_g = self.mmd_loss2(z_q, z_p_fix)
            elif self.distribution_loss == 'KLD':
#                 pass
                dl_rec = torch.distributions.kl_divergence(gt_distribution, standard_distribution)
                dl_view = torch.distributions.kl_divergence(ori_distribution, gt_distribution)
                
            else:
                raise NotImplementedError('Distribution loss is either MMD or KLD')

            if self.train_loss == 'cd':
                gt = gt.contiguous()
                pred_center = pred_center.contiguous()
                mean_local = mean_local.contiguous()
                mean_local_2 = mean_local_2.contiguous()
                mean_local_3 = mean_local_3.contiguous()
                
                
#                 print(pred_center.shape)
#                 print(mean_local.shape)
#                 print(mean_local_2.shape)
#                 print(mean_local_3.shape)
#                 print(dsads)
                proj_loss_1 = get_projection_loss(self.decoder.project)
                proj_loss_2 = get_projection_loss(self.decoder.project_2)
#                 loss3, _, f1_ = calc_cd(pred_sim, x)
                loss2, _, f1 = calc_cd(pred_center, gt) # 生成点云和GT点云的chamfer
                loss4, _, f1___ = calc_cd(mean_local, gt)
                _, _, f1____ = calc_cd(plus_mid, gt)
            
                loss3, _, f1__ = calc_cd(mean_local_3, gt)
                _, _, f1_ = calc_cd(mean_local_2, gt)
#                 total_train_loss = (loss3.mean() + proj_loss.mean() * 0.01 + loss2.mean() * 0.1) + (loss4.mean() + dl_rec.mean() * 20 + dl_view.mean() * 20) #+ dl_rec.mean() * 20 + dl_view.mean() * 20
                total_train_loss = proj_loss_1.mean() + proj_loss_2.mean() + loss3.mean() * alpha[0] + loss4.mean() + dl_rec.mean() * 10 + dl_view.mean() * 10
#                 print(loss2)
#                 print(loss3)
#                 print(loss4)
                if(torch.isnan(total_train_loss.mean())):
                    print(loss2)
                    print(loss3)
                    print(loss4)
                    print(proj_loss)
                    print(dsadsa)

#                 print_loss = []
                print_loss = [loss4.mean().data.cpu(),]#, loss3.mean().data.cpu(), dl_rec.mean().data.cpu(), dl_view.mean().data.cpu()]

            else:
                raise NotImplementedError('Only CD is supported')
            
            return (pred_center,mean_local), (f1,f1_,f1__,f1____,f1___), total_train_loss, print_loss
        elif prefix=="val":
            gt = gt.contiguous()
            pred_center = pred_center.contiguous()
            mean_local = mean_local.contiguous()
#             loss3 = self.crit(top, gt_top)
#             point_output = point_output.transpose(1,2).contiguous()
            if self.eval_emd:
                emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(mean_local, gt)
#             print(cd_p)
#             cd_p, cd_t, f1 = calc_cd(point_output, gt, calc_f1=True)
#             print(f1_.mean())
#             print(f1.mean())
#             print('diff:',f1_.mean() - f1.mean())
#             print(dsadsada)
            return {'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1, 'result': mean_local}
        elif prefix=="test":
            return {'result': fine}


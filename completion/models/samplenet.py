# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# import torch.nn.functional as F
# # from utils.model_utils import *
# from model_utils import *
# from models.pcn import PCN_encoder
# from vis_utils import plot_single_pcd
# import open3d as o3d
# import random
# sys.path.append("../utils")
# from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation


# class GRU(nn.Module):
#     def __init__(self, h_size, c_size):
#         super(GRU, self).__init__()
#         self.i_t_x = nn.Linear(h_size, c_size)
#         self.i_t_h = nn.Linear(h_size, c_size)
        
#         self.f_t_x = nn.Linear(h_size, c_size)
#         self.f_t_h = nn.Linear(h_size, c_size)
        
#         self.g_t_x = nn.Linear(h_size, c_size)
#         self.g_t_h = nn.Linear(h_size, c_size)
        
#         self.o_t_x = nn.Linear(h_size, h_size)
#         self.o_t_h = nn.Linear(h_size, h_size)
        
#         self.c_t_h = nn.Linear(c_size, h_size)
        
#         self.tanh = nn.Tanh()
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, input, h, c):
        
#         x = F.adaptive_avg_pool1d(input.permute(0,2,1), 1).squeeze()
        
#         i_t_x = self.i_t_x(x)
#         i_t_h = self.i_t_h(h)
#         i_t = self.sigmoid(i_t_x + i_t_h) # forget gate
        
#         f_t_x = self.f_t_x(x) #4,3,8192 -> 4,1024
#         f_t_h = self.f_t_h(h)
#         f_t = self.sigmoid(f_t_x + f_t_h)
        
#         g_t_x = self.g_t_x(x)
#         g_t_h = self.g_t_h(h)
#         g_t = self.tanh(g_t_x + g_t_h) 
        
#         o_t_x = self.o_t_x(x)
#         o_t_h = self.o_t_h(h)
#         o_t = self.sigmoid(o_t_x + o_t_h)
        
#         c_t = f_t * c + i_t * g_t
#         c_t_h = self.c_t_h(c_t)
        
#         h_t = o_t * torch.tanh(c_t_h)
        
#         return h_t, c_t
# class SA_module(nn.Module):
#     def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=16):
#         super(SA_module, self).__init__()
#         self.share_planes = share_planes
#         self.k = k
#         self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
#         self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

#         self.conv_w = nn.Sequential(nn.ReLU(inplace=False),
#                                     nn.Conv2d(rel_planes * (k + 1), mid_planes // share_planes, kernel_size=1,
#                                               bias=False),
#                                     nn.ReLU(inplace=False),
#                                     nn.Conv2d(mid_planes // share_planes, k * mid_planes // share_planes,
#                                               kernel_size=1))
#         self.activation_fn = nn.ReLU(inplace=False)

#         self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

#     def forward(self, input):
#         x, idx = input
#         batch_size, _, _, num_points = x.size()
#         identity = x  # B C 1 N
#         x = self.activation_fn(x)
#         xn = get_edge_features(x, idx)  # B C K N
#         x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

#         x2 = x2.view(batch_size, -1, 1, num_points).contiguous()  # B kC 1 N
#         w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k, num_points)
#         w = w.repeat(1, self.share_planes, 1, 1)
#         out = w * x3
#         out = torch.sum(out, dim=2, keepdim=True)

#         out = self.activation_fn(out)
#         out = self.conv_out(out)  # B C 1 N
#         out = out + identity
#         return [out, idx]


# class Folding(nn.Module):
#     def __init__(self, input_size, output_size, step_ratio, global_feature_size=1024, num_models=1):
#         super(Folding, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.step_ratio = step_ratio
#         self.num_models = num_models

#         self.conv = nn.Conv1d(input_size + global_feature_size + 2, output_size, 1, bias=True)

#         sqrted = int(math.sqrt(step_ratio)) + 1
#         for i in range(1, sqrted + 1).__reversed__():
#             if (step_ratio % i) == 0:
#                 num_x = i
#                 num_y = step_ratio // i
#                 break

#         grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
#         grid_y = torch.linspace(-0.2, 0.2, steps=num_y)

#         x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
#         self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)

#     def forward(self, point_feat, global_feat):
#         batch_size, num_features, num_points = point_feat.size()
#         point_feat = point_feat.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(
#             batch_size,
#             -1, num_features).transpose(1, 2).contiguous()
#         global_feat = global_feat.unsqueeze(2).repeat(1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
#         grid_feat = self.grid.unsqueeze(0).repeat(batch_size, num_points, 1).transpose(1, 2).contiguous().cuda(point_feat.device)
#         features = torch.cat([global_feat, point_feat, grid_feat], axis=1)
#         features = F.relu(self.conv(features))
#         return features


# class Linear_ResBlock(nn.Module):
#     def __init__(self, input_size=1024, output_size=256):
#         super(Linear_ResBlock, self).__init__()
#         self.conv1 = nn.Linear(input_size, input_size)
#         self.conv2 = nn.Linear(input_size, output_size)
#         self.conv_res = nn.Linear(input_size, output_size)

#         self.afff = nn.ReLU()

#     def forward(self, feature_input):
#         a = self.conv_res(feature_input)
        
#         b = self.afff(feature_input)
#         b = self.conv1(b)
#         b = self.afff(b)
#         b = self.conv2(b)
#         res = a+b
#         return res


# class SK_SA_module(nn.Module):
#     def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=[10, 20], r=2, L=32):
#         super(SK_SA_module, self).__init__()

#         self.num_kernels = len(k)
#         d = max(int(out_planes / r), L)

#         self.sams = nn.ModuleList([])

#         for i in range(len(k)):
#             self.sams.append(SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k[i]))

#         self.fc = nn.Linear(out_planes, d)
#         self.fcs = nn.ModuleList([])

#         for i in range(len(k)):
#             self.fcs.append(nn.Linear(d, out_planes))

#         self.softmax = nn.Softmax(dim=1)
#         self.af = nn.ReLU(inplace=False)

#     def forward(self, input):
#         x, idxs = input
#         assert (self.num_kernels == len(idxs))
#         for i, sam in enumerate(self.sams):
#             fea, _ = sam([x, idxs[i]])
#             fea = self.af(fea)
#             fea = fea.unsqueeze(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)

#         fea_U = torch.sum(feas, dim=1)
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)

#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)

#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return [fea_v, idxs]


# class SKN_Res_unit(nn.Module):
#     def __init__(self, input_size, output_size, k=[10, 20], layers=1):
#         super(SKN_Res_unit, self).__init__()
#         self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
#         self.sam = self._make_layer(output_size, output_size // 16, output_size // 4, output_size, int(layers), 8, k=k)
#         self.conv2 = nn.Conv2d(output_size, output_size, 1, bias=False)
#         self.conv_res = nn.Conv2d(input_size, output_size, 1, bias=False)
#         self.af = nn.ReLU(inplace=False)

#     def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
#         layers = []
#         for _ in range(0, blocks):
#             layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
#         return nn.Sequential(*layers)

#     def forward(self, feat, idx):
#         x, _ = self.sam([self.conv1(feat), idx])
#         x = self.conv2(self.af(x))
#         return x + self.conv_res(feat)


# class SA_SKN_Res_encoder(nn.Module):
#     def __init__(self, input_size=3, k=[10, 20], pk=16, output_size=64, layers=[2, 2, 2, 2],
#                  pts_num=[3072, 1536, 768, 384]):
#         super(SA_SKN_Res_encoder, self).__init__()
#         self.init_channel = 64

#         c1 = self.init_channel
#         self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))

#         c2 = c1 * 2
#         self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))

#         c3 = c2 * 2
#         self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))

#         c4 = c3 * 2
#         self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))

#         self.conv5 = nn.Conv2d(c4, 1024, 1)

#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 1024)

#         self.conv6 = nn.Conv2d(c4 + 1024, c4, 1)
#         self.conv7 = nn.Conv2d(c3 + c4, c3, 1)
#         self.conv8 = nn.Conv2d(c2 + c3, c2, 1)
#         self.conv9 = nn.Conv2d(c1 + c2, c1, 1)

#         self.conv_out = nn.Conv2d(c1, output_size, 1)
#         self.dropout = nn.Dropout()
#         self.af = nn.ReLU(inplace=False)
#         self.k = k
#         self.pk = pk
#         self.rate = 2

#         self.pts_num = pts_num

#     def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
#         layers = []
#         for _ in range(0, blocks):
#             layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
#         return nn.Sequential(*layers)

#     def _edge_pooling(self, features, points, rate=2, k=16, sample_num=None):
#         features = features.squeeze(2)

#         if sample_num is None:
#             input_points_num = int(features.size()[2])
#             sample_num = input_points_num // rate
#         ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, k)
#         ds_features = ds_features.unsqueeze(2)
#         return ds_features, p_idx, pn_idx, ds_points

#     def _edge_unpooling(self, features, src_pts, tgt_pts):
#         features = features.squeeze(2)
#         idx, weight = three_nn_upsampling(tgt_pts, src_pts)
#         features = three_interpolate(features, idx, weight)
#         features = features.unsqueeze(2)
#         return features

#     def forward(self, features):
#         batch_size, _, num_points = features.size()
#         pt1 = features[:, 0:3, :]

#         idx1 = []
#         for i in range(len(self.k)):
#             idx = knn(pt1, self.k[i])
#             idx1.append(idx)

#         pt1 = pt1.transpose(1, 2).contiguous()

#         x = features.unsqueeze(2)
#         x = self.sam_res1(x, idx1)
#         x1 = self.af(x)

#         x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk, self.pts_num[1])
#         idx2 = []
#         for i in range(len(self.k)):
#             idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
#             idx2.append(idx)

#         x = self.sam_res2(x, idx2)
#         x2 = self.af(x)

#         x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk, self.pts_num[2])
#         idx3 = []
#         for i in range(len(self.k)):
#             idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
#             idx3.append(idx)

#         x = self.sam_res3(x, idx3)
#         x3 = self.af(x)

#         x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk, self.pts_num[3])
#         idx4 = []
#         for i in range(len(self.k)):
#             idx = knn(pt4.transpose(1, 2).contiguous(), self.k[i])
#             idx4.append(idx)

#         x = self.sam_res4(x, idx4)
#         x4 = self.af(x)
#         x = self.conv5(x4)
#         x, _ = torch.max(x, -1)
#         x = x.view(batch_size, -1)
#         x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))

#         x = x.unsqueeze(2).repeat(1, 1, self.pts_num[3]).unsqueeze(2)
        
#         x = self.af(self.conv6(torch.cat([x, x4], 1)))
#         x = self._edge_unpooling(x, pt4, pt3)
#         x = self.af(self.conv7(torch.cat([x, x3], 1)))
#         x = self._edge_unpooling(x, pt3, pt2)
#         x = self.af(self.conv8(torch.cat([x, x2], 1)))
#         x = self._edge_unpooling(x, pt2, pt1)
#         x = self.af(self.conv9(torch.cat([x, x1], 1)))
#         x = self.conv_out(x)
#         x = x.squeeze(2)
#         return x


    
    
    
    
    
    
    
    
    
    
    
    
    
# class MSAP_SKN_decoder(nn.Module):
#     def __init__(self, num_coarse_raw, num_fps, num_coarse, num_fine, layers=[2, 2, 2, 2], knn_list=[10, 20], pk=10,
#                  points_label=False, local_folding=False):
#         super(MSAP_SKN_decoder, self).__init__()
#         self.num_coarse_raw = num_coarse_raw
#         self.num_fps = num_fps
#         self.num_coarse = num_coarse
#         self.num_fine = num_fine
#         self.points_label = points_label
#         self.local_folding = local_folding

#         self.fc1 = nn.Linear(1024, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, num_coarse_raw * 3)

#         self.dense_feature_size = 256
#         self.expand_feature_size = 64

#         if points_label:
#             self.input_size = 4
#         else:
#             self.input_size = 3


#         self.encoder = SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk,
#                                           output_size=self.dense_feature_size, layers=layers)

#         self.up_scale = int(np.ceil(num_fine / (num_coarse_raw + 2048)))

#         if self.up_scale >= 2:
#             self.expansion1 = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
#                                            step_ratio=self.up_scale, k=4)
#             self.conv_cup1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
#         else:
#             self.expansion1 = None
#             self.conv_cup1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)

#         self.conv_cup2 = nn.Conv1d(self.expand_feature_size, 3, 1, bias=True)

#         self.conv_s1 = nn.Conv1d(self.expand_feature_size, 16, 1, bias=True)
#         self.conv_s2 = nn.Conv1d(16, 8, 1, bias=True)
#         self.conv_s3 = nn.Conv1d(8, 1, 1, bias=True)

#         if self.local_folding:
#             self.expansion2 = Folding(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
#                                       step_ratio=(num_fine // num_coarse))
#         else:
#             self.expansion2 = EF_expansion(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
#                                            step_ratio=(num_fine // num_coarse), k=4)

#         self.conv_f1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
#         self.conv_f2 = nn.Conv1d(self.expand_feature_size, 3, 1)

#         self.af = nn.ReLU(inplace=False)

#     def forward(self, global_feat, point_input):
        
#         batch_size = global_feat.size()[0]

#         coarse_raw = self.fc3(self.af(self.fc2(self.af(self.fc1(global_feat))))).view(batch_size, 3,
#                                                                                       self.num_coarse_raw)

#         input_points_num = point_input.size()[2]
#         org_points_input = point_input
#         if self.points_label:
#             id0 = torch.zeros(coarse_raw.shape[0], 1, coarse_raw.shape[2]).cuda(point_input.device).contiguous()
#             coarse_input = torch.cat( (coarse_raw, id0), 1)
#             id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda(point_input.device).contiguous()
#             org_points_input = torch.cat( (org_points_input, id1), 1)
#         else:
#             coarse_input = coarse_raw
#         points = torch.cat((coarse_input, org_points_input), 2)
#         dense_feat = self.encoder(points)
#         if self.up_scale >= 2:
#             dense_feat = self.expansion1(dense_feat)
        
#         coarse_features = self.af(self.conv_cup1(dense_feat))
#         coarse_high = self.conv_cup2(coarse_features)
#         if coarse_high.size()[2] > self.num_fps:
#             idx_fps = furthest_point_sample(coarse_high.transpose(1, 2).contiguous(), self.num_fps)
#             coarse_fps = gather_points(coarse_high, idx_fps)
#             coarse_features = gather_points(coarse_features, idx_fps)
#         else:
#             coarse_fps = coarse_high

#         if coarse_fps.size()[2] > self.num_coarse:
#             scores = F.softplus(self.conv_s3(self.af(self.conv_s2(self.af(self.conv_s1(coarse_features))))))
#             idx_scores = scores.topk(k=self.num_coarse, dim=2)[1].view(batch_size, -1).int()
#             coarse = gather_points(coarse_fps, idx_scores)
#             coarse_features = gather_points(coarse_features, idx_scores)
#         else:
#             coarse = coarse_fps

#         if coarse.size()[2] < self.num_fine:
#             if self.local_folding:
#                 up_features = self.expansion2(coarse_features, global_feat)
#                 center = coarse.transpose(2, 1).contiguous().unsqueeze(2).repeat(1, 1, self.num_fine // self.num_coarse,
#                                                                                  1).view(batch_size, self.num_fine,
#                                                                                          3).transpose(2, 1).contiguous()
#                 fine = self.conv_f2(self.af(self.conv_f1(up_features))) + center
#             else:
#                 up_features = self.expansion2(coarse_features)
#                 fine = self.conv_f2(self.af(self.conv_f1(up_features)))
#         else:
# #             assert (coarse.size()[2] == self.num_fine)
#             fine = coarse
#         return coarse_raw, coarse_high, coarse, fine

    
# class Model(nn.Module):
#     def __init__(self, args, type_, size_z=128, global_feature_size=1024):
#         super(Model, self).__init__()

#         layers = [int(i) for i in args.layers.split(',')]
#         knn_list = [int(i) for i in args.knn_list.split(',')]
#         self.type = type_
#         self.size_z = size_z
#         self.distribution_loss = args.distribution_loss
#         self.train_loss = args.loss
#         self.eval_emd = args.eval_emd
#         self.encoder = PCN_encoder(output_size=global_feature_size)
#         self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
#         self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        
#         self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        
#         self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size)
#         self.decoder = MSAP_SKN_decoder(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse,
#                                         num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list,
#                                         pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)
#         self.gru = GRU(h_size = 8192, c_size = 1024)
        
#     def compute_kernel(self, x, y):
#         x_size = x.size()[0]
#         y_size = y.size()[0]
#         dim = x.size()[1]

#         tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
#         tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
#         return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / float(dim))

#     def mmd_loss(self, x, y):
#         x_kernel = self.compute_kernel(x, x)
#         y_kernel = self.compute_kernel(y, y)
#         xy_kernel = self.compute_kernel(x, y)
#         return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

#     def forward(self, x, gt=None, label = None,prefix="train", mean_feature=None, alpha=None):
#         B,N,C = x.size()
#         gt = gt.transpose(1, 2).contiguous()
#         # x.shape = 6,3,8192
#         # gt.shape = 6,3,2048
        
        
#         if prefix=="train":
#             feat_y = self.encoder(gt)
#             o_y = self.posterior_infer2(self.posterior_infer1(feat_y))
#             p_mu, p_std = torch.split(o_y, self.size_z, dim=1)
#             p_std = F.softplus(p_std)
#             gt_distribution = torch.distributions.Normal(p_mu, p_std)  #gt的正态分布
#             standard_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))  # 标准正态分布
#             z = gt_distribution.rsample()  #(2,128)
#             feat = feat_y
#         else:
#             feat_x = self.encoder(x)
#             o_x = self.posterior_infer2(self.posterior_infer1(feat_x))
#             q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
#             q_std = F.softplus(q_std)
#             ori_distribution = torch.distributions.Normal(q_mu, q_std)  #输入不完整的正态分布
#             z = ori_distribution.rsample()
#             feat = feat_x
        
#         c_0 = feat + self.generator(z) #2,1024   2,3,8192
#         h_0 = torch.zeros((x.shape[0],x.shape[2])).cuda(x.device) # 2,8192
#         x_0 = x
#         h_1, c_1 = self.gru(input = x_0, h = h_0, c = c_0)
#         _, top_k = h_1.topk(x_0.shape[2] - 2048, dim = 1)
#         x_1 = gather_points(x_0, top_k.int())
#         h_1 = torch.gather(h_1, 1, top_k)
#         feat_x_1 = self.encoder(x_1)
#         o_x_1 = self.posterior_infer2(self.posterior_infer1(feat_x_1))
#         p_mu_1, p_std_1 = torch.split(o_x_1, self.size_z, dim=1)
#         p_std_1 = F.softplus(p_std_1)
#         x_1_distribution = torch.distributions.Normal(p_mu_1, p_std_1)
#         z_1 = x_1_distribution.rsample()
#         feat_x_1 = feat_x_1 + self.generator(z_1)
#         _, _, _, fine_1 = self.decoder(feat_x_1, x_1) # 6144
        
        
        
#         h_1_ = torch.cat([h_1,torch.zeros(h_1.shape[0], h_0.shape[1] - h_1.shape[1]).cuda(x.device)], dim = 1)
#         x_1_ = torch.cat([x_1,torch.zeros(x_1.shape[0], 3, x_0.shape[2] - x_1.shape[2]).cuda(x.device)], dim = 2)
#         h_2, c_2 = self.gru(input = x_1_, h = h_1_, c = c_1)
#         h_2 = h_2[:,:h_1.shape[1]]
#         _, top_k = h_2.topk(x_1.shape[2] - 2048, dim = 1)
#         x_2 = gather_points(x_1, top_k.int())
#         h_2 = torch.gather(h_2, 1, top_k)
#         feat_x_2 = self.encoder(x_2)
#         o_x_2 = self.posterior_infer2(self.posterior_infer1(feat_x_2))
#         p_mu_2, p_std_2 = torch.split(o_x_2, self.size_z, dim=1)
#         p_std_2 = F.softplus(p_std_2)
#         x_2_distribution = torch.distributions.Normal(p_mu_2, p_std_2)
#         z_2 = x_2_distribution.rsample()
#         feat_x_2 = feat_x_2 + self.generator(z_2)
#         _, _, _, fine_2 = self.decoder(feat_x_2, x_2) #4096
        
        
#         h_2_ = torch.cat([h_2,torch.zeros(h_2.shape[0], h_1_.shape[1] - h_2.shape[1]).cuda(x.device)], dim = 1)
#         x_2_ = torch.cat([x_2,torch.zeros(x_2.shape[0], 3, x_1_.shape[2] - x_2.shape[2]).cuda(x.device)], dim = 2)
#         h_3, c_3 = self.gru(input = x_2_, h = h_2_, c = c_2)
#         h_3 = h_3[:,:h_2.shape[1]]
#         _, top_k = h_3.topk(x_2.shape[2] - 2048, dim = 1)
#         x_3 = gather_points(x_2, top_k.int())
#         feat_x_3 = self.encoder(x_3)
#         o_x_3 = self.posterior_infer2(self.posterior_infer1(feat_x_3))
#         p_mu_3, p_std_3 = torch.split(o_x_3, self.size_z, dim=1)
#         p_std_3 = F.softplus(p_std_3)
#         x_3_distribution = torch.distributions.Normal(p_mu_3, p_std_3)
#         z_3 = x_3_distribution.rsample()
#         feat_x_3 = feat_x_3 + self.generator(z_3)
#         _, _, _, fine_3 = self.decoder(feat_x_3, x_3) # 2048
        
#         x_3 = x_3 + fine_1 + fine_2 + fine_3
#         if prefix=="train":
#             if self.distribution_loss == 'MMD':
#                 z_m = m_distribution.rsample()
#                 z_q = q_distribution.rsample()
#                 z_p = p_distribution.rsample()
#                 z_p_fix = p_distribution_fix.rsample()
#                 dl_rec = self.mmd_loss(z_m, z_p)
#                 dl_g = self.mmd_loss2(z_q, z_p_fix)
#             elif self.distribution_loss == 'KLD':
#                 dl_rec = torch.distributions.kl_divergence(gt_distribution, standard_distribution)
#                 dl_view_0 = torch.distributions.kl_divergence(x_1_distribution, gt_distribution)
#                 dl_view_1 = torch.distributions.kl_divergence(x_2_distribution, gt_distribution)
#                 dl_view_2 = torch.distributions.kl_divergence(x_3_distribution, gt_distribution)
                
#             else:
#                 raise NotImplementedError('Distribution loss is either MMD or KLD')

#             if self.train_loss == 'cd':
#                 gt = gt.transpose(1, 2).contiguous()
#                 x_0 = x_0.transpose(1, 2).contiguous()
#                 x_1 = x_1.transpose(1, 2).contiguous()
#                 x_2 = x_2.transpose(1, 2).contiguous()
#                 x_3 = x_3.transpose(1, 2).contiguous()
#                 fine_1 = fine_1.transpose(1, 2).contiguous()
#                 fine_2 = fine_2.transpose(1, 2).contiguous()
#                 fine_3 = fine_3.transpose(1, 2).contiguous()
#                 loss1, _, _ = calc_cd(x_0, gt)
#                 loss2, _, _ = calc_cd(x_1, gt)
#                 loss3, _, _ = calc_cd(x_2, gt)
#                 loss4, _, _ = calc_cd(x_3, gt)
#                 print(111,loss1.mean())
#                 print(222,loss2.mean())
#                 print(333,loss3.mean())
#                 print(444,dl_rec.mean())
#                 print(555,dl_view_0.mean())
#                 print(666,dl_view_1.mean())
#                 print(777,dl_view_2.mean())
#                 print(9,x_1.shape)
#                 print(10,x_2.shape)
#                 print(11,x_3.shape)
#                 print(12,gt.shape)
#                 print(13,gt_distribution)
#                 print(14,x_1_distribution)
#                 print(15,x_2_distribution)
#                 print(16,x_3_distribution)
# #                 loss5, _, _ = calc_cd(fine_1, gt)
# #                 loss6, _, _ = calc_cd(fine_2, gt)
# #                 loss7, _, _ = calc_cd(fine_3, gt)
# #                 print(x_0.shape)
# #                 print(x_1.shape)
# #                 print(x_2.shape)
# #                 print(x_3.shape)
# #                 print(gt.shape)
                
# #                 print(1,loss1.mean())
# #                 print(2,loss2.mean())
# #                 print(3,loss3.mean())
# #                 print(4,loss4.mean())
# #                 print(dsadsda)
# #                 total_train_loss = loss1 + loss2 * 2 + loss3 * 3 + loss4 * 5 + (dl_rec.mean() + dl_view_0.mean() + dl_view_1.mean() + dl_view_2.mean())* 20
#                 total_train_loss = loss1.mean() * 4 + loss2.mean() * 1 + loss3.mean() * 2 + loss4.mean() + (dl_rec.mean() + dl_view_0.mean() + dl_view_1.mean() + dl_view_2.mean())* 20
#                 print_loss = [loss1.mean(),loss2.mean(),loss3.mean(),loss4.mean(),dl_rec.mean(),dl_view_0.mean(),dl_view_1.mean(),dl_view_2.mean()]
#             else:
#                 raise NotImplementedError('Only CD is supported')
                
#             return x_3.permute(0,2,1), loss4, total_train_loss, print_loss
#         elif prefix=="val":
#             gt = gt.transpose(1, 2).contiguous()
#             x_0 = x_0.transpose(1, 2).contiguous()
#             x_1 = x_1.transpose(1, 2).contiguous()
#             x_2 = x_2.transpose(1, 2).contiguous()
#             x_3 = x_3.transpose(1, 2).contiguous()
# #             print(x_3.shape)
# #             print(gt.shape)
# #             print(dsadsda)
#             if self.eval_emd:
#                 emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
#             else:
#                 emd = 0
#             cd_p, cd_t, f1 = calc_cd(x_3, gt, calc_f1=True)
#             return {'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
#         elif prefix=="test":
#             return {'result': fine}

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from utils.model_utils import *
from model_utils import *
from models.pcn import PCN_encoder
from vis_utils import plot_single_pcd
import open3d as o3d
import random
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation


class GRU(nn.Module):
    def __init__(self, h_size, c_size):
        super(GRU, self).__init__()
        self.i_t_x = nn.Linear(h_size, c_size)
        self.i_t_h = nn.Linear(h_size, c_size)
        
        self.f_t_x = nn.Linear(h_size, c_size)
        self.f_t_h = nn.Linear(h_size, c_size)
        
        self.g_t_x = nn.Linear(h_size, c_size)
        self.g_t_h = nn.Linear(h_size, c_size)
        
        self.o_t_x = nn.Linear(h_size, h_size)
        self.o_t_h = nn.Linear(h_size, h_size)
        
        self.c_t_h = nn.Linear(c_size, h_size)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, h, c):
        
        x = F.adaptive_avg_pool1d(input.permute(0,2,1), 1).squeeze()
        
        i_t_x = self.i_t_x(x)
        i_t_h = self.i_t_h(h)
        i_t = self.sigmoid(i_t_x + i_t_h) # forget gate
        
        f_t_x = self.f_t_x(x) #4,3,8192 -> 4,1024
        f_t_h = self.f_t_h(h)
        f_t = self.sigmoid(f_t_x + f_t_h)
        
        g_t_x = self.g_t_x(x)
        g_t_h = self.g_t_h(h)
        g_t = self.tanh(g_t_x + g_t_h) 
        
        o_t_x = self.o_t_x(x)
        o_t_h = self.o_t_h(h)
        o_t = self.sigmoid(o_t_x + o_t_h)
        
        c_t = f_t * c + i_t * g_t
        c_t_h = self.c_t_h(c_t)
        
        h_t = o_t * torch.tanh(c_t_h)
        
        return h_t, c_t
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
        self.init_channel = 64

        c1 = self.init_channel
        self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))

        c2 = c1 * 2
        self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))

        c3 = c2 * 2
        self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))

        c4 = c3 * 2
        self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))

        self.conv5 = nn.Conv2d(c4, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        self.conv6 = nn.Conv2d(c4 + 1024, c4, 1)
        self.conv7 = nn.Conv2d(c3 + c4, c3, 1)
        self.conv8 = nn.Conv2d(c2 + c3, c2, 1)
        self.conv9 = nn.Conv2d(c1 + c2, c1, 1)

        self.conv_out = nn.Conv2d(c1, output_size, 1)
        self.dropout = nn.Dropout()
        self.af = nn.ReLU(inplace=False)
        self.k = k
        self.pk = pk
        self.rate = 2

        self.pts_num = pts_num

    def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
        return nn.Sequential(*layers)

    def _edge_pooling(self, features, points, rate=2, k=16, sample_num=None):
        features = features.squeeze(2)

        if sample_num is None:
            input_points_num = int(features.size()[2])
            sample_num = input_points_num // rate
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

        x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk, self.pts_num[1])
        idx2 = []
        for i in range(len(self.k)):
            idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
            idx2.append(idx)

        x = self.sam_res2(x, idx2)
        x2 = self.af(x)

        x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk, self.pts_num[2])
        idx3 = []
        for i in range(len(self.k)):
            idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
            idx3.append(idx)

        x = self.sam_res3(x, idx3)
        x3 = self.af(x)

        x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk, self.pts_num[3])
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

        x = x.unsqueeze(2).repeat(1, 1, self.pts_num[3]).unsqueeze(2)
        
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

        self.fc1 = nn.Linear(3072, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_coarse_raw * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64

        if points_label:
            self.input_size = 4
        else:
            self.input_size = 3


        self.encoder = SA_SKN_Res_encoder(input_size=self.input_size, k=knn_list, pk=pk,
                                          output_size=self.dense_feature_size, layers=layers)

        self.up_scale = int(np.ceil(num_fine / (num_coarse_raw + 2048)))

        if self.up_scale >= 2:
            self.expansion1 = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                           step_ratio=self.up_scale, k=4)
            self.conv_cup1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion1 = None
            self.conv_cup1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)

        self.conv_cup2 = nn.Conv1d(self.expand_feature_size, 3, 1, bias=True)

        self.conv_s1 = nn.Conv1d(self.expand_feature_size, 16, 1, bias=True)
        self.conv_s2 = nn.Conv1d(16, 8, 1, bias=True)
        self.conv_s3 = nn.Conv1d(8, 1, 1, bias=True)

        if self.local_folding:
            self.expansion2 = Folding(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
                                      step_ratio=(num_fine // num_coarse))
        else:
            self.expansion2 = EF_expansion(input_size=self.expand_feature_size, output_size=self.dense_feature_size,
                                           step_ratio=(num_fine // num_coarse), k=4)

        self.conv_f1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv_f2 = nn.Conv1d(self.expand_feature_size, 3, 1)

        self.af = nn.ReLU(inplace=False)

    def forward(self, global_feat, point_input):
        
        batch_size = global_feat.size()[0]

        coarse_raw = self.fc3(self.af(self.fc2(self.af(self.fc1(global_feat))))).view(batch_size, 3,self.num_coarse_raw)

        org_points_input = point_input
        id0 = torch.zeros(coarse_raw.shape[0], 1, coarse_raw.shape[2]).cuda(point_input.device).contiguous()
        coarse_input = torch.cat( (coarse_raw, id0), 1)
        id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda(point_input.device).contiguous()
        org_points_input = torch.cat( (org_points_input, id1), 1)

        points = torch.cat((coarse_input, org_points_input), 2)
        dense_feat = self.encoder(points)
        coarse_features = self.af(self.conv_cup1(dense_feat))
        coarse_high = self.conv_cup2(coarse_features)
        coarse_fps = coarse_high
        idx_fps = furthest_point_sample(coarse_high.transpose(1, 2).contiguous(), point_input.shape[2])
        coarse_fps = gather_points(coarse_high, idx_fps)
        
        return coarse_fps

    
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
        self.encoder = PCN_encoder(output_size=global_feature_size)
        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=size_z * 2)
        
        self.generator = Linear_ResBlock(input_size=size_z, output_size=global_feature_size)
        self.decoder = MSAP_SKN_decoder(num_fps=args.num_fps, num_fine=args.num_points, num_coarse=args.num_coarse,
                                        num_coarse_raw=args.num_coarse_raw, layers=layers, knn_list=knn_list,
                                        pk=args.pk, local_folding=args.local_folding, points_label=args.points_label)
        self.gru = GRU(h_size = 8192, c_size = 2048)
        
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

    def forward(self, x, gt=None, label = None,prefix="train", mean_feature=None, alpha=None):
        B,N,C = x.size()
        gt = gt.transpose(1, 2).contiguous()
        # x.shape = 6,3,8192
        # gt.shape = 6,3,2048
        
        
        if prefix=="train":
            feat_y = self.encoder(gt)
            o_y = self.posterior_infer2(self.posterior_infer1(feat_y))
            p_mu, p_std = torch.split(o_y, self.size_z, dim=1)
            p_std = F.softplus(p_std)
            gt_distribution = torch.distributions.Normal(p_mu, p_std)  #gt的正态分布
            standard_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))  # 标准正态分布
            z = gt_distribution.rsample()  #(2,128)
            feat = feat_y
        else:
            feat_x = self.encoder(gt)
            o_x = self.posterior_infer2(self.posterior_infer1(feat_x))
            q_mu, q_std = torch.split(o_x, self.size_z, dim=1)
            q_std = F.softplus(q_std)
            ori_distribution = torch.distributions.Normal(q_mu, q_std)  #输入不完整的正态分布
            z = ori_distribution.rsample()
            feat = feat_x
        
        c_0 = torch.cat([feat, self.generator(z)], dim = 1) #2,1024   2,3,8192
        h_0 = torch.zeros((x.shape[0],x.shape[2])).cuda(x.device) # 2,8192
        x_0 = x
        h_1, c_1 = self.gru(input = x_0, h = h_0, c = c_0)
        _, top_k = h_1.topk(x_0.shape[2] - 2048, dim = 1)
        x_1 = gather_points(x_0, top_k.int())
        feat_x_1 = self.encoder(x_1)
        o_x_1 = self.posterior_infer2(self.posterior_infer1(feat_x_1))
        p_mu_1, p_std_1 = torch.split(o_x_1, self.size_z, dim=1)
        p_std_1 = F.softplus(p_std_1)
        x_1_distribution = torch.distributions.Normal(p_mu_1, p_std_1)
        z_1 = x_1_distribution.rsample()
        feat_x_1_ = torch.cat([feat_x_1, feat, self.generator(z_1)], dim = 1)
        fine_1 = self.decoder(feat_x_1_, x_1) # 6144
        x_1 = x_1 + fine_1

        h_1_ = h_1
        x_1_ = torch.cat([x_1,torch.zeros(x_1.shape[0], 3, x_0.shape[2] - x_1.shape[2]).cuda(x.device)], dim = 2)
        h_2, c_2 = self.gru(input = x_1_, h = h_1_, c = c_1)
        h_2_ = h_2[:,:h_1.shape[1] - 2048]
        _, top_k = h_2_.topk(x_1.shape[2] - 2048, dim = 1)
        x_2 = gather_points(x_1, top_k.int())
        feat_x_2 = self.encoder(x_2)
        o_x_2 = self.posterior_infer2(self.posterior_infer1(feat_x_2))
        p_mu_2, p_std_2 = torch.split(o_x_2, self.size_z, dim=1)
        p_std_2 = F.softplus(p_std_2)
        x_2_distribution = torch.distributions.Normal(p_mu_2, p_std_2)
        z_2 = x_2_distribution.rsample()
        feat_x_2_ = torch.cat([feat_x_2, feat_x_1, self.generator(z_2)], dim = 1)
        fine_2 = self.decoder(feat_x_2_, x_2) #4096
        x_2 = x_2 + fine_2
        
#         h_2_ = h_2
        x_2_ = torch.cat([x_2,torch.zeros(x_2.shape[0], 3, x_1_.shape[2] - x_2.shape[2]).cuda(x.device)], dim = 2)
        h_3, c_3 = self.gru(input = x_2_, h = h_2, c = c_2)
        h_3 = h_3[:,:h_2.shape[1] - 4096]
        _, top_k = h_3.topk(x_2.shape[2] - 2048, dim = 1)
        x_3 = gather_points(x_2, top_k.int())
        feat_x_3 = self.encoder(x_3)
        o_x_3 = self.posterior_infer2(self.posterior_infer1(feat_x_3))
        p_mu_3, p_std_3 = torch.split(o_x_3, self.size_z, dim=1)
        p_std_3 = F.softplus(p_std_3)
        x_3_distribution = torch.distributions.Normal(p_mu_3, p_std_3)
        z_3 = x_3_distribution.rsample()
        feat_x_3 = torch.cat([feat_x_3, feat_x_2, self.generator(z_3)], dim = 1)
        fine_3 = self.decoder(feat_x_3, x_3) # 2048
        x_3 = x_3 + fine_3
        
        if prefix=="train":
            if self.distribution_loss == 'MMD':
                z_m = m_distribution.rsample()
                z_q = q_distribution.rsample()
                z_p = p_distribution.rsample()
                z_p_fix = p_distribution_fix.rsample()
                dl_rec = self.mmd_loss(z_m, z_p)
                dl_g = self.mmd_loss2(z_q, z_p_fix)
            elif self.distribution_loss == 'KLD':
                dl_rec = torch.distributions.kl_divergence(gt_distribution, standard_distribution)
                dl_view_0 = torch.distributions.kl_divergence(x_1_distribution, gt_distribution)
                dl_view_1 = torch.distributions.kl_divergence(x_2_distribution, gt_distribution)
                dl_view_2 = torch.distributions.kl_divergence(x_3_distribution, gt_distribution)
                
            else:
                raise NotImplementedError('Distribution loss is either MMD or KLD')

            if self.train_loss == 'cd':
                gt = gt.transpose(1, 2).contiguous()
                x_1 = x_1.transpose(1, 2).contiguous()
                x_2 = x_2.transpose(1, 2).contiguous()
                x_3 = x_3.transpose(1, 2).contiguous()
                loss1, _, _ = calc_cd(x_1, gt)
                loss2, _, _ = calc_cd(x_2, gt)
                loss3, _, _ = calc_cd(x_3, gt)
                
#                 print(alpha[0])
#                 print(alpha[1])
#                 print(alpha[2])
#                 print(dsadsa)
                total_train_loss = loss1.mean() * alpha[0] + loss2.mean() * alpha[1] + loss3.mean() * alpha[2] 
                if((torch.isnan(dl_rec.mean()) == False )and (torch.isinf(dl_rec.mean()) == False ) ):
                    total_train_loss += (dl_rec.mean() * 0.5)
                if((torch.isnan(dl_view_0.mean()) == False )and (torch.isinf(dl_view_0.mean()) == False ) ):
                    total_train_loss += (dl_view_0.mean() * 0.5)
                if((torch.isnan(dl_view_1.mean()) == False )and (torch.isinf(dl_view_1.mean()) == False ) ):
                    total_train_loss += (dl_view_1.mean() * 0.5)
                if((torch.isnan(dl_view_2.mean()) == False )and (torch.isinf(dl_view_2.mean()) == False ) ):
                    total_train_loss += (dl_view_2.mean() * 0.5)
                if((torch.isnan(total_train_loss))):
                    print(111,loss1,111)
                    print(222,loss2,222)
                    print(333,loss3,333)
                    print(444,dl_rec,444)
                    print(555,dl_view_0,555)
                    print(666,dl_view_1,666)
                    print(777,dl_view_2,777)
                    np.savetxt('./loss1.txt', loss1.cpu().numpy())
                    np.savetxt('./loss2.txt', loss2.cpu().numpy())
                    np.savetxt('./loss3.txt', loss3.cpu().numpy())
                    np.savetxt('./loss4.txt', dl_rec.cpu().numpy())
                    np.savetxt('./loss5.txt', dl_view_0.cpu().numpy())
                    np.savetxt('./loss6.txt', dl_view_1.cpu().numpy())
                    np.savetxt('./loss7.txt', dl_view_2.cpu().numpy())

                    print(dsadsads)
#                 print(888,total_train_loss,888)
#                 print(9,x_1.shape)
#                 print(10,x_2.shape)
#                 print(11,x_3.shape)
#                 print(12,gt.shape)
#                 print(13,gt_distribution)
#                 print(14,x_1_distribution)
#                 print(15,x_2_distribution)
#                 print(16,x_3_distribution)
#                 print(dsadsa)
                print_loss = [loss1.mean(),loss2.mean(),loss3.mean(),dl_rec.mean(),dl_view_0.mean(),dl_view_1.mean(),dl_view_2.mean()]
#                 print_loss = []
            else:
                raise NotImplementedError('Only CD is supported')
                
            return x_3.permute(0,2,1), loss3, total_train_loss, print_loss
        elif prefix=="val":
            gt = gt.transpose(1, 2).contiguous()
            x_1 = x_1.transpose(1, 2).contiguous()
            x_2 = x_2.transpose(1, 2).contiguous()
            x_3 = x_3.transpose(1, 2).contiguous()
#             print(x_3.shape)
#             print(gt.shape)
#             print(dsadsda)
            if self.eval_emd:
                emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(x_3, gt, calc_f1=True)
            return {'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
        elif prefix=="test":
            return {'result': fine}

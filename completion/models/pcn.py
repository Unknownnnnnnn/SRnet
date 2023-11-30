from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math

# from utils.model_utils import gen_grid_up, calc_emd, calc_cd
from model_utils import gen_grid_up, calc_cd#cal_emd, calc_cd
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
        x_r = (self.act(self.after_norm(self.trans_conv(x - x_r))))
        x = x + x_r
        return x

class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(78, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(128, output_size, 1)
        self.att1 = SA_Layer(64)
        self.att2 = SA_Layer(256)
    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.att1(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.att2(x)
        
        
        global_feature = x.transpose(2,1)
        global_feature = global_feature.reshape((global_feature.shape[0]*global_feature.shape[1],-1))
        return x, global_feature
#         global_feature, _ = torch.max(x, 2)
#         return x, global_feature.view(batch_size, -1)
    
class PCN_encoder_2(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder_2, self).__init__()
        self.conv1 = nn.Conv1d(6, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, output_size, 1)
        self.att1 = SA_Layer(32)
        self.att2 = SA_Layer(64)
    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.att1(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.att2(x)
        global_feature, _ = torch.max(x, 2)
        return x, global_feature.view(batch_size, -1)


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
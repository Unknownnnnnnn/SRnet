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
sys.path.append("./models/")
from sample import Samplenet
from odometry import Odometrynet

class Model(nn.Module):
    def __init__(self, args, type_):
        super(Model, self).__init__()
        self.sample = Samplenet(args, type_)
        self.odometry = Odometrynet(args, type_)

    def forward(self, iter_, inputs, random_inputs, gt, transforms, alpha, prefix="train"):
#         print(inputs.shape)
#         print(random_inputs.shape)
#         print(gt.shape)
#         print(transforms.shape)
#         torch.Size([2, 3, 8192])
#         torch.Size([2, 3, 8192])
#         torch.Size([2, 2048, 3])
#         torch.Size([2, 4, 4, 4])

        if(prefix == 'train'):
            # step 1 假设sample绝对优秀，训练odometry
            if(iter_ == 1):
                ori_inputs = inputs
                total_loss_odometry = 0.
                for j in range(1):
                    now_inputs = inputs[:,:,j*2048 : (j+1)*2048]
                    now_random_inputs = random_inputs[:,:,j*2048 : (j+1)*2048]
                    now_transforms = transforms[:,j,:,:]
                    pred_regis, (r_err, t_err), loss, print_loss = self.odometry(now_inputs, now_random_inputs, gt, now_transforms, prefix = 'train')
                    ori_inputs[:,:,j*2048 : (j+1)*2048] = pred_regis.transpose(1,2).contiguous()
                    total_loss_odometry += loss
                
                
                
                ori_inputs = ori_inputs[:,:,:2048]
                random_inputs = random_inputs[:,:,:2048]
                
                
                
                
                
                (pred_sim_change,pred_center_change,mean_local_change), (f1__change,f1_change,f1___change), total_train_loss_change, print_loss_change = self.sample(ori_inputs, random_inputs, gt, transforms, prefix = 'train', alpha = alpha)
                (pred_sim_ori,pred_center_ori,mean_local_ori), (f1__ori,f1_ori,f1___ori), total_train_loss_ori, print_loss_ori = self.sample(random_inputs, random_inputs, gt, transforms, prefix = 'train', alpha = alpha)
                dist = f1___change - f1___ori
                print_loss = [total_loss_odometry.mean().item(), total_train_loss_ori.mean().item()]
                if(f1___change.mean() > f1___ori.mean()):
                    return 1, (ori_inputs,pred_sim_change,pred_center_change,mean_local_change), (f1__change,f1_change,f1___change), total_train_loss_ori, print_loss, total_loss_odometry
                else:
                    return 0, (ori_inputs,pred_sim_ori,pred_center_ori,mean_local_ori), (f1__ori,f1_ori,f1___ori), total_train_loss_ori, print_loss, total_loss_odometry



            # step 2 假设odometry绝对优秀，训练sample
            if(iter_ == 2):
                ori_inputs = inputs
                total_loss_odometry = 0.
                for j in range(0):
                    now_inputs = gt.transpose(2,1).contiguous()
                    now_random_inputs = inputs[:,:,j*2048 : (j+1)*2048]
                    now_transforms = transforms[:,j,:,:]
                    pred_regis, (r_err, t_err), loss, print_loss = self.odometry(now_inputs, now_random_inputs, gt, now_transforms, prefix = 'train')
                    ori_inputs[:,:,j*2048 : (j+1)*2048] = pred_regis.transpose(1,2).contiguous()
                    total_loss_odometry += loss

                    
                ori_inputs = ori_inputs[:,:,:2048]
                inputs = inputs[:,:,:2048]    
                    
                    
                    
                    
                    
                    
                (pred_sim_change,pred_center_change,mean_local_change), (f1__change,f1_change,f1___change), total_train_loss_change, print_loss_change = self.sample(ori_inputs, random_inputs, gt, transforms, prefix = 'train', alpha = alpha)
                (pred_sim_ori,pred_center_ori,mean_local_ori), (f1__ori,f1_ori,f1___ori), total_train_loss_ori, print_loss_ori = self.sample(inputs, random_inputs, gt, transforms, prefix = 'train', alpha = alpha)
                
                
                
                
                total_loss_odometry = total_train_loss_ori.mean()
                
                
                
                print_loss = [total_loss_odometry.mean().item(), total_train_loss_ori.mean().item()]
                dist = f1___change - f1___ori
                if(f1___change.mean() > f1___ori.mean()):
                    return 1, (ori_inputs,pred_sim_change,pred_center_change,mean_local_change), (f1__change,f1_change,f1___change), total_train_loss_ori, print_loss, total_loss_odometry
                else:
                    return 0, (ori_inputs,pred_sim_ori,pred_center_ori,mean_local_ori), (f1__ori,f1_ori,f1___ori), total_train_loss_ori, print_loss, total_loss_odometry
        else:
            ori_inputs = inputs
            total_loss_odometry = 0.
            for j in range(4):
                now_inputs = gt
                now_random_inputs = inputs[:,:,j*2048 : (j+1)*2048]
                now_transforms = transforms[:,j,:,:]
                pred_regis, (r_err, t_err), loss, print_loss = self.odometry(now_inputs, now_random_inputs, gt, now_transforms, prefix = 'train')
                ori_inputs[:,:,j*2048 : (j+1)*2048] = pred_regis
                total_loss_odometry += loss

            (pred_sim_change,pred_center_change,mean_local_change), (f1__change,f1_change,f1___change), total_train_loss_change, print_loss_change = self.sample(ori_inputs, random_inputs, gt, transforms, prefix = 'train')
            
            return {'odometry':total_loss_odometry, 'sample':total_train_loss_change, 'result':mean_local_change, 'f1':f1___change}
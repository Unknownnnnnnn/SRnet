import argparse
import numpy as np
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
# from time import time
# from torch.utils.data import DataLoader, Subset

import torch.optim as optim
from tqdm import tqdm
import os
import random
import sys

import logging
import math
import importlib
import datetime
import munch
import yaml
import argparse
import copy
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from train_utils import AverageValueMeter, save_model
from dataset import MVP_RG
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/new')




def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)
def plot_single_pcd(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

def train():
    flag__ = 0
    logging.info(str(args))
    metrics = ['RotE', 'transE', 'MSE', 'RMSE', 'recall']
    best_epoch_losses = {m: (0, 0) if m =='recall' else (0, math.inf) for m in metrics}
    # best_epoch_losses = (0, inf)
    # train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    val_split_loss_meters = []
    for i in range(args.num_rot_levels):
        row = []
        for j in range(args.num_corr_levels):
            row.append(copy.deepcopy(val_loss_meters))
        val_split_loss_meters.append(row)
    val_split_loss_meters = np.array(val_split_loss_meters)

    dataset = MVP_RG(prefix="train", args=args)
    dataset_test = MVP_RG(prefix="val", args=args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))
    
    varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
    varying_constant_1 = [float(c.strip()) for c in args.varying_constant_1.split(',')]
    varying_constant_2 = [float(c.strip()) for c in args.varying_constant_2.split(',')]
    varying_constant_3 = [float(c.strip()) for c in args.varying_constant_3.split(',')]
    
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args,'first'))
    net.cuda()

    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    lr = args.lr
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    elif args.optimizer == 'Adam':
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    else:
        optimizer = optimizer(net.module.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay_rate, min_lr=args.lr_clip)
    
    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        # train_loss_meter.reset()
        net.module.train()
        
        alpha = []
        for ind, ep in enumerate(varying_constant_epochs):
            if epoch < ep:
                alpha.append(varying_constant_1[ind])
                alpha.append(varying_constant_2[ind])
                alpha.append(varying_constant_3[ind])
                break
            elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                alpha.append(varying_constant_1[ind+1])
                alpha.append(varying_constant_2[ind+1])
                alpha.append(varying_constant_3[ind])
                break
        for i, data in enumerate(dataloader, 0):
#             break

            src, tgt, T_gt, pose_src, pose_tgt, complete_src, complete_tgt= data

            src = src.float().cuda()
            tgt = tgt.float().cuda()
            complete_src = complete_src.float().cuda()
            complete_tgt = complete_tgt.float().cuda()
            pose_src = pose_src.float().cuda()
            pose_tgt = pose_tgt.float().cuda()
            T_gt = T_gt.float().cuda()
#             break
            net_loss, r_err, t_err, rmse, mse = net(src, tgt, complete_src, complete_tgt, T_gt, pose_src, pose_tgt, epoch = epoch, gamma=alpha)
            
            optimizer.zero_grad()
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
#             clip_gradient(optimizer, opt.clip)
#             if (epoch <= 5):
            torch.nn.utils.clip_grad_norm_(parameters = net.parameters(), max_norm = 10, norm_type=2)
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d] total_loss: %.4f rot_loss: %.4f trans_loss: %.4f rmse_loss: %.4f mse_loss: %.4f lr: %f' %
                    (epoch, i, len(dataset) / args.batch_size, net_loss.detach().mean().item(), r_err.detach().mean().item(), t_err.detach().mean().item(), rmse.detach().mean().item(), mse.detach().mean().item(), lr) + ' alpha: ' + str(alpha))

                flag__ += 1
            
#             break
        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")
#         if (epoch >= 5):
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, val_split_loss_meters, dataloader_test, best_epoch_losses)
        writer.close()

def val(net, curr_epoch_num, val_loss_meters, val_split_loss_meters, dataloader_test, best_epoch_losses, rmse_thresh=0.1):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()

    for i in range(val_split_loss_meters.shape[0]):
        for j in range(val_split_loss_meters.shape[1]):
            for v in val_split_loss_meters[i][j].values():
                v.reset()
    
    # num_samples = np.ones(2, 2) * 1.e-6

    net.module.eval()

    with torch.no_grad():
        for _, data in enumerate(dataloader_test):
            src, tgt, T_gt, pose_src, pose_tgt, complete_src, complete_tgt, match_level, rot_level = data
            curr_batch_size = T_gt.shape[0]
            
            src = src.float().cuda()
            tgt = tgt.float().cuda()
            complete_src = complete_src.float().cuda()
            complete_tgt = complete_tgt.float().cuda()
            T_gt = T_gt.float().cuda()
            pose_src = pose_src.float().cuda()
            pose_tgt = pose_tgt.float().cuda()
            match_level = match_level.int().cuda()
            rot_level = rot_level.int().cuda()
            
            
            r_err, t_err, rmse, mse, res = net(src, tgt, complete_src, complete_tgt, T_gt = T_gt, pose_src = pose_src, pose_tgt = pose_tgt, prefix="val", epoch = curr_epoch_num)

            val_loss_meters['RotE'].update(r_err.detach().mean().item(), curr_batch_size)
            val_loss_meters['transE'].update(t_err.detach().mean().item(), curr_batch_size)
            val_loss_meters['MSE'].update(mse.detach().mean().item(), curr_batch_size)
            val_loss_meters['RMSE'].update(rmse.detach().mean().item(), curr_batch_size)
            val_loss_meters['recall'].update((rmse.detach() < rmse_thresh).to(torch.float32).mean().item(), curr_batch_size)
            
            for i in range(curr_batch_size):
                val_split_loss_meters[rot_level[i]][match_level[i]]['RotE'].update(r_err[i].detach().item())
                val_split_loss_meters[rot_level[i]][match_level[i]]['transE'].update(t_err[i].detach().item())
                val_split_loss_meters[rot_level[i]][match_level[i]]['MSE'].update(mse[i].detach().item())
                val_split_loss_meters[rot_level[i]][match_level[i]]['RMSE'].update(rmse[i].detach().item())
                val_split_loss_meters[rot_level[i]][match_level[i]]['recall'].update((rmse[i].detach() < rmse_thresh).item())
            
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'recall') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'recall'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)

        for i in range(val_split_loss_meters.shape[0]):
            for j in range(val_split_loss_meters.shape[1]):
                curr_val_level = val_split_loss_meters[i][j]
                
                curr_split_log = '[rot_level %d, match_level %d] ' % (i, j)
                for loss_type, meter in curr_val_level.items():
                    curr_split_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

                logging.info(curr_split_log)

def file_work(LOG_DIR):
    import glob
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    # backup some files
    py_file = glob.glob('./models/*.py')
    log_code = os.path.join(LOG_DIR, 'code')
    os.makedirs(log_code)
    for f in py_file:
        os.system('cp %s %s' % (f, log_code))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
        
    else:
        exp_name = args.model_name + '_' + args.benchmark + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_work(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    train()
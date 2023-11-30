import torch.optim as optim
import torch
# from utils.train_utils import *
from train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import MVP_CP
from vis_utils import plot_single_pcd
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.tensorboard import SummaryWriter
from model_utils import *
sys.path.append("../utils")
writer = SummaryWriter('runs/ex1')
def val_(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    print(val_loss_meters)
    for v in val_loss_meters.values():
        v.reset()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            label, inputs, gt = data
            # mean_feature = None
            curr_batch_size = gt.shape[0]

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            label = label.cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dictt = net(inputs, gt, prefix="val")
            for k, v in val_loss_meters.items():
                v.update(result_dictt[k].mean().item(), curr_batch_size)

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
#                 save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
def train():
    logging.info(str(args))
    if args.eval_emd:
        metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    else:
        metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataset = MVP_CP(prefix="train")
    dataset_test = MVP_CP(prefix="val")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))


    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args,'second'))
    net = net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    cascade_gan = (args.model_name == 'cascade')
    net_d = None
    if cascade_gan:
        net_d = torch.nn.DataParallel(model_module.Discriminator(args))
        net_d.cuda()
        net_d.module.apply(model_module.weights_init)

    lr = args.lr
    if cascade_gan:
        lr_d = lr / 2
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if cascade_gan:
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, weight_decay=0.00001, betas=(0.5, 0.999))

    varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
    varying_constant_1 = [float(c.strip()) for c in args.varying_constant_1.split(',')]
    varying_constant_2 = [float(c.strip()) for c in args.varying_constant_2.split(',')]
    varying_constant_3 = [float(c.strip()) for c in args.varying_constant_3.split(',')]

    if args.load_model:
        
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        if cascade_gan:
            net_d.module.load_state_dict(ckpt['D_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
    
    for epoch in range(args.start_epoch, args.nepoch):
        train_loss_meter.reset()
        net.train()
        if cascade_gan:
            net_d.module.train()

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

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
#             break
            optimizer.zero_grad()
            if cascade_gan:
                optimizer_d.zero_grad()

            label, inputs, random_inputs, transforms, gt = data

            inputs = inputs.float().cuda().transpose(2, 1).contiguous()
            gt = gt.float().cuda()
            label = label.cuda()
            random_inputs = random_inputs.float().cuda().transpose(2, 1).contiguous()
            transforms = transforms.float().cuda().contiguous()
            
            out2, loss2, net_loss, loss_tuple= net(inputs, random_inputs, gt, transforms, alpha=alpha)
            if cascade_gan:
                d_fake = generator_step(net_d, out2, net_loss, optimizer)
                discriminator_step(net_d, gt, d_fake, optimizer_d)
            else:
                train_loss_meter.update(net_loss.mean().item())
                net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
                optimizer.step()
#             break
            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f %f %f %f %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss2[0].mean().item(), loss2[1].mean().item(), loss2[2].mean().item(),loss2[3].mean().item(), loss2[4].mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))
                writer.add_scalar('loss(1)',loss_tuple[0],i)
                writer.close()
#             break
        for ind in range(2):
            plot_single_pcd(inputs[ind].permute(1,0).detach().cpu().numpy().squeeze(),'./res/ori_'+'step_'+str(ind)+'.png')
            plot_single_pcd(random_inputs[ind].permute(1,0).detach().cpu().numpy().squeeze(),'./res/ori__'+'step_'+str(ind)+'.png')
            plot_single_pcd(gt[ind].cpu().numpy().squeeze(),'./res/tar_'+'step_'+str(ind)+'.png')
            plot_single_pcd(out2[0][ind].detach().cpu().numpy().squeeze(),'./res/pred_'+'step_'+str(ind)+'.png') 
            plot_single_pcd(out2[1][ind].detach().cpu().numpy().squeeze(),'./res/pred__'+'step_'+str(ind)+'.png') 
        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net, net_d=net_d)
            logging.info("Saving net...")
        
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)
#     writer.close()
        
def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    print(val_loss_meters)
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            label, inputs, random_inputs, transforms, gt = data
            # mean_feature = None
            curr_batch_size = gt.shape[0]

            inputs = inputs.float().cuda().transpose(2, 1).contiguous()
            gt = gt.float().cuda()
            label = label.cuda()
            random_inputs = random_inputs.float().cuda().transpose(2, 1).contiguous()
            transforms = transforms.float().cuda().contiguous()
            
            result_dictt = net(inputs, random_inputs, gt, transforms, prefix="val")
            for k, v in val_loss_meters.items():
                v.update(result_dictt[k].mean().item(), curr_batch_size)
#             break
        for ind in range(2):
            plot_single_pcd(inputs[ind].permute(1,0).detach().cpu().numpy().squeeze(),'./res/ori_'+'step_'+str(ind)+'.png')
            plot_single_pcd(random_inputs[ind].permute(1,0).detach().cpu().numpy().squeeze(),'./res/ori__'+'step_'+str(ind)+'.png')
            plot_single_pcd(gt[ind].cpu().numpy().squeeze(),'./res/tar_'+'step_'+str(ind)+'.png')
            plot_single_pcd(result_dictt['result'][ind].detach().cpu().numpy().squeeze(),'./res/pred_'+'step_'+str(ind)+'.png') 
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
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
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    train()
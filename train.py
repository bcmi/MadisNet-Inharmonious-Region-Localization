import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
import numpy as np
import glob
import os
import itertools
import cv2
import multiprocessing as mp


from networks import DIRLNet, UNet,  HDRPointwiseNN, DomainEncoder
from evaluation.metrics import FScore, normPRED, compute_mAP, compute_IoU, AverageMeter
# import matplotlib.pyplot as plt


from dataset.ihd_dataset import IhdDataset
from dataset.multi_objects_ihd_dataset import MultiObjectsIhdDataset
from options import ArgsParser
import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target, loss_weights=[1.0,1.0,1.0]):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out*loss_weights[0] + ssim_out*loss_weights[1] + iou_out*loss_weights[2]
    return {"total":loss, "bce":bce_out, "ssim":ssim_out, "iou":iou_out}
    
    
def multi_bce_loss_fusion(preds, labels_v, side_weights=1, loss_weights=[1.0,1.0,1.0]):
    total_loss = 0
    bce_out = 0
    ssim_out = 0
    iou_out = 0
    if isinstance(side_weights, int):
        side_weights = [side_weights] * len(preds)

    for pred,w in zip(preds,side_weights):
        loss = bce_ssim_loss(pred, labels_v, loss_weights)
        total_loss += loss['total'] * w
        bce_out += loss['bce']
        ssim_out += loss['ssim']
        iou_out += loss['iou']
    
    return {"total":total_loss, "bce":bce_out, "ssim":ssim_out, "iou":iou_out}





class Trainer(object):
    def __init__(self, opt):
        self.opt = opt

        # Set Loggers
        if opt.is_train:
            log_dir = os.path.join(opt.checkpoints_dir, "logs")
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)   # create a visualizer that display/save images and plots
        # Set Device
        self.gpus = opt.gpu_ids.split(',')
        self.gpus = [int(id) for id in self.gpus]
        self.device = torch.device('cuda:{}'.format(self.gpus[0])) if self.gpus[0]>-1 else torch.device('cpu')  # get device name: CPU or GPU
        print(self.device)
        self.opt.device = self.device
        self.opt.gpus = self.gpus

        self.best_acc = 0
        # ------- 3. define model --------
        self.domain_encoder = DomainEncoder(style_dim=16)
        self.ihdrnet = HDRPointwiseNN(opt)
        if self.opt.model == 'dirl':
            print("DIRL is used for MadisNet !")
            self.g = DIRLNet(opt,3)
        elif self.opt.model == 'unet':
            print("UNet is used for MadisNet !")
            self.g = UNet(in_ch=3, n_downs=5)
        else:
            raise ValueError("Unknown model:\t{}".format(self.opt.model))
        
        
        g_size = sum(p.numel() for p in self.g.parameters())/1e6
        ihdrnet_size = sum(p.numel() for p in self.ihdrnet.parameters())/1e6
        e_dom_size = sum(p.numel() for p in self.domain_encoder.parameters())/1e6
        print('--- G params: %.2fM' % (g_size))
        print('--- iHDRNet params: %.2fM' % (ihdrnet_size))
        print('--- E_dom params: %.2fM' % (e_dom_size))
        print('--- Total params: %.2fM' % (g_size + ihdrnet_size + e_dom_size))


        if len(self.gpus) > 1:
            self.dataparallel_func = nn.DataParallel
        else:
            self.dataparallel_func = None
       
        if opt.is_train == 1:
            if self.dataparallel_func is not None:
                self.domain_encoder = self.dataparallel_func(self.domain_encoder.to(self.device), self.gpus)
                self.g = self.dataparallel_func(self.g.to(self.device), self.gpus)
                self.ihdrnet = self.dataparallel_func(self.ihdrnet.to(self.device), self.gpus)
            else:
                self.domain_encoder.to(self.device)
                self.g.to(self.device)
                self.ihdrnet.to(self.device)
        # Test
        else:
            self.domain_encoder.to(self.device)
            self.g.to(self.device)
            self.ihdrnet.to(self.device)
            self.domain_encoder.eval()
            self.g.eval()
            self.ihdrnet.eval()

            

        # ------- 2. set the directory of training dataset --------
        self.data_mean = opt.mean.split(",")
        self.data_mean = [float(m.strip()) for m in self.data_mean]
        self.data_std = opt.std.split(",")
        self.data_std = [float(m.strip()) for m in self.data_std]

        dataset_loader = IhdDataset
        inharm_dataset = dataset_loader(opt)
        if opt.is_train == 0:
            opt.batch_size = 1
            opt.num_threads = 1
            opt.serial_batches = True

        # Training Set
        self.inharm_dataloader = torch.utils.data.DataLoader(
                    inharm_dataset,
                    batch_size=opt.batch_size,
                    shuffle=not opt.serial_batches,
                    num_workers=int(opt.num_threads),
                    drop_last=True)
        # Validation Set
        opt.is_train = 0
        opt.is_val = 1
        opt.preprocess = 'resize'
        opt.no_flip = True
        self.val_dataloader = torch.utils.data.DataLoader(
                    dataset_loader(opt),
                    batch_size=1,
                    shuffle=False,
                    num_workers=1)
        # Reset training state
        opt.is_train = True

        # ------- 4. define optimizer --------
        if opt.is_train :
            print("---define optimizer...")
            self.image_display = None
            self.domain_encoder_opt = optim.Adam(self.domain_encoder.parameters(), lr=opt.lr, betas=(0.9,0.999), weight_decay=opt.weight_decay)
            self.g_opt = optim.Adam(self.g.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
            self.ihdrnet_opt = optim.Adam(self.ihdrnet.parameters(), lr=opt.lr, betas=(0.9,0.999), weight_decay=opt.weight_decay)

            self.domain_encoder_schedular   = optim.lr_scheduler.MultiStepLR(self.domain_encoder_opt, milestones=[30, 40, 50, 55], gamma=0.5)
            self.g_schedular  = optim.lr_scheduler.MultiStepLR(self.g_opt, milestones=[30, 40, 50, 55], gamma=0.5)
            self.ihdrnet_schedular  = optim.lr_scheduler.MultiStepLR(self.ihdrnet_opt, milestones=[30, 40, 50, 55], gamma=0.5)

    def adjust_learning_rate(self):
        self.domain_encoder_schedular.step()
        self.g_schedular.step()
        self.ihdrnet_schedular.step()

    def write_display(self, total_it, model, batch_size):
        # write loss
        members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and attr.startswith('loss')]
        for m in members:
            self.writer.add_scalar(m, getattr(model, m), total_it)
        # write img
        if isinstance(model.image_display, torch.Tensor):
            image_dis = torchvision.utils.make_grid(model.image_display, nrow=batch_size)
            mean = torch.zeros_like(image_dis)
            mean[0,:,:] = .485
            mean[1,:,:] = .456
            mean[2,:,:] = .406
            std = torch.zeros_like(image_dis)
            std[0,:,:] = 0.229
            std[1,:,:] = 0.224
            std[2,:,:] = 0.225
            image_dis = image_dis*std + mean
            self.writer.add_image('Image', image_dis, total_it)

    def load_dict(self, net, name, resume_epoch, strict=True, checkpoints_dir=''):
        if checkpoints_dir == '':
            checkpoints_dir = self.opt.checkpoints_dir
        ckpt_name = "{}_epoch{}.pth".format(name, resume_epoch)
        if not os.path.exists(os.path.join(checkpoints_dir, ckpt_name)): 
            ckpt_name = "{}_epoch{}.pth".format(name, "best")
            if not os.path.exists(os.path.join(checkpoints_dir, ckpt_name)): 
                ckpt_name = "{}_epoch{}.pth".format(name, "latest")
        print("Loading model weights from {}...".format(ckpt_name))

        # restore lr
        sch = getattr(self, '{}_schedular'.format(name))
        sch.last_epoch = resume_epoch if resume_epoch > 0 else 0
        decay_coef = 0
        for ms in sch.milestones.keys():
            if sch.last_epoch <= ms: decay_coef+=1

        for group in sch.optimizer.param_groups:
            group['lr'] = group['lr'] * sch.gamma ** decay_coef
        
        ckpt_dict = torch.load(os.path.join(checkpoints_dir,ckpt_name), map_location=self.device)
        if 'best_acc' in ckpt_dict.keys():
            new_state_dict = ckpt_dict['state_dict']
            save_epoch = ckpt_dict['epoch']
            self.best_acc  = ckpt_dict['best_acc']
            print("The model from epoch {} reaches acc at {:.4f} !".format(save_epoch, self.best_acc))
        else:
            new_state_dict = ckpt_dict
            
        current_state_dict = net.state_dict()
        new_keys = tuple(new_state_dict.keys())
        for k in new_keys:
            if k.startswith('module'):
                v = new_state_dict.pop(k)
                nk = k.split('module.')[-1]
                new_state_dict[nk] = v
        if len(self.gpus) > 1:
            net.module.load_state_dict(new_state_dict, strict=strict)
        else:
            net.load_state_dict(new_state_dict, strict=True) # strict


    def resume(self, resume_epoch, strict=True, is_pretrain=False, preference=[], checkpoints_dir=''):
        if preference != []:
            for net_name in preference:
                net = getattr(self, net_name)
                self.load_dict(net, net_name, resume_epoch, strict=strict, checkpoints_dir=checkpoints_dir)
            return 

    def save(self, epoch, is_pretrain=False, preference=[]):
        if preference != []:
            for net_name in preference:
                model_name = "{}_epoch{}.pth".format(net_name, epoch)
                net = getattr(self, net_name)
                save_dict = {
                    'epoch':epoch,
                    'best_acc':self.best_acc,
                    'state_dict':net.state_dict(),
                    'opt':getattr(self, '{}_schedular'.format(net_name)).state_dict()
                }
                torch.save(save_dict, os.path.join(self.opt.checkpoints_dir, model_name))
            return 

    def denormalize(self, x, isMask=False):
        if isMask:
            mean = 0
            std=1
        else:
            mean = torch.zeros_like(x)
            mean[:,0,:,:] = .485
            mean[:,1,:,:] = .456
            mean[:,2,:,:] = .406
            std = torch.zeros_like(x)
            std[:,0,:,:] = 0.229
            std[:,1,:,:] = 0.224
            std[:,2,:,:] = 0.225
        x = (x*std + mean)*255
        x = x.cpu().detach().numpy().transpose(0,2,3,1).astype(np.uint8)
        if isMask:
            if x.shape[3] == 1:
                x = x.repeat(3, axis=3)
        return x

    def norm(self, x):
        mean = torch.zeros_like(x)
        mean[:,0,:,:] = .485
        mean[:,1,:,:] = .456
        mean[:,2,:,:] = .406
        std = torch.zeros_like(x)
        std[:,0,:,:] = 0.229
        std[:,1,:,:] = 0.224
        std[:,2,:,:] = 0.225
        x = (x - mean) / std #*255
        return x

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def forward(self, img, mask=None):
        retouched_img, guide_map = self.ihdrnet(img, img)
        delta_img = retouched_img
        mask_main = self.g(delta_img)['mask']
        # domain codes
        if mask is not None:
            z_b = self.domain_encoder(img, 1-mask)
            z_f = self.domain_encoder(img, mask)
            z_mb = self.domain_encoder(retouched_img, 1-mask)
            z_mf = self.domain_encoder(retouched_img, mask)
            return mask_main, retouched_img,guide_map, z_b,z_f,z_mb,z_mf
        else:
            return mask_main, retouched_img, guide_map

    def val(self, epoch=0, is_test=False):
        print("---start validation---")
        total_iters = 0
        mAPMeter = AverageMeter()
        F1Meter = AverageMeter()
        FbMeter = AverageMeter()
        IoUMeter = AverageMeter()
        
        self.g.eval()
        self.ihdrnet.eval()
        self.domain_encoder.eval()
        
        for i_test, data in enumerate(self.val_dataloader):
            inharmonious, mask_gt = data['comp'], data['mask']

            inharmonious = inharmonious.type(torch.FloatTensor).to(self.device)
            mask_gt = mask_gt.type(torch.FloatTensor).to(self.device)            
            with torch.no_grad():
                masks, _, guide_map = self.forward(inharmonious)
                inharmonious_pred = masks[0]
                inharmonious_pred = normPRED(inharmonious_pred)
                mask_gt = normPRED(mask_gt)

                pred = inharmonious_pred
                label = mask_gt

                F1 = FScore(pred, label)
                
                mAP = compute_mAP(pred, label)

                IoUMeter.update(compute_IoU(pred, label), label.size(0))
                mAPMeter.update(mAP, inharmonious_pred.size(0))
                F1Meter.update(F1, inharmonious_pred.size(0))

                total_iters += 1
                if total_iters % 100 == 0:
                    print("Batch: [{}/{}],\tmAP:\t{:.4f}\tF1:\t{:.4f}\t\tIoU:\t{:.4f}".format((i_test+1) , len(self.val_dataloader), \
                        mAPMeter.avg, F1Meter.avg, IoUMeter.avg))

        if is_test:
            name = self.opt.checkpoints_dir.split('/')[-1]
            print("Model\t{}:\nmAP:\t{:.4f}\nF1:\t{:.4f}\nIoU:\t{:.4f}".format(name,\
                        mAPMeter.avg, F1Meter.avg, IoUMeter.avg))
        else:
            val_mIoU = IoUMeter.avg
            if self.best_acc < val_mIoU:
                self.best_acc = val_mIoU
                self.save("best", preference=['g','ihdrnet','domain_encoder'])
                print("New Best score!\nmAP:\t{:.4f},\tF1:\t{:.4f},\tIoU:\t{:.4f}".format(mAPMeter.avg, F1Meter.avg, val_mIoU))
        
        self.g.train()
        self.ihdrnet.train()
        self.domain_encoder.train()

    

    def train_epoch(self, epoch, total_epoch=100):
        # ------- 5. training process --------
        total_iters = epoch * len(self.inharm_dataloader)
        running_loss = 0.0
        running_tar_loss = 0.0

        # Set meters
        loss_total_meter = AverageMeter()
        loss_det_meter = AverageMeter()
        loss_reg_meter = AverageMeter()
        loss_triplet_meter = AverageMeter()

        F1Meter = AverageMeter()
        
        self.ihdrnet.train()
        self.g.train()
        self.domain_encoder.train()

        for i, data in enumerate(self.inharm_dataloader):
            total_iters = total_iters + 1

            inharmonious, mask_gt = data['comp'], data['mask']
            inharmonious = inharmonious.type(torch.FloatTensor).to(self.device)
            mask_gt = mask_gt.type(torch.FloatTensor).to(self.device)

            # update the main generator and lut branch
            self.ihdrnet_opt.zero_grad()
            self.g_opt.zero_grad()
            self.domain_encoder_opt.zero_grad()
            masks, retouched_img, guide_map, z_b, z_f, z_mb,z_mf = self.forward(inharmonious, mask_gt)
            inharmonious_pred = masks
            
            if self.opt.model == 'dirl':
                loss_inharmonious = multi_bce_loss_fusion([inharmonious_pred[0]], mask_gt, loss_weights=[1,self.opt.lambda_ssim, self.opt.lambda_iou])
                self.loss_attention = multi_bce_loss_fusion(inharmonious_pred[1:], mask_gt, loss_weights=[1,self.opt.lambda_ssim, self.opt.lambda_iou])['total']
            else:
                loss_inharmonious = multi_bce_loss_fusion([inharmonious_pred[0]], mask_gt, loss_weights=[1,self.opt.lambda_ssim, self.opt.lambda_iou])

            
            self.loss_detection_ssim = loss_inharmonious['ssim']
            self.loss_detection_bce = loss_inharmonious['bce']
            self.loss_detection =  loss_inharmonious['total'] 
            
            self.loss_total = self.loss_detection * self.opt.lambda_detection
            if self.opt.model == 'dirl':
                self.loss_total = self.loss_total + self.loss_attention * self.opt.lambda_attention

            ## triplet loss
            eps = 1e-6
            z_fb = z_f - z_b
            z_mfmb = z_mf - z_mb
            input_distance = (z_fb**2).sum(dim=1,keepdim=True)
            magnify_distance = (z_mfmb**2).sum(dim=1,keepdim=True)
            
            dir_cos = (z_fb*z_mfmb).sum(dim=1,keepdim=True) / (torch.norm(z_fb, dim=1, keepdim=True)*torch.norm(z_mfmb, dim=1, keepdim=True)+eps)
            loss_reg = (1-dir_cos).mean()
            
            loss_ddm = nn.ReLU()(input_distance-magnify_distance+self.opt.m).mean()
            self.loss_triplet =  loss_ddm * self.opt.lambda_tri + loss_reg * self.opt.lambda_reg

            self.loss_total = self.loss_total + self.loss_triplet 
            self.loss_total.backward()
            
            self.g_opt.step()
            self.domain_encoder_opt.step()
            self.ihdrnet_opt.step()

            loss_total_meter.update(self.loss_total.item(), n=inharmonious.shape[0])
            loss_det_meter.update(self.loss_detection.item(), n=inharmonious.shape[0])
            loss_triplet_meter.update(self.loss_triplet.item(), n=inharmonious.shape[0])
            F1Meter.update(FScore(inharmonious_pred[0], mask_gt), n=inharmonious.shape[0])

            if total_iters % self.opt.print_freq == 0:              
                print("Epoch: [%d/%d], Batch: [%d/%d], train loss: %.3f, det loss: %.3f, tri loss: %.3f,  F1 score: %.4f" % (
                epoch + 1, self.opt.nepochs, (i + 1) , len(self.inharm_dataloader),
                loss_total_meter.avg,
                loss_det_meter.avg,
                loss_triplet_meter.avg,
                F1Meter.avg
                ))
            if total_iters %  self.opt.display_freq== 0: #
                show_size = 5 if inharmonious.shape[0] > 5 else inharmonious.shape[0]
                self.image_display = torch.cat([
                        inharmonious[0:show_size].detach().cpu(),             # input image
                        mask_gt[0:show_size].detach().cpu().repeat(1,3,1,1),                        # ground truth
                        retouched_img[0:show_size].detach().cpu(),
                        inharmonious_pred[0][0:show_size].detach().cpu().repeat(1,3,1,1),
                ],dim=0)
                self.write_display(total_iters, self, show_size)
            # del temporary outputs and loss
            del inharmonious_pred

    def train(self, start_epoch=0):
        # ------- 5. training process --------
        print("---start training...")
        for epoch in range(start_epoch, self.opt.nepochs):
            self.train_epoch(epoch, total_epoch=self.opt.nepochs)
            if (epoch+1) % self.opt.save_epoch_freq == 0:
                self.save("{}".format(epoch), preference=['ihdrnet', 'g','domain_encoder'])
            self.adjust_learning_rate()
            if (epoch+1) < 30:
                if (epoch+1) % self.opt.save_epoch_freq == 0:
                    self.val(epoch)
            else:
                if (epoch+1) % 3 == 0:
                    self.val(epoch)
            
        print('-------------Congratulations, No Errors!!!-------------')

if __name__ == '__main__':
    opt = ArgsParser()
    opt.seed = 42
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
    print(opt.checkpoints_dir.split('/')[-1])
    
    trainer = Trainer(opt)
    
    start_epoch = 0
    if opt.resume > -1:
        trainer.resume(opt.resume, preference=['ihdrnet','g', 'domain_encoder'], checkpoints_dir=opt.pretrain_path)
        start_epoch = opt.resume
    trainer.train(start_epoch=start_epoch)
    

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from dataset import ihd_dataset
from train import Trainer
from options import ArgsParser
from dataset.ihd_dataset import IhdDataset
from dataset.multi_objects_ihd_dataset import MultiObjectsIhdDataset

from evaluation.metrics import MAE, FScore, compute_IoU, normPRED, compute_mAP, AverageMeter
import warnings
warnings.filterwarnings("ignore")





def tensor2np(x, isMask=False):
	if isMask:
		if x.shape[1] == 1:
			x = x.repeat(1,3,1,1)
		x = ((x.cpu().detach()))*255
	else:
		x = x.cpu().detach()
		mean = torch.zeros_like(x)
		std = torch.zeros_like(x)
		mean[:,0,:,:] = 0.485
		mean[:,1,:,:] = 0.456
		mean[:,2,:,:] = 0.406
		std[:,0,:,:]  = 0.229
		std[:,1,:,:]  = 0.224
		std[:,2,:,:]  = 0.225
		x = (x * std + mean)*255
		
	return x.numpy().transpose(0,2,3,1).astype(np.uint8)

def save_output(preds, gts, save_dir, img_fn, extra_infos=None,  verbose=False, alpha=0.5):
	outs = []
	input = gts['inharmonious']
	mask_label = gts['mask_gt']
	mask_pred = preds['mask_main']
	retouched_img = preds['retouched_img']
	guide_map = preds['guide_map']

	input = cv2.cvtColor(tensor2np(input)[0], cv2.COLOR_RGB2BGR)
	retouched_img = cv2.cvtColor(tensor2np(retouched_img, isMask=True)[0], cv2.COLOR_RGB2BGR)
	guide_map = ((((guide_map * 0.5) + 0.5)*255).cpu().detach().repeat(1,3,1,1).numpy().transpose(0,2,3,1).astype(np.uint8))[0]
	mask_label = tensor2np(mask_label, isMask=True)[0]
	outs += [input, mask_label]
	outs += [retouched_img]
	outs += [tensor2np(mask_main[0], isMask=True)[0]]
	outimg = np.concatenate(outs, axis=1) 
	if verbose==True:
		print("show")
		cv2.imshow("out",outimg)
		cv2.waitKey(0)
	else:
		sub_key = os.path.split(img_fn)[1][0]
		if sub_key == 'a': sub_dir = 'adobe'
		if sub_key == 'f': sub_dir = 'flickr'
		if sub_key == 'd': sub_dir = 'day2night'
		if sub_key == 'c': sub_dir = 'coco'
		save_dir = os.path.join(save_dir, sub_dir)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		img_fn = os.path.split(img_fn)[1]
		prefix,suffix = os.path.splitext(img_fn)
		# cv2.imwrite(os.path.join(save_dir, "{}_f1{:.4f}_iou{:.4f}{}".format(prefix, extra_infos['f1'], extra_infos['iou'], suffix)), outimg)
		cv2.imwrite(os.path.join(save_dir, "{}".format(img_fn)), outimg)

# --------- 2. dataloader ---------
#1. dataload
opt = ArgsParser()
opt.phase = 'test'
test_inharm_dataset = IhdDataset(opt)
# test_inharm_dataset = MultiObjectsIhdDataset(opt)
test_inharm_dataloader = DataLoader(test_inharm_dataset, batch_size=1,shuffle=False,num_workers=1)

# --------- 3. model define ---------


print("...load MadisNet...")
checkpoints_dir = opt.checkpoints_dir
prediction_dir = os.path.join(opt.checkpoints_dir, "rst")
if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)

opt.is_train = 0
trainer = Trainer(opt)
trainer.resume(opt.resume, preference=['ihdrnet', 'g', 'domain_encoder'])
device = trainer.device


# ------------ Global Evaluation Metrics -------------
total_iters = 0
gmAP_meter = AverageMeter()
gF1_meter = AverageMeter()
gIoU_meter = AverageMeter()


# ------------- Sub-dataset Metrics ----------
sub_dataset = ['HAdobe', 'HCOCO', 'HDay2Night', 'HFlickr']
lmAP_meters = {k:AverageMeter() for k in sub_dataset}
lF1_meters = {k:AverageMeter() for k in sub_dataset}
lIoU_meters = {k:AverageMeter() for k in sub_dataset}


save_flag = False
trainer.g.eval()
trainer.ihdrnet.eval()
trainer.domain_encoder.eval()

# --------- 4. inference for each image ---------
for i_test, data in enumerate(test_inharm_dataloader):
	inharmonious, mask_gt = data['comp'],  data['mask']

	inharmonious = inharmonious.type(torch.FloatTensor).to(device)
	mask_gt = mask_gt.type(torch.FloatTensor).to(device)

	with torch.no_grad():
		rsts = {}
		model = trainer
		mask_main, retouched_img, guide_map = model.forward(inharmonious)
		inharmonious_pred = mask_main[0]

		inharmonious_pred = normPRED(inharmonious_pred)
		mask_gt = normPRED(mask_gt)

		pred = inharmonious_pred
		label = mask_gt

		F1 = FScore(pred, label)
		mAP = compute_mAP(pred, label)
		IoU = compute_IoU(pred, label)
		
		gF1_meter.update(F1, n=1)
		gmAP_meter.update(mAP, n=1)
		gIoU_meter.update(IoU, n=1)

		# sub dataset
		sub_key = os.path.split(data['img_path'][0])[1][0]
		if sub_key == 'a': key = 'HAdobe'
		if sub_key == 'f': key = 'HFlickr'
		if sub_key == 'd': key = 'HDay2Night'
		if sub_key == 'c': key = 'HCOCO'
		lmAP_meters[key].update(mAP, n=1)
		lF1_meters[key].update(F1, n=1)
		lIoU_meters[key].update(IoU, n=1)

		total_iters += 1
		if total_iters % 100 == 0:
			print("Batch: [{}/{}] | AP:\t{:.4f} | F1:\t{:.4f} | IoU:\t{:.4f}".format(
				total_iters, len(test_inharm_dataloader), 
				gmAP_meter.avg, gF1_meter.avg, gIoU_meter.avg, 
				))

		if save_flag:
    			save_output({'mask_main':mask_main, 'retouched_img':retouched_img, 'guide_map':guide_map},
						{'inharmonious':inharmonious, 'mask_gt':mask_gt}, 
						prediction_dir,
						data['img_path'][0],
						extra_infos={'f1':F1, 'iou':IoU},
						verbose=False)

		

print("\nModel:\t{}".format('MadisNet-{}'.format(opt.model)))
print("Average AP:\t{:.4f}".format(gmAP_meter.avg))
print("Average F1 Score:\t{:.4f}".format(gF1_meter.avg))
print("Average IoU:\t{:.4f}".format(gIoU_meter.avg))		
		
		
for key in sub_dataset:
	print("{}:".format(key))
	print("AP:\t{:.4f}\tF1:\t{:.4f}\tIoU:\t{:.4f}".format(lmAP_meters[key].avg, lF1_meters[key].avg, lIoU_meters[key].avg))
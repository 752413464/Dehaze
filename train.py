import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import DMSHN
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

def to_psnr(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [10.0 * np.log10(1 / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	dehaze_net = DMSHN.DMSHN().cuda()
	dehaze_net.apply(weights_init)

	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path)		
	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, drop_last =True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, drop_last= True)

	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	try:
		dehaze_net.load_state_dict(torch.load('snapshots/dehaze.pth'))  # 载入权重文件
		print('加载成功，继续训练')
	except:
		print('加载失败，重新训练')
	for epoch in range(config.num_epochs):
		for iteration, (img_orig, img_haze) in tqdm((enumerate(train_loader))):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean= dehaze_net(img_haze)

			loss = criterion(clean , img_orig)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			psnr_list = []
			ssim_list = []


			if ((iteration+1) % config.snapshot_iter) == 0:

				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		

		# Validation Stage
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()
			clean = dehaze_net(img_haze)
			psnr_list.extend(to_psnr(clean, img_orig))
			ssim_list.extend(to_ssim_skimage(clean, img_orig))
			# clean_image = img_haze - noise
			torchvision.utils.save_image(torch.cat((img_haze, clean, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

		torch.save(dehaze_net.state_dict(), config.snapshots_folder + "derain.pth")
		ave_psnr = sum(psnr_list) / len(psnr_list)
		ave_ssim = sum(ssim_list) / len(ssim_list)

		print('epoch:', epoch, 'dehazed', '---psnr:---', ave_psnr,
			  '---ssim:---', ave_ssim)


			





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="gt4/")
	parser.add_argument('--hazy_images_path', type=str, default="hazy4/")
	parser.add_argument('--lr', type=float, default=0.0003)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=2)
	parser.add_argument('--val_batch_size', type=int, default=2)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=200)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)








	

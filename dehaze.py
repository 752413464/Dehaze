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
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


if not os.path.exists('results'):
	os.makedirs('results')


def dehaze_image(x):

	data_hazy = Image.open(x)
	data_hazy = data_hazy.resize((1024,1024), Image.ANTIALIAS)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.cuda().unsqueeze(0)

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))
	clean_image = dehaze_net(data_hazy)

	torchvision.utils.save_image( clean_image, "results/" + str(i) + 'dehaze' + '.jpg')
	torchvision.utils.save_image( data_hazy, "results/" + str(i) + '.jpg')
	

if __name__ == '__main__':

	test_list = glob.glob("hazy4/*")
	for image in test_list:
		s_time = time.time()
		dehaze_image(image)
		t_time = time.time()
		cost_time = t_time -s_time
		print(image, 'cost_time = %.4f'%cost_time)

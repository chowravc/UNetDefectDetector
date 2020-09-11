import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# from torchvision import transforms
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pims
# import pathlib
# import torch.optim as optim
# from torch.autograd import Variable
# import skimage as skm
# import glob

def double_conv(in_c, out_c):

	conv = nn.Sequential(
		nn.Conv2d(in_c, out_c, kernel_size = 3),
		nn.ReLU(inplace = True),
		nn.Conv2d(out_c, out_c, kernel_size = 3),
		nn.ReLU(inplace = True)
	)

	return conv

class UNet(nn.Module):

	def __init__(self):

		super(UNet, self).__init__()

		self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.down_conv_1 = double_conv(1, 64) # Only 1 channel at the moment
		self.down_conv_2 = double_conv(64, 128)
		self.down_conv_3 = double_conv(128, 256)
	
		self.up_trans_1 = nn.ConvTranspose2d(
			in_channels = 256,
			out_channels = 128,
			kernel_size = 2,
			stride = 2
		)

		self.up_conv_1 = double_conv(256, 128)

		self.up_trans_2 = nn.ConvTranspose2d(
			in_channels = 128,
			out_channels = 64,
			kernel_size = 2,
			stride = 2
		)

		self.up_conv_2 = double_conv(128, 64)

		self.out = nn.Conv2d(
			in_channels = 64,
			out_channels = 1,
			kernel_size = 1
		)

	def forward(self, image):

		# bs, c, h, w
		# Encoder
		x1 = self.down_conv_1(image) # Keep
		x3 = self.max_pool_2x2(x1)

		x3 = self.down_conv_2(x3) # Keep
		x5 = self.max_pool_2x2(x3)

		x5 = self.down_conv_3(x5) # Keep

		# Decoder

		x = self.up_trans_1(x5)
		y = crop_img(x3, x)
		x = self.up_conv_1(torch.cat([x, y], 1))

		x = self.up_trans_2(x)
		y = crop_img(x1, x)
		x = self.up_conv_2(torch.cat([x, y], 1))
	
		x1, x3, x5, y = None, None, None, None,

		x = self.out(x)
		return x

if __name__ == '__main__':

	net = UNet()
	print(type(net))
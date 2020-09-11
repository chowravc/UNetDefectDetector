import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import datetime

from UNetTiny import UNet

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):

	for epoch in range(1, n_epochs + 1):  # <2>

		loss_train = 0.0

		for imgs, labels in train_loader:  # <3>

			imgs = imgs.to(device=device)
			labels = labels.to(device=device)
			outputs = model(imgs)  # <4>

			loss = loss_fn(outputs, crop_img(labels, outputs))  # <5>
			optimizer.zero_grad()  # <6>

			loss.backward()  # <7>

			optimizer.step()  # <8>

			loss_train += loss.item()  # <9>

		if epoch == 1 or epoch % 10 == 0:

			print('{} Epoch {}, Training loss {}, std{} , teststd {} '.format(
				datetime.datetime.now(), epoch, loss_train/len(train_loader), outputs.std(), labels.float().std()))

if __name__ == '__main__':

	uModel = UNet()

	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Training on device {device}.")
	loss_fn = nn.BCEWithLogitsLoss()
	lr = 1e-3
	model = uModel.to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)

	#losses.append(training_loop(100, optimizer, model, loss_fn, train_loader))

	PATH = './UNet.pth'
	torch.save(uModel.state_dict(), PATH)
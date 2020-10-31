import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cpu'#'cuda:0'
BATCHSIZE = 50

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()

        self.batch_size = BATCHSIZE
        self.widthImageNet = 224
        self.heightImageNet = 224
        self.widthMNIST = 28
        self.heightMNIST = 28
        self.mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False, device=device).reshape(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False, device=device).reshape(3,1,1)

        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        # Mask
        self.M = torch.ones(3, 224, 224, requires_grad=False, device=device)
        self.M[:,98:126, 98:126] = 0

        self.W = Parameter(torch.randn(self.M.shape, requires_grad=True))


    def hg(self, imagenet_label):
        return imagenet_label[:,:10]

    def forward(self, img, test=False):
        X = torch.zeros(self.batch_size, 3, self.heightImageNet, self.widthImageNet)

        X[:,:,98:126, 98:126] = img.repeat(1,3,1,1).data.clone()
        X = Variable(X, requires_grad=True).to(device)

        P = torch.tanh(self.W * self.M)
        if test:
	        plt.imshow((P + X[0]).detach().cpu().numpy().transpose(1,2,0))
        X = (X - self.mean) / self.std
        X_adv = P + X
        Y_adv = self.net(X_adv)
        return self.hg(Y_adv)


class Adversarial_Reprogramming():
    def __init__(self, args):
        self.mode = args.mode
        self.batch_size = BATCHSIZE
        self.lmd = 1e-8
        self.lr = 0.1
        self.decay = 0.96
        self.epochs = 100
        train_set = torchvision.datasets.MNIST('./', train=True, transform=transforms.ToTensor(), download=True)
        self.test_set = torchvision.datasets.MNIST('./', train=False, transform=transforms.ToTensor(), download=True)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.Adversary = Adversary()
        self.set_mode()

    def set_mode(self):
        if self.mode == 'train':
            self.criterion = torch.nn.CrossEntropyLoss()
            self.Adversary.to(device)
        elif self.mode == 'validate' or self.mode == 'test_one_image':
            self.Adversary.to(device)

    @property
    def get_W(self):
        return next(self.Adversary.parameters())

    def train(self):
        optimizer = torch.optim.Adam(self.Adversary.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=self.decay)
        for epoch in range(0, self.epochs + 1):
            for j, (image, label) in tqdm(enumerate(self.train_loader)):
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                output = self.Adversary(image)
                loss = self.criterion(output, label) + self.lmd*torch.norm(self.get_W) ** 2
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            print('epoch: %3d/%3d, loss: %.6f' % (epoch, self.epochs, loss.data.cpu().numpy()))
            torch.save({'W': self.get_W}, 'W.pt')
            self.validate()

    def validate(self):
        weigths = torch.load('W.pt')
        self.Adversary.load_state_dict(weigths, strict=False)
        acc = 0.0
        for (image, label) in self.test_loader:
            image, label = image.to(device), label.to(device)
            out = self.Adversary(image)
            pred = out.data.cpu().numpy().argmax(1)
            acc += sum(label.cpu().numpy() == pred) / float(len(label) * len(self.test_loader))
        print('Test accuracy: %.6f' % acc)


    def test_one_image(self, img_id):
        (image, label) = self.test_set[img_id]
        image = image.reshape(1,1,28,28)
        weigths = torch.load('W.pt')
        self.Adversary.load_state_dict(weigths, strict=False)
        image = image.to(device)
        out = self.Adversary(image, test=True)
        pred = out.data.cpu().numpy().argmax(1)
        print("Ground Truth: ", label, " Predicted: ", pred[0])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'validate', 'test_one_image'])
    parser.add_argument('-i', '--image_id', default=None, type=int,help='Image ID of Test MNIST image' )
    args = parser.parse_args()
    model = Adversarial_Reprogramming(args)
    if args.mode == 'train':
        model.train()
    elif args.mode == 'validate':
        model.validate()
    elif args.mode == 'test_one_image':
    	if args.image_id is None:
    		print("Please enter the path of the test image")
    		exit(0)	
    	# (img,label) = test_set[]
    	model.test_one_image(args.image_id)
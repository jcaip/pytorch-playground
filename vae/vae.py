from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
from tqdm import tqdm
  
use_cuda = torch.cuda.is_available()

        
class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2_mean = nn.Linear(hidden_size, latent_size)
        self.linear2_logvar = nn.Linear(hidden_size, latent_size)
    
    def forward(self, x):
        h1 = F.relu(self.linear1(x))
        return self.linear2_mean(h1), self.linear2_logvar(h1)
    
class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
    
    def forward(self, z):
        h1 = F.relu(self.linear1(z))
        return F.sigmoid(self.linear2(h1))
    
class VAE(nn.Module):
    
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(input_size, hidden_size, latent_size)

        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar
        
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            epsilon = torch.randn_like(std) 
            return mean + epsilon* std 
        else:
            return mean

    def generate(self, input_noise=None):
        if input_noise is None:
            input_noise = torch.randn(self.latent_size)

        input_noise = input_noise.cuda()
        return self.decoder(input_noise)

    def encoder(self, x):
        mean, logvar = self.encoder(x)
        return mean



def vae_loss(reconstruction, x, mean, logvar):
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, size_average=False)
    kl_loss = 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return reconstruction_loss - kl_loss
    
import os
root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])

train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

model = VAE(784, 100, 2)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

num_epochs = 50
for epoch in range(num_epochs):
    epoch_run = tqdm(enumerate(train_loader))
    for idx, (data, target) in epoch_run:
        data.cuda()
        target.cuda()
        optimizer.zero_grad()
        data_vec = data.view(-1, 784)
        data_vec = data_vec.cuda()
        reconstruction, mean, logvar = model(data_vec)
        loss = vae_loss(reconstruction, data_vec, mean, logvar)
        loss.backward()
        optimizer.step()
        epoch_run.set_description('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, idx* len(data), len(train_loader.dataset),  loss.item()))
w=10
h=10
fig=plt.figure(figsize=(8, 8))

columns = 25
rows = 25
p = 1
for c, dim1 in enumerate(np.linspace(-2, 2, num=columns)):
    for r, dim2 in enumerate(np.linspace(-2, 2, num=rows)):

        input_noise= torch.Tensor([dim1, dim2])
        test_image = model.generate(input_noise=input_noise).view(28, 28)

        test_image = test_image.cpu()
        fig.add_subplot(rows, columns, p)
        np_image = np.squeeze(test_image.data.numpy())
        plt.axis('off')
        plt.imshow(np_image)
        p += 1

fig.subplots_adjust(wspace=0)
fig.subplots_adjust(hspace=0)
plt.show()

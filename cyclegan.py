# -*- coding: utf-8 -*-

import argparse
from torch.utils.data import DataLoader
from dataset import ImageDataset
from torchvision import transforms
from PIL import Image
from models import *
import torch
import time
import itertools
import numpy as np
from utils import *
import datetime
from torchvision.utils import make_grid, save_image
import fire

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=128, help="size of image")
parser.add_argument("--batch_size", type=int, default=8, help="size of minibatch")
parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in the generator")
parser.add_argument("--max_epoch", type=int, default=200, help="numbers of epoch to train")
parser.add_argument("--lr", type=float, default=0.0002, help="the learning rate of Adam")
parser.add_argument("--decay_start_epoch", type=int, default=100, help="start of epoch to decay lr")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving models")
parser.add_argument("--save_images", type=str, default="images", help="directory of saving generator outputs")
parser.add_argument("--save_models", type=str, default="save_models", help="directory of saving models")

args = parser.parse_args()

transforms_ = transforms.Compose([
        transforms.Resize(int(args.img_size*1.12), Image.BICUBIC),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

dataset = ImageDataset(args.dataset_name, transforms_)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = ImageDataset(args.dataset_name, transforms_, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

G = GeneratorResNet(args.n_residual_blocks)
F = GeneratorResNet(args.n_residual_blocks)
D_X = Discriminator()
D_Y = Discriminator()

criterion_gan = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    G, F = G.cuda(), F.cuda()
    D_X, D_Y = D_X.cuda(), D_Y.cuda()
    criterion_gan.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

optimizer_Gen = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_Gen, lr_lambda=LambdaLR(args.max_epoch, args.decay_start_epoch).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(args.max_epoch, args.decay_start_epoch).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(args.max_epoch, args.decay_start_epoch).step)

prev_time = time.time()
for epoch in range(args.max_epoch):
    for i, images in enumerate(dataloader):
        real_X = images['X'].type(Tensor)
        real_Y = images['Y'].type(Tensor)  # 1 x 3 x 128 x 128
        
        G.train(), F.train()

        fake_X = F(real_Y)
        fake_Y = G(real_X)
        
        real = Tensor(np.ones((args.batch_size, 1, args.img_size//2**4, args.img_size//2**4)))
        fake = Tensor(np.zeros((args.batch_size, 1, args.img_size//2**4, args.img_size//2**4)))
        
        # ------------------------
        # Training Generator
        # ------------------------
        loss_gan_G = criterion_gan(D_Y(fake_Y), real)
        loss_gan_F = criterion_gan(D_X(fake_X), real)
        loss_gan = (loss_gan_G + loss_gan_F) / 2
        
        loss_cycle_G = criterion_cycle(G(fake_X), real_Y)
        loss_cycle_F = criterion_cycle(F(fake_Y), real_X)
        loss_cycle = (loss_cycle_G + loss_cycle_F) / 2
        
        loss_identity_G = criterion_identity(G(real_Y), real_Y)
        loss_identity_F = criterion_identity(F(real_X), real_X)
        loss_identity = (loss_identity_G + loss_identity_F) / 2
        
        loss_G = loss_gan + 10.0 * loss_cycle + 5.0 * loss_identity
        
        optimizer_Gen.zero_grad()
        loss_G.backward()
        optimizer_Gen.step()
        
        # ------------------------
        # Training Discriminator X
        # ------------------------
        loss_D_X_real = criterion_gan(D_X(real_X), real)
        loss_D_X_fake = criterion_gan(D_X(fake_X.detach()), fake)
        loss_D_X = loss_D_X_real + loss_D_X_fake
        
        optimizer_D_X.zero_grad()
        loss_D_X.backward()
        optimizer_D_X.step()
        
        # ------------------------
        # Training Discriminator Y
        # ------------------------
        loss_D_Y_real = criterion_gan(D_Y(real_Y), real)
        loss_D_Y_fake = criterion_gan(D_Y(fake_Y.detach()), fake)
        loss_D_Y = loss_D_Y_real + loss_D_Y_fake
        
        optimizer_D_Y.zero_grad()
        loss_D_Y.backward()
        optimizer_D_Y.step()
        
        # Print log
        batch_done = epoch * len(dataloader) + i
        batch_left = args.max_epoch * len(dataloader) - batch_done
        time_left = datetime.timedelta(seconds=batch_left * (time.time()-prev_time))
        prev_time = time.time()
        print(f'{epoch}/{args.max_epoch} {i}/{len(dataloader)} loss_gan: {loss_G.item():.2f} loss_D_X: {loss_D_X.item():.2f} loss_D_Y: {loss_D_Y.item():.2f} Remaining time: {time_left}')
        
    # Save generator outputs if epoch at sample interval
    if epoch % args.sample_interval == 0:
        test_images = next(iter(test_dataloader))
        G.eval(), F.eval()
        real_X = test_images['X'].type(Tensor)
        real_Y = test_images['Y'].type(Tensor)
        fake_X = F(real_Y)
        fake_Y = G(real_X)
        
        real_X = make_grid(real_X, nrow=5, normalize=True)
        fake_X = make_grid(fake_X, nrow=5, normalize=True)
        real_Y = make_grid(real_Y, nrow=5, normalize=True)
        fake_Y = make_grid(fake_X, nrow=5, normalize=True)
        
        output = torch.cat((real_X, fake_X, real_Y, fake_Y), dim=1)
        save_image(output, f'{args.save_images}/{epoch}.jpg')
    
    # Update learning rate
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()
    
    # Save models if epoch at checkpoint interval
    if epoch % args.checkpoint_interval == 0:
        torch.save(G.state_dict(), f'{args.save_models}/G_{epoch}.pth')
        torch.save(F.state_dict(), f'{args.save_models}/F_{epoch}.pth')
        torch.save(D_X.state_dict(), f'{args.save_models}/D_X_{epoch}.pth')
        torch.save(D_Y.state_dict(), f'{args.save_models}/D_Y_{epoch}.pth')
        
if __name__ == '__main__':
    fire.Fire()
            

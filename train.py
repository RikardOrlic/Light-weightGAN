import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import ImageDataset, InfiniteSamplerWrapper
from torchvision import utils as vutils
import os

import argparse
import random
from tqdm import tqdm

import network

from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.LPIPS(net='vgg')

def train_d(disc, data, label='real'):
    if label == 'real':
        #part = network.center_crop_image
        pred, i_, i_part = disc(data, label)

        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
            percept( i_, F.interpolate(data, i_.shape[2]) ).sum() + \
            percept( i_part, F.interpolate(network.center_crop_image(data, 128), i_part.shape[2]) ).sum()
        
        err.backward()
        return pred.mean().item(), i_, i_part
    
    else:
        pred = disc(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):
    run_name = args.run_name
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    save_interval = 1000

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)

    dataset = ImageDataset(root=data_root, transform=trans)

    dataLoader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), pin_memory=True))
    
    gen = network.Generator(im_size)
    disc = network.Discriminator(im_size)

    gen.to(device)
    disc.to(device)
    percept.to(device)

    genOpt = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    discOpt = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

    if checkpoint != 'None':
        ckpt = torch.load(f'training_runs/{checkpoint}.pth')
        gen.load_state_dict(ckpt['g'])
        disc.load_state_dict(ckpt['d'])
        genOpt.load_state_dict(ckpt['opt_g'])
        discOpt.load_state_dict(ckpt['opt_d'])
    
    os.makedirs(f'training_runs/{run_name}', exist_ok=True)

    fixed_noise = torch.normal(0, 1, size=(25, 256, 1, 1)).to(device)

    for iteration in tqdm(range(1, total_iterations + 1)):
        real_images = next(dataLoader)
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        noise = torch.normal(0, 1, size=(current_batch_size, 256, 1, 1)).to(device)  #torch.Tensor(current_batch_size, 256).normal_(0, 1).to(device)

        fake_images = gen(noise)

        real_images = DiffAugment(real_images, policy=policy)
        fake_images = DiffAugment(fake_images, policy=policy)

        #train disc
        disc.zero_grad()
        #err_d, rec_img_all, rec_img_small, rec_img_part = train_d(disc, real_images, label="real")
        err_d, i_, i_part = train_d(disc, real_images, label="real")
        #train_d(disc, torch.stack([fi.detach() for fi in fake_images]), label="fake")
        train_d(disc, fake_images.detach(), label="fake")
        discOpt.step()

        #train gen
        gen.zero_grad()
        pred_g = disc(fake_images, 'fake')
        err_g  = -pred_g.mean()

        err_g.backward()
        genOpt.step()

        if iteration % 100 == 0:
            print("GAN: disc loss: %.5f    gen loss: %.5f"%(err_d, -err_g.item()))
            with torch.no_grad():
                vutils.save_image(gen(fixed_noise).add(1).mul(0.5), f'training_runs/{run_name}/{iteration}.jpg', nrow=5)
                #save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)


            
        if iteration % save_interval == 0 or iteration == total_iterations:
            torch.save({'g':gen.state_dict(),
                        'd':disc.state_dict(),
                        'opt_g': genOpt.state_dict(),
                        'opt_d': discOpt.state_dict()}, f'training_runs/{run_name}/{iteration}.pth')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='desc')

    parser.add_argument('--run_name', type=str, default='default', help='name of the training run')
    parser.add_argument('--path', type=str, default='dataset/klimt256', help='path of resource dataset')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')

    args = parser.parse_args()
    print(args)

    train(args)

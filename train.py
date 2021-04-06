import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from pywt import dwt2, idwt2
import pywt

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()
#===========================dwt=============================
def haar_transform(img):
  coeffs2 = pywt.dwt2(img, 'haar')
  LL, (LH, HL, HH) = coeffs2
  LL = np.expand_dims(LL, axis=2)
  LH = np.expand_dims(LH, axis=2)
  HL = np.expand_dims(HL, axis=2)
  HH = np.expand_dims(HH, axis=2)
  merge_img = np.concatenate([LL,LH,HL,HH],axis=2)
  return merge_img

def tensor_dwt(batch_img):
    dim = batch_img[0][0][0].size(0)//2
    new_batch_img = torch.zeros([opt.batchSize, 4, dim, dim])
    for i in range(batch_img.size(0)):
        img_tensor = batch_img[i]
        img_array = img_tensor.cuda().data.cpu().numpy()
        gray_img = np.squeeze(img_array, axis=0)
        wave_img = haar_transform(gray_img)
        tran_wave_img = np.transpose(wave_img, (2, 0, 1))
        tensor_wave_img = torch.from_numpy(tran_wave_img)
        new_batch_img[i] = tensor_wave_img
    return new_batch_img
#===========================dwt=============================



#===========================idwt============================
def haar_inverse_transform(np_out_imgs):
  LL, (LH, HL, HH) = np_out_imgs[0],(np_out_imgs[1],np_out_imgs[2],np_out_imgs[3])
  ori_img = idwt2((LL, (LH, HL, HH)), 'haar')
  return ori_img

def tensor_idwt(batch_img):
  dim = batch_img[0][0][0].size(0) * 2
  new_batch_img = torch.zeros([opt.batchSize, 1, dim, dim])
  for i in range(batch_img.size(0)):
    img_tensor = batch_img[i]
    img_array = img_tensor.cuda().data.cpu().numpy()
    np_out_imgs = np.split(img_array, 4)
    ori_img = haar_inverse_transform(np_out_imgs)
    tensor_ori_img = torch.from_numpy(ori_img)
    new_batch_img[i] = tensor_ori_img
  return new_batch_img
#===========================idwt============================

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=4, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # criterion.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(net).to(device)
    criterion.to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    ep = []
    ps = []
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise#torch.Size([32, 1, 40, 40])
            #================dwt================
            imgn_train = tensor_dwt(imgn_train)
            img_train = tensor_dwt(img_train)
            noise = tensor_dwt(noise)
            #====================================
            img_train, imgn_train = img_train.to(device), imgn_train.to(device)
            noise = noise.to(device)
            out_train = model(imgn_train)#torch.Size([32, 4, 20, 20])

            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = imgn_train - model(imgn_train)
            out_train = tensor_idwt(out_train)
            out_train = torch.clamp(out_train, 0., 1.)
            img_train = tensor_idwt(img_train)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                imgn_val = tensor_dwt(imgn_val)
                img_val = tensor_dwt(img_val)
                noise = tensor_dwt(noise)
                img_val, imgn_val = img_val.to(device), imgn_val.to(device)
                out_val = imgn_val-model(imgn_val)
                out_val = tensor_idwt(out_val)
                out_val = torch.clamp(out_val, 0., 1.)
                img_val = tensor_idwt(img_val)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        ep.append(epoch + 1)
        ps.append(psnr_val)
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # # log the images
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        name = "net_epoch%d_PSNR%.4f.pth" % (epoch + 1, psnr_val)
        torch.save(model.state_dict(), os.path.join(opt.outf, name))
    # save chart
    plt.plot(ep, ps)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.title("PSNR values during training")
    plt.savefig('./psnr_val.jpg')
    plt.show()

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()

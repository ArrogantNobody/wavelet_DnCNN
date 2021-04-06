import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
from pywt import dwt2, idwt2
import pywt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
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
    new_batch_img = torch.zeros([batch_img.size(0), 4, dim, dim])
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
  new_batch_img = torch.zeros([batch_img.size(0), 1, dim, dim])
  for i in range(batch_img.size(0)):
    img_tensor = batch_img[i]
    img_array = img_tensor.cuda().data.cpu().numpy()
    np_out_imgs = np.split(img_array, 4)
    ori_img = haar_inverse_transform(np_out_imgs)
    tensor_ori_img = torch.from_numpy(ori_img)
    new_batch_img[i] = tensor_ori_img
  return new_batch_img
#===========================idwt============================

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=4, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)#torch.Size([1, 1, x, x])
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise#torch.Size([1, 1, x, x]), 这里我们要对其进行小波变换，变成([1, 4, x/2, x/2])
        INoisy = tensor_dwt(INoisy)
        print('1', INoisy.shape)
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = model(INoisy)
            Out = torch.clamp(Out, 0., 1.)#输出的维度和输入的维度理应相同，但是我们在计算psnr的时候需要将结果反变换才可以和原始图像进行对比
            print('2', Out.shape)
        Out= tensor_idwt(Out)
        print('3', Out.shape)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

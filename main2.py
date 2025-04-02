import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import random
import shutil
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.net import net
from data import get_training_set, get_eval_set
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='/content/drive/MyDrive/PairLIE-training-dataset-v2/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='/content/drive/MyDrive/PairLIE_checkpoints_v2/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')
opt = parser.parse_args()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True

def train():
    model.train()
    loss_print = 0
    for iteration, batch in enumerate(training_data_loader, 1):

        im1, im2, im3, file1, file2, file3 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        im1 = im1.cpu()
        im2 = im2.cpu()
        im3 = im3.cpu()

        L1, R1, X1 = model(im1)
        L2, R2, X2 = model(im2)
        L3, R3, X3 = model(im3)

        loss_C1 = C_loss(R1, R2)
        loss_C2 = C_loss(R2, R3)
        loss_C3 = C_loss(R1, R3)
        loss1 = (loss_C1 + loss_C2 + loss_C3) / 3

        loss_R1 = R_loss(L1, R1, im1, X1)
        loss_R2 = R_loss(L2, R2, im2, X2)
        loss_R3 = R_loss(L3, R3, im3, X3)
        loss2 = (loss_R1 + loss_R2 + loss_R3) / 3

        loss_P1 = P_loss(im1, X1)
        loss_P2 = P_loss(im2, X2)
        loss_P3 = P_loss(im3, X3)
        loss3 = (loss_P1 + loss_P2 + loss_P3) / 3

        loss = loss1 * 1 + loss2 * 1 + loss3 * 500

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print += loss.item()

        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(
                epoch, iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']
            ))
            loss_print = 0


def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_v2_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ')
model = net().cpu()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

score_best = 0
# shutil.rmtree(opt.save_folder)
# os.mkdir(opt.save_folder)
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % opt.snapshots == 0:
        checkpoint(epoch)

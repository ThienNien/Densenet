'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from dataloader import *
from torch.utils.data import DataLoader

import csv
parser = argparse.ArgumentParser(description='PyTorch Custom Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
img_size =[32,32]

# Data
print('==> Preparing data..')
train_data = Image_Loader(root_path="./train_set.csv", image_size=img_size, transforms_data=True)
test_data = Image_Loader_test(root_path="./test_set.csv", image_size=img_size, transforms_data=True)
total_train_data = len(train_data)
total_test_data = len(test_data)
print('total_train_data:',total_train_data, 'total_test_data:',total_test_data)

# Generate the batch in each iteration for training and testing
trainloader = DataLoader(train_data, batch_size=128, shuffle=True) # shuffle = true nghĩa là có sáo trộn ảnh khi lấy data bath_size
testloader = DataLoader(test_data, batch_size=100)
# Model
print('==> Building model..')
net = DenseNet121()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
train_acc_list = []
test_acc_list = []
train_loss_list = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.type(torch.int64)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    global train_acc_list,train_loss_list
    train_acc_list.append(100.*correct/total)
    train_loss_list.append((train_loss/(batch_idx+1))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets = targets.type(torch.int64)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    global test_acc_list
    test_acc_list.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+5):
    train(epoch)
    test(epoch)
    scheduler.step()

with open('train_acc.csv','r',newline='') as tr_file:
    tr_writer = csv.writer(tr_file)
    tr_writer.writerow(['Epoch','Accuracy'])
    for item in train_acc_list:
        tr_writer.writerow(item)

with open('test_acc.csv','r',newline='') as t_file:
    t_writer = csv.writer(t_file)
    t_writer.writerow(['Epoch','Accuracy'])
    for item in test_acc_list:
        tr_writer.writerow(item)

with open('train_loss.csv','r',newline='') as l_file:
    l_writer = csv.writer(l_file)
    l_writer.writerow(['Epoch','Loss'])
    for item in train_loss_list:
        l_writer.writerow(item)
# for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

# for os and plot 
import os
import shutil
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# for my models
from models import *

# for dataset
# given a batch of images, can I scale them all ? 
import mymnist

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'mnist', help='mnist | cifar10')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.01, help='learning rate of generator, default=0.01')
parser.add_argument('--lrD', type=float, default=0.01, help='learning rate of discriminator, default=0.01')
parser.add_argument('--ratio', type=float, default=1e-2, help = 'ratio of GAN loss')
parser.add_argument('--mode', type = str, default = 'MSE', help = 'MSE | GAN | visual')
parser.add_argument('--resume', type = bool, default = True, help = 'True | False')

parser.add_argument('--G', default='G_best.pth.tar', help="path to netG (to continue training)")
parser.add_argument('--D', default='', help="path to netD (to continue training)")

opt = parser.parse_args()
print(opt)

# some helper function
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(state, is_best, filename='G.pth.tar'):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    print('==> Saving checkpoint')
    torch.save(state, 'checkpoint/' + filename)
    if is_best:
        print('This is best checkpoint ever, copying')
        shutil.copyfile('checkpoint/'+filename, 'checkpoint/'+'G_best.pth.tar')
        
        
def train(train_loader, model, criterion, optimizer, epoch):
    print('==> Starting Training Epoch {}'.format(epoch))
    
    losses = AverageMeter()
    
    model.train()  # Set the model to be in training mode
    
    for batch_index, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # update loss
        losses.update(loss.data[0], inputs.size(0))
        
        # Backward
        optimizer.zero_grad()  # Set parameter gradients to zero
        loss.backward()        # Compute (or accumulate, actually) parameter gradients
        optimizer.step()       # Update the parameters
        
    print('==> Train Epoch : {}, Average loss is {}'.format(epoch, losses.avg))
    
def validate(validate_loader, model, criterion, epoch):   
    
    print('==> Starting validate')
    model.eval()
    
    losses = AverageMeter()
    
    correct = 0
    
    for batch_idx, (inputs, targets) in enumerate(validate_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # update loss, accuracy
        losses.update(loss.data[0], inputs.size(0))
        
    print('==> Validate Epoch : {}, Average loss, {:.4f}'.format(
        epoch,losses.avg))
    
    return losses.avg


def imshow(img):
    img = (img - img.min()) / (img.max() - img.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    plt.show()
    


    
# GPU setting
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('==> GPU is available and will train with GPU')
else :
    print('==> GPU is not available and will train with CPU')

# set dataset
if opt.dataset == 'mnist':
    mnist = mymnist.MNIST('data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))
    loader = torch.utils.data.DataLoader(mnist,batch_size=opt.batch_size, shuffle=True)
    
    validate_mnist = mymnist.MNIST('data', download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
    validate_loader = torch.utils.data.DataLoader(validate_mnist,batch_size=opt.batch_size, shuffle=True)  
    print('==> Dataset and Dataloader have been set !')
    
     
    
# train with MSE loss

if opt.mode == 'MSE':
    # parameters
    start_epoch = 0
    best_prec1 = 10   
    betas = (0.9, 0.999)
    
    # network
    G = Adversarial_G()
    optimizerG = torch.optim.Adam(G.parameters(), lr = opt.lrG, betas = betas, weight_decay=5e-4)
    criterionG = nn.MSELoss()
    
    # resume
    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint found")
        print('\n')
    
    if use_cuda:
        G.cuda()
        criterionG.cuda()
                 
    
    print('==> Train with MSE loss')
    print('==========================')
        
    for epoch in range(start_epoch, opt.niter):
        # train for one epoch
        train(loader, G, criterionG, optimizerG, epoch)
        prec1 = validate(validate_loader, G, criterionG, epoch)
            

        # remember minimal loss and save checkpoint
        # only G network
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': G.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
                  
    print('Training with MSE is done')
                  
    print('Save the ouputs : ')
    G.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            G_outputs = G(inputs)
            
            torchvision.utils.save_image(inputs.data, 'inputs_samples_{}_{}.png'.format(opt.mode, opt.niter))
            torchvision.utils.save_image(G_outputs.data, 'outputs_samples_{}_{}.png'.format(opt.mode,opt.niter))
            torchvision.utils.save_image(targets.data, 'origin_samples_{}_{}.png'.format(opt.mode,opt.niter))

if opt.mode == 'GAN':
    print('I love GAN\n')
    
    # networks
    G = Adversarial_G()
    D = Adversarial_D(input_size = 28)
  
    optimizerG = torch.optim.RMSprop(G.parameters(), lr = opt.lrG)
    optimizerD = torch.optim.RMSprop(D.parameters(), lr = opt.lrD)
    
    criterionG = nn.MSELoss()
    
    if opt.resume:
        if os.path.isfile('checkpoint/'+ 'G_best.pth.tar'):
            print("==> loading checkpoint {}".format('G_best.pth.tar'))
            checkpoint = torch.load('checkpoint/'+ 'G_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format('G_best.pth.tar'))
        else:
            print("=> no checkpoint found")
        print('\n')
    
    one = torch.FloatTensor([1]*opt.batch_size)
    mone = one * -1

    if use_cuda:
        G.cuda()
        D.cuda()
        one, mone = one.cuda(), mone.cuda()
          
    isstart = True # at first, train D with more iterations
    for epoch in range(opt.niter):
        print('==> Start training epoch {}'.format(epoch))
        
        data_iter = iter(loader)
        i = 0
        while i < len(loader):
            # update D network
            for p in D.parameters():
                p.requires_grad = True
            if isstart:
                Diters = 500
            else :
                Diters = 5
                
            j=0
            while j < Diters and i < len(loader) - 1:
                j+=1
                
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
                    
                data = data_iter.next()
                i+=1
                
                # train with real
                D.zero_grad()        
                small_img, origin_img = data
                if use_cuda:
                    origin_img = origin_img.cuda()
                    small_img = small_img.cuda()
                
                inputv = Variable(origin_img)
                errD_real = D(inputv)
                errD_real.backward(one)
                
                # train with fake
                noisev = Variable(small_img, volatile = True) # totally freeze G
                fake = Variable(G(noisev).data)
                inputv = fake
                errD_fake = D(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()
            
            # update G
            for p in D.parameters():
                p.requires_grad = False # to avoid computation
            
            G.zero_grad()
            
            data = data_iter.next()
            i += 1 
                
            small_img, origin_img = data
            if use_cuda:
                origin_img = origin_img.cuda()
                small_img = small_img.cuda()
                    
            small_img_var, origin_img_var = Variable(small_img), Variable(origin_img)
            fake = G(small_img_var)
            errG_GAN = D(fake)
            errG_GAN.backward(opt.ratio * one, retain_variables=True)
                      
            errG_content = criterionG(fake, origin_img_var)
            errG_content.backward()
                
            #errG = errG_GAN + errG_content

            optimizerG.step()
            
            torch.save({'epoch': epoch + 1,
                        'state_dict': G.state_dict(),
                        'best_prec1': 0,
                       }, 'checkpoint/G_GAN.pth.tar' )
            torch.save({'epoch': epoch + 1,
                        'state_dict': D.state_dict(),
                        'best_prec1': 0,
                       }, 'checkpoint/D_GAN.pth.tar' )
            
if opt.mode == 'visual':
    G = Adversarial_G()
    if opt.resume:
        if os.path.isfile('checkpoint/'+ opt.G):
            print("==> loading checkpoint {}".format(opt.G))
            checkpoint = torch.load('checkpoint/'+ opt.G)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            G.load_state_dict(checkpoint['state_dict'])
            print('==> {} has been loaded'.format(opt.G))
        else:
            print("=> no checkpoint found")
        print('\n')
        
    G.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index == 0:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            G_outputs = G(inputs)
            
            imshow(torchvision.utils.make_grid(inputs.data))
            imshow(torchvision.utils.make_grid(G_outputs.data))
            imshow(torchvision.utils.make_grid(targets.data))


        

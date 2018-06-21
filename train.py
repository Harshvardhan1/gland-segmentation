import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import numpy as np
from scipy import misc
import model
import augmentation as aug
import random
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from loader import glandDataset,glandTest
import losses

### LOAD dataset
transformed_dataset = glandDataset(root_dir='GLAS_dataset/')

dataloader = DataLoader(transformed_dataset, batch_size=3,
                        shuffle=True, num_workers=4)

## Instantiate architecture

Uarch = model.UNet(True)
Uarch.cuda()
try:
    model_dict = torch.load('model/checkpoint.pth')
    Uarch.load_state_dict(model_dict['state_dict'])
    print('model weights loaded')
except:
    print("could not load model weights")

#print Uarch.parameters()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print count_parameters(Uarch)
#import pdb;pdb.set_trace()

epochs = 50
lr = 0.01
optimizer = optim.SGD(Uarch.parameters(), lr=lr,momentum=0.99)
loss_sum = 0


def save_checkpoint(state,filepath='model/checkpoint.pth'):
    torch.save(state,filepath)


def train():
    # Training:
    running_loss = 0
    for epoch in range(201):
        Uarch.eval()
        for i,data in enumerate(dataloader,0):
            
            image,mask = data
            image = Variable((image.float()).cuda())
            mask = Variable(((mask > 0).float()).cuda())

            optimizer.zero_grad()
            
            outputs = Uarch(image)
            #import pdb;pdb.set_trace()
            mask_pred = F.sigmoid(outputs).view(-1)
            true_mask = mask.view(-1)

            loss = losses.BinaryCrossEntropyLoss2d().forward(outputs, mask) \
                    + losses.SoftDiceLoss().forward(outputs,mask)

            loss.backward()
            optimizer.step()

            #import pdb;pdb.set_trace()
            running_loss += loss.data[0]
            
        if(epoch%100==0):
            print ('Epoch: ' +str(epoch) + ' Loss: ' + str(running_loss/100.0))
            running_loss = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': Uarch.state_dict(),
                'optimizer' : optimizer.state_dict(),
            })

            test()




transformed_testset = glandTest(root_dir='GLAS_dataset/')

valloader = DataLoader(transformed_testset, batch_size=1,
                        shuffle=True, num_workers=4)

def save_output(image,output,mask):
    image = image.squeeze(0)
    inp = TF.to_pil_image(image)
    inp.save('images/input.png')
    outputs = output.squeeze(0)
    outputs = F.sigmoid(outputs.data).cpu()
    outputs = (outputs>0.7).float()
    #import pdb;pdb.set_trace()
    result = TF.to_pil_image(outputs)
    result.save('images/output.png')
    mask = mask.squeeze(0)
    #import pdb;pdb.set_trace()
    mask = TF.to_pil_image(mask)
    #import pdb;pdb.set_trace()
    mask.save('images/true.png')


def test():
    model_dict = torch.load('model/checkpoint.pth')
    Uarch.load_state_dict(model_dict['state_dict'])
    Uarch.eval()
    running_loss = 0
    for i,data in enumerate(valloader,0):
        
        image,mask = data
        img = Variable(image.cuda(),volatile=True)
        mask = ((mask > 0).float())
        label = Variable(mask.cuda(),volatile=True)

        outputs = Uarch(img)

        #outputs = torch.cat((outputs,outputs),1)
        mask_pred = F.sigmoid(outputs).view(-1)
        true_mask = label.view(-1)
        loss = F.binary_cross_entropy(input=mask_pred,target=true_mask)
        
        #show input
        #save_output(image,outputs,mask)
        #sys.exit()
        running_loss += loss.data[0]
    print("test loss: " + str(running_loss/60.0))


if __name__== '__main__':
    test()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla RNN for MNIST classification

"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 64
N_INPUTS = 28*28 #image size is 28x28 and one input so, flatten it to 1D 
N_HIDDEN = 64 #hidden state 
N_OUTPUTS = 10 #0-#9
N_EPOCHS = 10

# functions to show an image
def imshow(img):
     #img = img / 2 + 0.5     # unnormalize
     npimg = img.numpy()
     plt.imshow(np.transpose(npimg, (1, 2, 0)))
     
#accuracy calculations
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
     
class RNN(nn.Module):
    
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = data_size + hidden_size
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.input_size, output_size)  
        self.softmax = nn.LogSoftmax(dim=1) #0-9 categories

    def forward(self, data, hidden):
        combined = torch.cat((data,hidden), 1)  # combine 
        hidden = self.i2h(combined)
        output = self.h2o(combined)              #we may use hidden alone
        output = self.softmax(output)
        return hidden, output

    def init_hidden(self):
        return torch.zeros(BATCH_SIZE,self.hidden_size) # batch size X hidden size

if __name__ == '__main__':

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    # list all transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # download and load training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2,drop_last=True)
     
    # download and load testing dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False,
    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2,drop_last=True)
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
     
    # show images
    imshow(torchvision.utils.make_grid(images))
    
    rnnmodel = RNN(N_INPUTS,N_HIDDEN,N_OUTPUTS).to(device)
    #optimizer and loss func
 
    lr=0.001
    optimizer = torch.optim.SGD(rnnmodel.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    
    all_losses=[]
    all_accuracy=[]
    print('Epochs # ' , end='')
    
    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    
        train_running_loss = 0.0
        train_acc = 0.0
        rnnmodel.train()
        
        # TRAINING ROUND
        for i, data in enumerate(trainloader):
            
            # zero the parameter gradients
            optimizer.zero_grad()
             
            # reset hidden states
            hidden = rnnmodel.init_hidden() 
            hidden = hidden.to(device)
             
            # get the inputs
            inputs, labels = data
                   
            inputs = inputs.view(-1, 28*28).to(device) 
     
            # forward + backward + optimize
            outputs,hidden = rnnmodel(inputs,hidden)
             
            outputs = outputs.to(device)
            hidden = hidden.to(device)
             
            labels = labels.to(device)
            loss = loss_func(outputs, labels)
             
            loss.backward()
            optimizer.step()
     
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{N_EPOCHS}], "
                    f"Step [{(i + 1)*BATCH_SIZE}/{len(trainset)}], "
                    f"Loss: {loss.item():.4f}"
                )
             
        rnnmodel.eval()
        print('%d '%( epoch),end='')
        all_losses.append(train_running_loss / i)
        all_accuracy.append(train_acc/i)
        
    test_acc = 0.0
 
    for i, data in enumerate(testloader, 0):
        print(i)
        inputs, labels = data
        inputs = inputs.view(-1, 28*28).to(device)
        hidden = rnnmodel.init_hidden().to(device) 
        labels = labels.to(device)
     
        outputs,hidden = rnnmodel(inputs,hidden)
     
        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
         
    print('Test Accuracy: %.2f'%( test_acc/i))
    
    t=inputs[0:15].view(-1, 28*28).to(device)
    h=torch.zeros(t.size(0),64).to(device)

    #model returns ouput nad hidden, ignore hidden. output is size of batch,
    #take the max and round it
    o,h=rnnmodel(t,h)
    #compare predicted vs original
    print(torch.max(o, 1)[1].data)
    print(labels[0:15])

    
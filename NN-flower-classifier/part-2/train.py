'''
Author  :   Aaron Balson Caroltin .J
Purpose :   Image Classifier (Command line version) to train model with flower datasets
            under 102 categories using transfer learning (imagenet) and NN classifier
For     :   Udacity AI python nanodegree - Aug-Sep 2018
'''

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse
import nn_utils 

def train_model(model, criterion, optimizer, epochs, train_losses, test_losses, train_dataloader, valid_dataloader, device):
    '''
    To train NN model using transfer learning from imagenet (densenet121)
    '''
    print("Training started ..")
    model.to(device)
    
    from_range = model.completed_epoch + 1
    to_range = from_range + epochs
    running_loss = 0
    trainlen = len(train_dataloader)
    validlen = len(valid_dataloader)
    
    for epoch in range(from_range, to_range):
        for inputs, labels in train_dataloader:
            with torch.set_grad_enabled(True):
                model.train()
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        else:
            with torch.no_grad():
                model.eval()
                valid_loss = 0
                accuracy = 0
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch}/{to_range-1}.. "
                  f"Train loss: {running_loss/trainlen:.3f}.. "
                  f"Validation loss: {valid_loss/validlen:.3f}.. "
                  f"Test accuracy: {accuracy/validlen:.3f}")

            train_losses.append(running_loss/trainlen)
            test_losses.append(valid_loss/validlen)

            running_loss = 0
            
        model.completed_epoch += 1
            
    print("Training done !!")  
    return model, criterion, optimizer, train_losses, test_losses



def train(data_dir, gpu, save_dir, l_rate, dropout, epochs, arch, hidden_1, hidden_2):
    
    '''
    Performs dataloading, NN building, train on images and save checkpoint
    '''

    # Load data
    train_transform, valid_transform, test_transform, train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader = nn_utils.load_data(data_dir)
    
    print("Data loaded successfully ..")
    # imagenet architecture used for transfer learning
    model_input_units = 1024
    model_hidden_1 = hidden_1
    model_hidden_2 = hidden_2
    model_output_units = 102
    model_drop_out = dropout
    model_class_state_dict = OrderedDict([])
    model_class_to_idx = {}
    model_nn_arch = arch
    optim_learn_rate = l_rate
    optim_state_dict = OrderedDict([])
    optim_nn_arch = "Adam"

    # use GPU if available (only nVidia cuda based cards)
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu=="gpu" else "cpu")
    print("Device setup done ..", device)
    
    # build NN components
    model = nn_utils.build_model(model_input_units, model_hidden_1, model_hidden_2, model_output_units, 
                        model_drop_out, model_class_state_dict, model_class_to_idx, model_nn_arch)
    model.completed_epoch=0
    print("Initial Model created ..")
    
    criterion, optimizer = nn_utils.build_optimizer(model, optim_learn_rate, optim_state_dict, optim_nn_arch)
    print("Criterion, Optimizer created ..")
    
    model.to(device)
    print("Model pushed to ", device)
    
    #train model
    model, criterion, optimizer, train_losses, test_losses = train_model(model, criterion, optimizer, epochs, [], [],train_dataloader, valid_dataloader, device) 

    #test model
    print("Calculating Model Accuracy ..")
    nn_utils.test_model(model, test_dataloader, device)

    #save model
    print("Saving to ", save_dir, " ..")
    model.class_to_idx = train_dataset.class_to_idx
    state = {'model_input_units' : model_input_units,
            'model_hidden_1' : model_hidden_1,
            'model_hidden_2' : model_hidden_2,
            'model_output_units' : model_output_units,
            'model_drop_out' : model_drop_out,
            'model_state_dict' : model.state_dict(),
            'model_class_state_dict' : model.classifier.state_dict(),
            'model_class_to_idx' : model.class_to_idx,
            'model_nn_arch' : model_nn_arch,
            'model_completed_epoch': model.completed_epoch,
            'optim_learn_rate' : optim_learn_rate,
            'optim_state_dict': optimizer.state_dict(),
            'optim_nn_arch' : optim_nn_arch}
    nn_utils.save_checkpoint(model, criterion, optimizer, save_dir, state, device)
    
    print("")
    done = input("Model is trained. Press any key to close")

def train_settings():
    
    '''
    To collect user options for starting training phase
    '''
    arg = argparse.ArgumentParser(description='Train.py')
    arg.add_argument('data_dir', action="store", default="./flowers")
    arg.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    arg.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    arg.add_argument('--learning_rate', dest="l_rate", action="store", default=0.001)
    arg.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
    arg.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    arg.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)
    arg.add_argument('--hidden_units_1', type=int, dest="hidden_1", action="store", default=512)
    arg.add_argument('--hidden_units_2', type=int, dest="hidden_2", action="store", default=256)

    inp = arg.parse_args()
    
    data_dir = str(inp.data_dir)
    gpu = inp.gpu
    save_dir = inp.save_dir
    l_rate = inp.l_rate
    dropout = inp.dropout
    epochs = inp.epochs
    arch = inp.arch
    hidden_1 = inp.hidden_1
    hidden_2 = inp.hidden_2
    
    #todo: code validation
    #instead ask user to validate
    print("")
    print("--------------------------------------------------")
    print("     Welcome to Flower Classification Trainer")
    print("--------------------------------------------------")
    print("")
    print("Confirm your settings:")
    print("data_dir =", data_dir)
    print("gpu =", gpu)
    print("save_dir =", save_dir)
    print("l_rate =", l_rate)
    print("dropout =", dropout)
    print("epochs =", epochs)
    print("arch =", arch)
    print("hidden_1 =", hidden_1)
    print("hidden_2 =", hidden_2)
    print("")
    command = input('Press y to proceed, else to cancel >> ')
    if(command!='y'):
        return
    
    print("")
    train(data_dir, gpu, save_dir, l_rate, dropout, epochs, arch, hidden_1, hidden_2)

train_settings()


    

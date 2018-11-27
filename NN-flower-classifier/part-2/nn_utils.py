'''
Author  :   Aaron Balson Caroltin .J
Purpose :   Image Classifier (Command line version) utility functions
            to be used by predict.py and train.py
For     :   Udacity AI python nanodegree - Aug-Sep 2018
'''

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from collections import OrderedDict
import json
import numpy as np
import sys

def process_image(image_path, astensor=True):
    '''
    To load the given image as PIL image from the path
    and to return it in either numpy or tensor
    '''
    if astensor==True:
        return process_image_as_tensor(image_path)
    else:
        return process_image_as_numpy(image_path)

def process_image_as_numpy(image):
    '''
    To load the given image as PIL image from the path
    and use test transforms before returning it in numpy
    '''
    test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        
    pil_image = Image.open(image)
    pil_image = test_transform(pil_image).float()
    np_image = np.array(pil_image)    
  
    return np_image

def process_image_as_tensor(image):
    '''
    To load the given image as PIL image from the path
    and use test transforms before returning it in tensor
    '''
    test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    pil_image = Image.open(image)
    tensor_image = test_transform(pil_image)
    return tensor_image

def save_checkpoint(model, criterion, optimizer, path, state, device):
    '''
    To save model, classifier, optimizer states into checkpoint file
    for future processing (train or predict) or sending to other systems
    '''
    # move model to cpu/ram for saving to disk
    model.to('cpu')
    
    #model.class_to_idx = train_dataset.class_to_idx
    
    torch.save(state, path)
    
    model.to(device)
    print("Checkpoint Saved !!")

def load_checkpoint(path, device):
    '''
    To load checkpoint file and rebuild model, criterion, optimizer
    with their states intact for further processing (train or predict)
    '''
    checkpoint = torch.load(path)
    
    model = build_model(checkpoint['model_input_units'], 
                          checkpoint['model_hidden_1'], 
                          checkpoint['model_hidden_2'], 
                          checkpoint['model_output_units'], 
                          checkpoint['model_drop_out'], 
                          checkpoint['model_class_state_dict'],
                          checkpoint['model_class_to_idx'],
                          checkpoint['model_nn_arch'])

    model.class_to_idx = checkpoint['model_class_to_idx']
    model.completed_epoch = checkpoint['model_completed_epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier.load_state_dict(checkpoint['model_class_state_dict'])
    
    model.to(device)
    
    criterion, optimizer = build_optimizer(model, 
                          checkpoint['optim_learn_rate'], 
                          checkpoint['optim_state_dict'],
                          checkpoint['optim_nn_arch'])
    
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    print("Checkpoint Loaded !!")
    return model, criterion, optimizer

def test_model(model, dataloader, device):    
    '''
    Calculates and returns the accuracy of the model against the test dataset
    '''
    model.to(device)

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            idx, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)
    print('Test Accuracy: %d %%' % accuracy)
    return accuracy

def build_model(input_units, hidden_1, hidden_2, output_units, drop_out, 
                class_state_dict, class_to_idx={}, nn_arch="vgg16"):
    '''
    Builds and returns NN model using imagenet architecture and flower classifier
    '''
    
    # densenet121's classifier has 1024 input layer units and 1000 output units for imagenet's classes
    if nn_arch=="densenet121":
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        
    # feature parameters are freezed from using autograd
    for param in model.parameters():
        param.requires_grad = False
    
    #if(len(class_to_idx)>0):
    #    model.class_to_idx = class_to_idx
        
    # new flowerclassifier relevant to our domain
    if nn_arch=="densenet121":
        flowerclassifier = nn.Sequential(OrderedDict([
                            ('dropout1',nn.Dropout(drop_out)),                 
                            ('fc1', nn.Linear(input_units, hidden_1)),
                            ('relu1', nn.ReLU()),
                            ('dropout2',nn.Dropout(drop_out)), 
                            ('fc2', nn.Linear(hidden_1, hidden_2)),
                            ('relu2', nn.ReLU()),
                            ('dropout3',nn.Dropout(drop_out)),                 
                            ('fc3', nn.Linear(hidden_2, output_units)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    else:
        flowerclassifier = nn.Sequential(OrderedDict([
                            ('dropout1',nn.Dropout(drop_out)),                 
                            ('fc1', nn.Linear(input_units, hidden_1)),
                            ('relu1', nn.ReLU()),
                            ('dropout2',nn.Dropout(drop_out)), 
                            ('fc2', nn.Linear(hidden_1, output_units)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    
    #if(len(class_state_dict)>0):
    #    flowerclassifier.load_state_dict(class_state_dict)
        
    # flowerclassifier replaces densenet121's classifier with 1024 input units and 102 output units
    model.classifier = flowerclassifier
    return model

def build_optimizer(model, learn_rate, state_dict, nn_arch="SGD"):
    '''
    Builds and returns the criterion and optimizer for training models
    '''
    # Adam uses momemtum to arrive at global/local minima at faster rate than SGD.
    if(nn_arch == "Adam"):
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    else:
        optimizer = optim.SGD(model.classifier.parameters(), lr=learn_rate)
    
    #if(len(state_dict)>0):
    #   optimizer.load_state_dict(state_dict)
    
    criterion = nn.NLLLoss()
    return criterion, optimizer

def print_exception(s, e):
    '''
    To print exception message (e) along with custom message (s)
    '''
    print(s)
    if hasattr(e, 'message'):
        print(e.message)
    else:
        print(e)


def load_data(data_dir):
    '''
    This function performs data augmentation, data normalization, data loading, data batching on the datasets, loaders
    Arguments : 
        data_dir : location from where to load data
    Returns : 
        Transforms, datasets, dataloaders for train, test, valid 
    '''
    if(data_dir == None or len(data_dir) ==0):
            raise ValueError("Invalid data_dir")
    
    try:
        
        
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        valid_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])


        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

        valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

        test_dataset = datasets.ImageFolder(test_dir ,transform = test_transform)

        #Criteria :: Data loading, Data batching
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size =32,shuffle = True)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
    
    except Exception as e:
        print_exception("ERROR::load_data", e)
        raise
    
    return train_transform, valid_transform, test_transform, train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader

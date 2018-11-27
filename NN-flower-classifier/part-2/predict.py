'''
Author  :   Aaron Balson Caroltin .J
Purpose :   Image Classifier (Command line version) to predict flower datasets
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
from time import sleep
import json
#from termcolor import colored

def predict(img_path, checkpoint, gpu, category_names, top_k=1):

    '''
    To make use of model loaded from checkpoint to perform top-K prediction
    '''

    # load cateogry names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    print("Flower categories loaded")
    
    # use GPU if available (only nVidia cuda based cards)
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu=="gpu" else "cpu")
    print("Device setup done ..", device)
    
    # load model from checkpoint
    model,_,_ = nn_utils.load_checkpoint(checkpoint, device)
    model.to(device)
    print("Model restored from checkpoint and moved to ", device)
    print("")
    
    # apply model for prediction
    print("Prediction in progress ..")
    with torch.no_grad():
        model.eval()
        predict_image = nn_utils.process_image(img_path, True).unsqueeze_(0).float()
        all_prob = torch.exp(model.forward(predict_image.to(device)))
        top_prob, top_cls = all_prob.topk(top_k)
        top_prob = top_prob[0].tolist()
        top_cls = top_cls[0].tolist()
    
        # create a linking structure between class_to_idx and cat_to_name
        # so we can return top_names instead of top_ids
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}

        top_names = [idx_to_class.get(cls,"invalid-cls") for cls in top_cls]
        top_names = [cat_to_name.get(cls,"invalid-cls") for cls in top_names]
        
    '''
    poor man's animation (every 0.5 sec)
    .
    ..
    ...
    ....
    .....
    ......
    .......
    ........
    .........
    ..........

    '''
    t=0
    p=""
    while(t<10):
        sleep(0.05)
        p+="."
        print(p)
        t+=1

    # print results
    print("")
    print(*["{} : {:.1%}".format(x,y) for x,y in zip(top_names, top_prob*100)], sep = "\n")
    print("")
    #print("Your selected flower appears to be {} with probability of {:.1%}".format(colored(top_names[0], 'red'), colored(top_prob[0], 'red')))
    print("Your selected flower appears to be {} with probability of {:.1%}".format(top_names[0], top_prob[0]))
    print("")
    done = input("Thanks for trying out. Press any key to close")        
        
def predict_settings():
    
    '''
    To collect user options for starting prediction phase
    '''
    arg = argparse.ArgumentParser(description='Predict.py')
    arg.add_argument('img_path', action="store", default="./flowers/test/1/image_06743.jpg")
    arg.add_argument('checkpoint', action="store", default="./checkpoint.pth")
    arg.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    arg.add_argument('--category_names', dest="category_names", action="store", default="cat_to_name.json")
    arg.add_argument('--top_k', dest="top_k", action="store", type=int, default=5)

    inp = arg.parse_args()
    
    img_path = str(inp.img_path)
    checkpoint = str(inp.checkpoint)
    gpu = inp.gpu
    category_names = inp.category_names
    top_k = inp.top_k
    
    #todo: code validation
    #instead ask user to validate
    print("")
    print("----------------------------------------------------")
    print("     Welcome to Flower Classification Predictor")
    print("----------------------------------------------------")
    print("")
    print("Confirm your settings:")
    print("img_path =", img_path)
    print("checkpoint =", checkpoint)
    print("gpu =", gpu)
    print("category_names =", category_names)
    print("top_k =", top_k)

    command = input('Press y to proceed, else to cancel >> ')
    if(command!='y'):
        return
    
    print("")
    predict(img_path, checkpoint, gpu, category_names, top_k)

predict_settings()

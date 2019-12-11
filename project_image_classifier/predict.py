# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json
from workspace_utils import keep_awake
from workspace_utils import active_session

# Define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--GPU', help = "Option to use GPU. Optional", type = str)

# Define function that loads a checkpoint and rebuilds the model
def loading_model(load_dir):
    model = torch.load(load_dir)
    for param in model.parameters():
        param.requires_grad = False # Turning off tuning of the model
    return model

# Define function to process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)  # Open imange
    
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]) # Transform to same as above
    
    image_transformed = transform(pil_image)
    return image_transformed


# Defining prediction function
def predict(image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    
    image = process_image(image)   # Load and process the image

    image.unsqueeze_(0)
    model.eval() 
    
    with torch.no_grad():                         # Turn off gradients
        image = image.to(device)
        logps = model(image)                      # Calculate probabilities by forward pass
        ps = torch.exp(logps)                     # Calculate probabilities using torch.exp
        top_p, top_class = ps.topk(topk, dim=1)   # Get the top k predicted classes
    
    probs_topk = np.array(top_p)[0]       # Prob of each of the top k
    index_topk = np.array(top_class)[0]   # Index numbers associated with each topk
    
    class_to_idx = model.class_to_idx     # Loading index and class mapping
    indx_to_class = {x: y for y, x in class_to_idx.items()} # Inverting index-class dictionary

    classes_topk = []
    for index in index_topk:
        classes_topk += [indx_to_class[index]]              # Converting index list to class list
        
    return probs_topk, classes_topk


# Setting values data loading
args = parser.parse_args()
file_path = args.image_dir

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# Loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Loading model from checkpoint provided
model = loading_model(args.load_dir)

# Defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

# Calculating probabilities and classes
probs_topk, classes_topk = predict(file_path, model, nm_cl, device)

# Preparing class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes_topk]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),
           "Class name: {}.. ".format(class_names [l]),
           "Probability: {:.3f}.. ".format(probs_topk [l]))
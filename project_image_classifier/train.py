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
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Model vgg11 can be used if this argument specified. Default is Alexnet', type = str)
parser.add_argument ('--lr', help = 'Learning rate, default value 0.003', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 4096', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs. Default is 5', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)


# Define data directory
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

if data_dir: # Making sure we do have value for data_dir
	# Define your transforms for the training, validation, and testing sets
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    	   transforms.RandomResizedCrop(224),
                                    	   transforms.RandomHorizontalFlip(),
                                    	   transforms.ToTensor(),
                                    	   transforms.Normalize([0.485, 0.456, 0.406],
                                            	                [0.229, 0.224, 0.225])])

	test_transforms = transforms.Compose ([transforms.Resize(255),
                                    	   transforms.CenterCrop(224),
                                    	   transforms.ToTensor(),
                                    	   transforms.Normalize([0.485, 0.456, 0.406],
                                               	                [0.229, 0.224, 0.225])])

	valid_transforms = transforms.Compose([transforms.Resize(255),
                                    	   transforms.CenterCrop(224),
                                      	   transforms.ToTensor(),
                                      	   transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	test_data  = datasets.ImageFolder(test_dir,  transform=test_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
	testloader  = torch.utils.data.DataLoader(test_data,  batch_size=64)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    # End of data loading block


# Category number to name label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def load_model (arch, hidden_units):
    if arch == 'vgg11': # Setting model based on vgg11
        model = models.vgg11 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False  # Freeze parameters - keep the features untouched
        if hidden_units: # In case hidden_units were given
            	classifier = nn.Sequential(nn.Linear(25088, 4096),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Sequentialnn.Linear(4096, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 102),
                             nn.LogSoftmax(dim=1))
        else: # If hidden_units not given
            	classifier = nn.Sequential(nn.Linear(25088, 4096),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(4096, 102),
                             nn.LogSoftmax(dim=1))
    else: # Setting model based on default Alexnet ModuleList
        arch = 'alexnet' # Will be used for checkpoint saving, so should be explicitly defined
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False  # Freeze parameters - keep the features untouched
        if hidden_units: # In case hidden_units were given
            	classifier = nn.Sequential(nn.Linear(9216, 4096),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Sequentialnn.Linear(4096, hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units, 102),
                             nn.LogSoftmax(dim=1))
        else: # If hidden_units not given
            	classifier = nn.Sequential(nn.Linear(9216, 4096),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(4096, 102),
                             nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model, arch


# Defining validation Function. will be used during training
def validation(model, validloader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    model.eval()                                        # Set model to evaluation mode
            
    with torch.no_grad():                               # Turn off gradients
        for inputs, labels in validloader:              # Validation pass here
            inputs, labels = inputs.to(device), labels.to(device) # Move these to current device - GPU or CPU
            logps = model(inputs)                       # Calculate probabilities by forward pass
            
            batch_loss = criterion(logps, labels)       # Calculate loss with output    
            valid_loss += batch_loss.item()              
                    
            # Calculate accuracy 
            ps = torch.exp(logps)                       # Calculate probabilities using torch.exp
            top_p, top_class = ps.topk(1, dim=1)        # Get the predicted classes
            equals = top_class == labels.view(*top_class.shape)             # Predicted classes = true classes?
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()   # Measure the accuracy
    return valid_loss, accuracy


# Loading model using above defined functiion
model, arch = load_model(args.arch, args.hidden_units)


## Training of the model ##


# Initializing criterion and optimizer
# Define loss 
criterion = nn.NLLLoss()                                        

if args.lr: # If learning rate was provided
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003) # Only train classifier parameters, not features

# Device can be either cuda or cpu
model.to(device)

# Setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:
    epochs = 5

# Running through epochs

# valid_loss = 0
print_every = 40
steps = 0
running_loss = 0

for epoch in range(epochs):
    
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)   # Move input and label tensors to the default device
        
        optimizer.zero_grad()            # Clear the gradients, do this because gradients are accumulated       
        logps = model(inputs)            # Calculate probabilities by forward pass
        loss = criterion(logps, labels)  # Calculate loss with output
        loss.backward()                  # Calculate the gradients by backward pass through
        optimizer.step()                 # Update the model - in this case, only classifier
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval () # Switching to evaluation mode so that dropout is turned off
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
                    
    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
          "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
          "Validation Accuracy: {:.3f}.. ".format(accuracy/len(validloader)))

# Saving trained Model
model.to('cpu') # No need to use cuda for saving/loading model.

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx  # Add class to index

# Saving trained model for future use
if args.save_dir:
	torch.save(model, args.save_dir+'flowers_whole.pth')
else:
    torch.save(model, 'flowers_whole.pth')
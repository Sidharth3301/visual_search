import numpy as np
import torch.nn.functional as F
import cv2
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.models as models
import torch
import torch.nn as nn


TotalTrials = 600
targetsize = 28
stimulisize = 224


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            # compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            # simple module
            try:
                n = int(n)
                model[n] = new
            except:
                setattr(model, n, new)


transform_target = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((targetsize, targetsize)), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.ConvertImageDtype(torch.float)]
)

transform_source = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((stimulisize, stimulisize)), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.ConvertImageDtype(torch.float)]
)

# mean = torch.FloatTensor([103.939, 116.779, 123.68]).view(3, 1, 1)


def preprocess(img, name="src"):
    if name == "src":
        size = transform_source(img).size()
        # - mean.expand(size),
        return torch.unsqueeze(transform_source(img)*256, dim=0)
    else:
        size = transform_target(img).size()
        # - mean.expand(size),dim=0)
        return torch.unsqueeze(transform_target(img)*256, dim=0)


model_stim = models.vgg16(pretrained=True).features[:-1]
model_target = models.vgg16(pretrained=True).features
replace_layers(
    model_stim,
    nn.MaxPool2d,
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=True),
)

replace_layers(
    model_target,
    nn.MaxPool2d,
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                 dilation=1, ceil_mode=True),
)
folder_stimuli = "stimuli/"
folder_target = "target/"

# comment out these lines for CPU runtimes
model_stim.cuda()
model_target.cuda()

model_stim.eval()
model_target.eval()
stack = dict()

target_ims = dict()
for i in range(len(os.listdir('target'))//2):
    target_ims[i] = model_target(preprocess(cv2.imread(
        folder_target + f"target_{i+1}.jpg"), 'tar').cuda())

stim_ims = dict()
for i in range(len(os.listdir('stimuli'))):
    k = cv2.imread(folder_stimuli + f"array_{i+1}.jpg")
    im_tens = preprocess(k, 'src')
    stim_ims[i] = im_tens

for i in range(len(os.listdir('stimuli'))):
    with torch.no_grad():
        stim_ims[i] = model_stim(stim_ims[i].cuda())

for idx in range(len(os.listdir('stimuli'))):

    stack[idx] = dict()
    for i, j in target_ims.items():

        out_tens = F.conv2d(
            stim_ims[idx], j, None, (1, 1), (1, 1)
        )
        out_tens = torch.squeeze(out_tens, dim=0)
        stack[idx][i] = []
        stack[idx][i].append(out_tens.cpu().detach().numpy())
    if idx % 50 == 0:
        print(f'Done with {idx} stim ims')


pickle.dump(stack, open('./att_maps.pkl', 'wb'))

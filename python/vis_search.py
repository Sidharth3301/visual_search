import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import cv2
from scipy.io import savemat
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

TotalTrials = 600
targetsize = 28
stimulisize = 224
model = models.vgg16(pretrained=True)
folder = "choopednaturaldesign"
file = "croppednaturaldesign_img.txt"
f = open(file, "r", newline="\n")
locs = f.readlines()
imagename_target = "sampleimg/targetgray004.jpg"
target = cv2.imread(imagename_target)
# source = []
# for i in range(len(list_ims)):
#     imagename_stimuli = folder + list_ims[i]

#     source[i] = cv2.imread(imagename_stimuli)

# source = cv2.resize(source, (stimulisize, stimulisize))
# target = cv2.resize(target, (targetsize, targetsize))
transform_target = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((targetsize, targetsize))]
)

transform_source = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((stimulisize, stimulisize))]
)

mean = torch.FloatTensor([103.939, 116.779, 123.68]).view(3, 1, 1)


def preprocess(img, name="src"):
    if name == "src":
        size = transform_source(img).size()
        return transform_source(img) * 256 - mean.expand(size)
    else:
        size = transform_target(img).size()
        return torch.unsqueeze((transform_target(img) * 256 - mean.expand(size)) * 1, 0)


targ_tens = preprocess(target, "tar")


class vis_dataset(Dataset):
    def __init__(self, file, transform) -> None:
        super().__init__()
        f = open(file, "r", newline="\n")
        self.locs = f.readlines()

        self.transform = transform

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        im = cv2.imread(self.locs[idx].strip())
        if self.transform:
            im = self.transform(im)
        return im


source = vis_dataset(file, preprocess)
batch_size = 30
batch_source = DataLoader(source, batch_size=batch_size, shuffle=True, num_workers=0)

m2 = model.features[:-1]

m2.eval()
targ_stim = m2(targ_tens)

for i_batch, sample_batched in enumerate(batch_source):
    out_tens = F.conv2d(m2(sample_batched), targ_stim, None, (1, 1), (1, 1))
    out_tens.permute(0, 2, 3, 1)

for i, j in enumerate(out_tens.detach().numpy()):
    mdic = {"x": j, "label": "exper"}
    s = locs[i].split("/")[1].split(".")[0].strip()
    savemat("python/results/" + s + ".mat", mdic)

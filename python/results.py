import pickle
import cv2
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import pylab as plt


def gray(img):
    img = np.array(img)
    return img/img.max().max()


gtpos = pickle.load(open('gtpos.pkl', 'rb'))
f = pickle.load(open('att_maps.pkl', 'rb'))

num_cates = 6
num_ims = 300

x = loadmat('./saliencyMask/maskind.mat')
total_mask = x['maskind']
masks = dict()
for i in range(1, 7):

    masks[i] = loadmat(f'./eval/saliencyMask/mask{i}.mat')['mask']

attmaps = dict()
for i in range(num_ims):

    attmaps[i] = gray(cv2.resize(np.squeeze(f[i][i][0]), total_mask.T.shape))

posx = dict()
posy = dict()
for i in range(num_ims):
    posx[i] = []
    posy[i] = []

scorematrix = np.zeros(num_ims)
for i in range(num_ims):
    if i % 100 == 0:
        print(f'done with {i}')
    att_map = attmaps[i]
    for j in range(num_cates):
        if j == 0:
            salimg = np.multiply(att_map, total_mask)
        else:
            chosenmask = masks[chosenfix]
            temp = np.multiply(salimg, (1-chosenmask))
            salimg = gray(temp)

        indx, indy = np.unravel_index(np.argmax(salimg), shape=salimg.shape)

        chosenfix = total_mask[indx, indy]
        posx[i].append(indx)
        posy[i].append(indy)

        scorematrix[i] += 1
        if gtpos[i % 6][i]+1 == chosenfix:
            break

plt.hist(scorematrix, density=True)

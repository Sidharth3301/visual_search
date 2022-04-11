import pandas as pd
from mat4py import loadmat
import numpy as np
import pickle
d1 = loadmat('array.mat', 'r')

d1.keys()
df1 = pd.DataFrame.from_dict(d1['MyData'])
pos = dict()

cates = [16, 18, 19, 20, 34, 78]
for i in range(6):
    pos[i] = []
    for j in range(300):
        b, _ = np.where(cates[i] == np.array(df1['arraycate'].iloc[j]))
        pos[i].append(b)
pickle.dump(pos, open('gtpos.pkl', 'wb'))

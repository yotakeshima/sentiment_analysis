import re
from tkinter import N
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import string

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2)) 
kf = KFold(n_splits=10)
result = next(kf.split(df), None)
train = df.iloc[result[0]]
test = df.iloc[result[1]]

#print(test)
#print(train)

for train_idx, test_idx in kf.split(df):
    print(df.iloc[train_idx])

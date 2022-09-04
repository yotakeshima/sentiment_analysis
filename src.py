import re
from tkinter import N
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import string

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
kf = KFold(n_splits=10)                     # initialize kf variable, splits into k=10 folds
result = next(kf.split(df), None)           # result variable takes just one of the Kfold splits.
train = df.iloc[result[0]]                  
test = df.iloc[result[1]]

# print(test)
# print(train)

for train_idx, test_idx in kf.split(df):
    print(df.iloc[test_idx]['Reviews'])

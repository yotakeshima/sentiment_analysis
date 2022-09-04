import re
import numpy as np
import pandas as pd
import string

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2)) 
df.head()
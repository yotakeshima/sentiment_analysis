import re
import numpy as np
import pandas as pd

f = open("trainhw1new.txt","r")
reviews = f.readlines()[1:10]
for line in reviews:
    text = re.split(r"\t", line)
    print(text[1] + '\n')
    ratings = text[0]
    comments = text[1]
    ratings = ratings.replace(" ", "")
    ratings = ratings.replace(".", "")
print(reviews)
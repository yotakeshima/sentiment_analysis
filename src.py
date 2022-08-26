import re
import numpy as np

f = open("trainhw1new.txt","r")
reviews = f.readlines()[1:10]
for line in reviews:
    text = re.split(r"\t\w", line)
    print(text[0])
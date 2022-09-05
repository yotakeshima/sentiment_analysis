

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from scipy import spatial
import pandas as pd
import nltk
import re
import string
import scipy.sparse

 
def preprocess(text):
    text.loc[:,"Reviews"] = text.Reviews.apply(lambda x : str.lower(x))
    text.loc[:,"Reviews"] = text.Reviews.apply(lambda x : " ".join(re.findall('[\w]+',x)))
    return text
def testPrep(text):
    text.iloc[1] = text.iloc[1].lower()
    text.iloc[1] = text.iloc[1].translate(str.maketrans('','', string.punctuation))
    return text

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
test = df.iloc[0]
train = df.iloc[1:10]

train = preprocess(train)
test = testPrep(test)

similar = []



# train.loc[:,"Reviews"] = train.Reviews.apply(lambda x : str.lower(x))
# train.loc[:,"Reviews"] = train.Reviews.apply(lambda x : " ".join(re.findall('[\w]+',x)))

# test = "I love being pummeled by ugly people"
# train = "I love being killed by ugly people"


# idf = TfidfVectorizer()
# idf_matrix = idf.fit_transform(matrix)

# cv = CountVectorizer(stop_words='english')
# cv_matrix = cv.fit_transform(matrix)


# X = scipy.sparse.hstack([cv_matrix, idf_matrix])
# X = X.toarray()

# result = spatial.distance.cosine(X[0], X[1])
# print(result)

# def knn(test, train, k):
    
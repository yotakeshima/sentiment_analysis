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
    text.loc[:,"Labels"] = text.Labels.apply(lambda x : "".join(re.findall('[\d]|\-',x)))
    return text
def testPrep(text):
    text.iloc[1] = text.iloc[1].lower()
    text.iloc[1] = text.iloc[1].translate(str.maketrans('','', string.punctuation))
    return text

def vectorize(data, test):
    idf = TfidfVectorizer()
    idf_matrix = idf.fit_transform([data.iloc[0]['Reviews']])
    cv = CountVectorizer(stop_words='english')
    cv_matrix = cv.fit_transform([data.iloc[0]['Reviews']])
    train_vector = scipy.sparse.hstack([cv_matrix, idf_matrix])
    train_vector = train_vector.toarray()

    test_idf_matrix = idf.fit_transform([test.Reviews])
    test_cv_matrix = cv.fit_transform([test.Reviews])
    test_vector = scipy.sparse.hstack([test_cv_matrix, test_idf_matrix])
    test_vector = test_vector.toarray()
    
    # train_vector[0] = spatial.distance.cosine(train_vector[0], test_vector[0])
    print(test_vector[0])
    print(train_vector[0])
  




df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
test = df.iloc[1]
train = df.iloc[1:10]

train = preprocess(train)
test = testPrep(test)
new_train = train[['Labels','Reviews']].copy()


print(train)

vectorize(new_train, test)



# train.loc[:,"Reviews"] = train.Reviews.apply(lambda x : str.lower(x))
# train.loc[:,"Reviews"] = train.Reviews.apply(lambda x : " ".join(re.findall('[\w]+',x)))


# idf = TfidfVectorizer()
# idf_matrix = idf.fit_transform([new_train.iloc[0]['Reviews'],new_train.iloc[1]['Reviews']])


# cv = CountVectorizer(stop_words='english')
# cv_matrix = cv.fit_transform([new_train.iloc[0]['Reviews'],new_train.iloc[1]['Reviews']])


# X = scipy.sparse.hstack([cv_matrix, idf_matrix])
# X = X.toarray()
# print(X)

# result = spatial.distance.cosine(X[0], X[1])
# print(result)

# def knn(test, train, k):
    
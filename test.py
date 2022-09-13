from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from scipy import spatial
import pandas as pd
import numpy as np
import nltk
import re


nltk.download('omw-1.4')
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text['Reviews'] = text['Reviews'].apply(str.lower)
    text['Reviews'] = text['Reviews'].apply(lambda x: re.sub('[0-9]+', '', x))
    text['Reviews'] = text['Reviews'].str.replace('-', ' ')
    text['Reviews'] = text['Reviews'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    text['Reviews'] = text['Reviews'].apply(lambda x: ' '.join(w for w in x.split() if w not in stopwords))
    text['Lemmatized'] = text.Reviews.apply(lemmatize_text)
    text['Reviews'] = text['Lemmatized'].apply(lambda x: ' '.join(x))
    text['Labels'] = text['Labels'].str.replace(' ', '')
    text['Labels'] = text['Labels'].apply(lambda x: re.sub(r'[/.]', '', x))
    return text

def vectorize(data):
    idf = TfidfVectorizer()
    idf_matrix = idf.fit_transform(data['Reviews'])
    idf_matrix = idf_matrix.toarray()
    return idf_matrix

def knn(test, train, k):
    newArray = []
    result = 0
    for x in train['Vectors']:
        newArray.append(1 - spatial.distance.cosine(test, x))
    train['Cosine'] = newArray
    temp = train.nlargest(n=k, columns=['Cosine'])
    for i in range(0, len(temp)):
        result += int(temp.iloc[i]['Labels'])
    return result

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', usecols=range(2), engine='python') # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
train = df.iloc[0:1800].copy()
nltk.download('stopwords')
stopwords = stopwords.words('english')

train = train.replace(to_replace='None', value=np.nan).dropna()
train = preprocess(train)
vectors = vectorize(train)
train['Vectors'] = vectors.tolist()

accuracy = 0
for i in range(0, 100):
    result = knn(train.iloc[i]['Vectors'], train, 10)
    print(str(i+1) + ': ', end='')
    if(result >= 0):
        if(int(train.iloc[i]['Labels']) == 1):
            accuracy +=1
    else: 
        if(int(train.iloc[i]['Labels']) == -1):
            accuracy += 1
    print(accuracy) 

accuracy = (accuracy/len(100)) * 100

print('Accuracy for the test is: ' + str(accuracy) + '%')

    

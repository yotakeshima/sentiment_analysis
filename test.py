from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
import re

nltk.download('omw-1.4')
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stopwords = stopwords.words('english')

#Cleans noise from the text data. Removes punctuation, numbers, stopwords. Converts all strings to lower case, and lemmatizes the reviews
def preprocess(data):
    data['Reviews'] = data['Reviews'].apply(str.lower)
    data['Reviews'] = data['Reviews'].apply(lambda x: re.sub('[0-9]+', '', x))
    data['Reviews'] = data['Reviews'].str.replace('-', ' ')
    data['Reviews'] = data['Reviews'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data['Reviews'] = data['Reviews'].apply(lambda x: ' '.join(w for w in x.split() if w not in stopwords))
    data['Lemmatized'] = data.Reviews.apply(lemmatize_text)
    data['Reviews'] = data['Lemmatized'].apply(lambda x: ' '.join(x))
    data['Labels'] = data['Labels'].str.replace(' ', '')
    data['Labels'] = data['Labels'].apply(lambda x: re.sub(r'[/.]', '', x))
    data['Labels'] = data['Labels'].astype(np.int64)
    return data

#Converts the 
def vectorize(data):
    idf = TfidfVectorizer()
    idf_matrix = idf.fit_transform(data['Reviews'])
    idf_matrix = idf_matrix.todense()
    test = pd.DataFrame(idf_matrix, columns=idf.get_feature_names_out())
    return test

def knn(test, train, k):
    f  = open('output.txt', 'w')
    m_df = pd.DataFrame(cosine_similarity(train, test, dense_output=True))
    ndf = pd.concat([m_df,df_test], axis=1)
    for i in range(0, test.shape[0]):
        result = 0
        temp = ndf.nlargest(n=k, columns=i)
        for j in range(0, k):
            result += int(temp.iloc[j]['Labels'])
        if(result >= 0):
            f.write("+1\n")    
        else:
            f.write("-1\n")
        print(".",end="")
    f.close
    
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

train = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', engine='python', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
test = open('testdatahw1.txt', "r")
test = test.read().split("\n")
test = pd.DataFrame(test, columns=['Reviews'])
train_size = train.shape[0]
test_size = test.shape[0]               
labels = pd.DataFrame('0', index = np.arange(test_size), columns=['Labels'])
test = pd.concat([labels,test], axis=1)
train = train.replace(to_replace=np.nan, value="1")
train = preprocess(train)
test = preprocess(test)
df = pd.concat([test,train], axis=0)
print("Finished Preprocessing...")
m_df = vectorize(df)
df_test = df.iloc[train_size:,:]
test = m_df.iloc[:test_size,:]
train = m_df.iloc[train_size:,:]
knn(test, train, 21)

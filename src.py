from cmath import nan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
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
    idf_matrix = idf_matrix.todense()
    test = pd.DataFrame(idf_matrix, columns=idf.get_feature_names_out())
    return test

def knn(test, train, k, df, index):
    accuracy = 0
    ndf = df.copy()
    m_df = pd.DataFrame(cosine_similarity(train, test, dense_output=True))
    ndf = pd.concat([m_df,ndf], axis=1)
    print(test.shape[0])
    for i in range(0, test.shape[0]):
        result = 0
        temp = ndf.nlargest(n=k, columns=i)
        for j in range(0, k):
            result += int(temp.iloc[j]['Labels'])
        if(result >= 0 & int(df.iloc[index[i]]['Labels']) == 1):
            accuracy += 1
        else:
            if(int(df.iloc[index[i]]['Labels']) == -1): 
                accuracy +=1
        print(result, end=" ")  
    print()
    print("Accuracy for this set: ", accuracy/test.shape[0] * 100)
    print("DONE")

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', engine='python', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
df = df.iloc[0:5000].copy()
kf = KFold(n_splits=10)                     # initialize kf variable, splits into k=10 folds

df = df.replace(to_replace=np.nan, value="1")
df = preprocess(df)
<<<<<<< HEAD
cv = CountVectorizer()
matrix = cv.fit_transform(df['Reviews'])
matrix = matrix.todense()
m_df = pd.DataFrame(matrix, colums=cv.get_feature_names_out())
=======
print("Finished Preprocessing...")
m_df = vectorize(df)
# cv = CountVectorizer()
# matrix = cv.fit_transform(df['Reviews'])
# matrix = matrix.todense()
# m_df = pd.DataFrame(matrix, columns=cv.get_feature_names_out())
test = m_df.loc[2].copy()

# print(test)
# print(train)
dj = pd.DataFrame(cosine_similarity(m_df, dense_output=True))
>>>>>>> 745ff764d5efb8e9607216862edfdfa0b8117a15

ndf = pd.concat([dj,df], axis=1)
# temp = ndf.nlargest(n=11, columns=0)
# result = 0
# for i in range(1,11):
#     result += int(temp.iloc[i]['Labels'])
# print(result)
accuracy = 0
for i in range(0,500):
    result = 0
    temp = ndf.nlargest(n=11, columns=0)
    for j in range(1, 11):
        result += int(temp.iloc[j]['Labels'])
        if(result >= 0 & int(df.iloc[i]['Labels']) == 1):
            accuracy += 1
        else:
            if(int(df.iloc[i]['Labels']) == -1): 
                accuracy +=1
        print(result, end=" ")  
print()
print("Accuracy for this set: ", accuracy/500 * 100)
print("DONE")

# for train_idx, test_idx in kf.split(m_df):
#     knn(m_df.loc[test_idx], m_df.loc[train_idx], 9, df, test_idx)


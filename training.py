from sklearn.feature_extraction.text import TfidfVectorizer
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

def vectorize(data):
    idf = TfidfVectorizer()
    idf_matrix = idf.fit_transform(data['Reviews'])
    idf_matrix = idf_matrix.todense()
    test = pd.DataFrame(idf_matrix, columns=idf.get_feature_names_out())
    return test

def knn(test, train, k, df, index, labels):
    accuracy = 0
    ndf = df.copy()
    m_df = pd.DataFrame(cosine_similarity(train, test, dense_output=True))
    ndf = pd.concat([m_df,ndf], axis=1)
    
    for i in range(0, test.shape[0]):
        result = 0
        temp = ndf.nlargest(n=k, columns=i)
        for j in range(0, k):
            result += int(temp.iloc[j]['Labels'])
        if(result >= 0):
            labels.append(1)
            if (int(df.iloc[index[i]]['Labels']) == 1):
                accuracy += 1
                
        else:
            labels.append(-1)
            if(int(df.iloc[index[i]]['Labels']) == -1): 
                accuracy += 1
        print(".",end="")
    
        

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', engine='python', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 

size = df.shape[0]
kf = KFold(n_splits=10)                    

df = df.replace(to_replace=np.nan, value="1")
df = preprocess(df)
print("Finished Preprocessing...")
m_df = vectorize(df)

for x in range(20,30):
    newLabels = []
    for train_idx, test_idx in kf.split(m_df):
        knn(m_df.loc[test_idx], m_df.loc[train_idx], x, df, test_idx, newLabels)
    df['NewLabels'] = newLabels
    m_acc = pd.concat([df['NewLabels'],df['Labels']], axis=1)
    accuracy = 0
    for i in range(0,size):
        if(m_acc.iloc[i]['Labels'] == m_acc.iloc[i]['NewLabels']): accuracy +=1
    accuracy = accuracy/size * 100
    print("Accuracy for k as ", x, " is: ", accuracy, "%")
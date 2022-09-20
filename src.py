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

def knn(test, train, k):
    result = 0
    newArray = pd.DataFrame(cosine_similarity(test, dense_output=True))
    print(newArray)
    # train['Cosine'] = train['Vectors'].apply(lambda x: 1 - spatial.distance.cosine(x, test))
    temp = train.nlargest(n=k, columns=['Cosine'])
    for i in range(0, len(temp)):
        result += int(temp.iloc[i]['Labels'])
    return result

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', engine='python', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews' 
kf = KFold(n_splits=10)                     # initialize kf variable, splits into k=10 folds

df = df.replace(to_replace='None', value=np.nan).dropna()
df = preprocess(df)
cv = CountVectorizer()
matrix = cv.fit_transform(df['Reviews'])
matrix = matrix.todense()
m_df = pd.DataFrame(matrix, colums=cv.get_feature_names_out())

for train_idx, test_idx in kf.split(df):
    print(df.iloc[train_idx]['Reviews'])


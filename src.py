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

# Cleans noise from the input text data. Removes punctuation, numbers, stopwords. Converts all strings to lower case, and lemmatizes the reviews
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

# Converts text data to BOW and TFIDF.
# Parameters: data - input text data.
def tfidf(data):
    idf = TfidfVectorizer()
    idf_matrix = idf.fit_transform(data['Reviews'])
    idf_matrix = idf_matrix.todense()
    test = pd.DataFrame(idf_matrix, columns=idf.get_feature_names_out())
    return test

# KNN algorithm and writes an output .txt file that contains the sentiment scores.
# Parameters:   v_test - test data vectors
#               v_train - training data vectors
#               k - 'k' value of KNN
def knn(v_test, v_train, k):
    f  = open('output.txt', 'w')
    m_df = pd.DataFrame(cosine_similarity(v_train, v_test, dense_output=True))
    ndf = pd.concat([m_df,df_test], axis=1)
    for i in range(0, v_test.shape[0]):
        result = 0
        temp = ndf.nlargest(n=k, columns=i)
        for j in range(0, k):
            result += int(temp.iloc[j]['Labels'])
        if(result >= 0):
            f.write("+1\n")    
        else:
            f.write("-1\n")
        # Indicates how many reviews have been predicted thus far.
        print(".",end="")
    f.close

# KNN algorithm specifically used for cross-validation. Adds an extra parameter to keep track
# and compare the 'Labels' of the training data to the predicted 'NewLabels'.
# Parameters:   v_test - test data vectors
#               v_train - training data vectors
#               k - 'k' value of KNN
def knn_train(v_test, v_train, k, index):
    accuracy = 0
    ndf = train.copy()
    matrix = pd.DataFrame(cosine_similarity(v_train, v_test, dense_output=True))
    ndf = pd.concat([matrix, ndf], axis=1)
    for i in range(0, v_test.shape[0]):
        result = 0
        temp = ndf.nlargest(n=k, columns=i)
        for j in range(0, k):
            result += int(temp.iloc[j]['Labels'])
        if(result >= 0):
            newLabels.append(1)
            if (int(train.iloc[index[i]]['Labels']) == 1):
                accuracy += 1               
        else:
            newLabels.append(-1)
            if(int(train.iloc[index[i]]['Labels']) == -1): 
                accuracy += 1

# Cross-validation function that uses a range of 'k' values to determine the most accurate k for KNN function.
# Parameters:   x - minimum starting 'k' value
#               y - maximum ending 'k' value 
def cross_valid(x, y):
    k_array = []
    for acc in range(x,y):
        for train_idx, test_idx in kf.split(train_vectors):
            knn_train(train_vectors.loc[test_idx], train_vectors.loc[train_idx], acc, test_idx)
        train['NewLabels'] = newLabels
        newLabels.clear()
        m_acc = pd.concat([train['NewLabels'],train['Labels']], axis=1)
        accuracy = 0
        for i in range(0,train_size):
            if(m_acc.iloc[i]['Labels'] == m_acc.iloc[i]['NewLabels']): accuracy +=1
        accuracy = accuracy/train_size * 100
        print("Accuracy for k as ", acc, " is: ", accuracy, "%")
        k_array.append(accuracy)
    return max(k_array)

# Function to lemmatize text.
# Parameters:   text - string 
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# Function to convert a txt. file into a DataFrame.
# Parameters:   data - txt. file to be converted
def read_test(data):
    test = data.read().split("\n")
    test = pd.DataFrame(test, columns=['Reviews'])
    test_size = test.shape[0]               
    labels = pd.DataFrame('0', index = np.arange(test_size), columns=['Labels'])
    test = pd.concat([labels,test], axis=1)
    return test

#Start of Program
train = pd.read_csv('trainhw1new.txt', names=['Labels', 'Reviews'], sep='(?<=\d)\t|(?<=.)\t', engine='python', usecols=range(2)) # Create a dataframe from trainhw1new.txt file. Then split the data into 2 columns by 'Labels' | 'Reviews'
test = open('testdatahw1.txt', "r")
test = read_test(test)
train_size = train.shape[0]
test_size = test.shape[0]
kf = KFold(n_splits=10)                    
train = train.replace(to_replace=np.nan, value="1")
train = preprocess(train)
test = preprocess(test)

# Indicates that the preprocessing is done.
print("Finished Preprocessing...")

df = pd.concat([test,train], axis=0)
train_vectors = tfidf(train)
newLabels = []
k = cross_valid(20,30)
test_vectors = tfidf(df)
df_test = df.iloc[train_size:,:]
test = test_vectors.iloc[:test_size,:]
train = test_vectors.iloc[train_size:,:]
knn(test, train, k)


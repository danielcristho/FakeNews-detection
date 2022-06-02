from unittest import skip
import pandas as pd
import csv
import numpy as np
import nltk 
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

testFilename = 'data/test.csv'
trainFilename = 'data/train.csv'
validFilename = 'data/valid.csv'

train_news = pd.read_csv(trainFilename)
test_news = pd.read_csv(testFilename)
valid_news = pd.read_csv(validFilename)

#data observation
def data_obs():
    print("training dataset size:")
    print(train_news.shape)
    print(train_news.head(10))

    #below dataset were used for testing and validation purposes
    print(test_news.shape)
    print(test_news.head(10))

    print(valid_news.shape)
    print(valid_news.head(10))

#distribution of classes for prediction
def create_distribution(dataFile):
    return sb.countplot(x='Label', data=dataFile, palette='his')


#by calling bellow we can see that training, test and valid data seems to be failry
create_distribution(train_news)
create_distribution(test_news)
create_distribution(valid_news)

#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    print("Checking data qualitites...")
    train_news.isnull().sum()
    train_news.info()

    print("check finished.")

    test_news.isnull().sum()
    test_news.info()

    valid_news.isnull().sum()
    valid_news.info()

#Steeming
def stemTokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#process the data
def processData(data, excludeStopword=True, stem=True):
    tokens = [w.lower() for w  in data]
    tokensStemmed = tokens
    tokensStemmed = stemTokens(tokens, engStemmer)
    tokensStemmed = [w for w in tokensStemmed if w not in stopwords]
    return tokensStemmed

#creating ngrams
#unigram
def createUnigram(words):
    assert type(words) == list
    return words

#bigram
def createBigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len-1):
            for k in range(1, skip+2):
                if i+k < Len:
                    lst.append(join_str.join([words[i], words[i+k]]))

    else:
        #set it as unigram
        lst = createUnigram(words)
    return lst

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizerPorter(text):
    return [porter.stem(word) for word in text.split()]
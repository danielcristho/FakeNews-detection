import Prepocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
nltk.download('treebank')
#creating feature vector - document term matrix
countV = CountVectorizer()
trainCount = countV.fit_transform(Prepocessing.train_news['Statement'].values)

print(countV)
print(trainCount)

#print training doc term matrix
#we have matrix of size of (10240, 12196) by calling below
def get_countVectorizer_stats():
    #vocab size
    trainCount.shape

    #check vocabulary using below command
    print(countV.vocabulary_)

    #get feature names
    print(countV.get_feature_names()[:25])

#create tf-df frequency features
#tf-idf
tfidV = TfidfTransformer()
train_tfidf = tfidV.fit_transform(trainCount)

def get_tfidf_stats():
    train_tfidf.shape
    #get train data feature names
    print(train_tfidf.A[:10])


#bag of words - with n-grams
countV_ngram = CountVectorizer(ngram_range=(1,3),stop_words='english')
tfidf_ngram  = TfidfTransformer(use_idf=True,smooth_idf=True)

tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1,4), use_idf=True, smooth_idf=True)

#POS Tagging
taggingSentences = nltk.corpus.treebank.tagged_sents()
cutoff = int(.75 * len(taggingSentences))
trainingSentences = Prepocessing.train_news['Statement']
print(trainingSentences)

#Training POS tagger based on words
def features(sentence, index):
    """ Sentence: [w1,w2, ...], index: the index of the word"""
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) -1 else sentence[index + 1],
        'has_hypen': '-' in sentence[index],
        'ls_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }   

#helper function to strip tags from tagged corpus
def untag(taggingSentences):
    return [w for w, t in taggingSentences]

#using Word2Vec
with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
        for line in lines}

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                or [np.zeros(self.dim)], axis=0)
                for words in X
        ])        
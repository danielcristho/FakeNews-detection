# FakeNews-detection using Tensorflow&Twilio

### Kelompok 8

| Nama                      | NRP           |
|---------------------------|---------------|
|Gloriyano C. Daniel Pepuho |5025201121     |
|Khuria Khusna              |5025201053     |

## NB: dataset yg digunakan: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

### Processing.py

File preprocessing.py berisi semua fungsi preprocessing yang diperlukan untuk memproses semua dokumen dan teks masukan. Pertama kita membaca file data kereta, pengujian dan validasi kemudian melakukan beberapa preprocessing seperti tokenizing, stemming dll. Ada beberapa analisis data eksplorasi yang dilakukan seperti distribusi variabel respon dan pemeriksaan kualitas data seperti nilai nol atau hilang dll.

```python
#Stemming
def stem_tokens(tokens, stemmer):
   stemmed = []
   for token in tokens:
       stemmed.append(stemmer.stem(token))
   return stemmed

#process the data
def process_data(data,exclude_stopword=True,stem=True):
   tokens = [w.lower() for w in data]
   tokens_stemmed = tokens
   tokens_stemmed = stem_tokens(tokens, eng_stemmer)
   tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
   return tokens_stemmed


#creating ngrams
#unigram
def create_unigram(words):
   assert type(words) == list
   return words

#bigram
def create_bigrams(words):
   assert type(words) == list
   skip = 0
   join_str = " "
   Len = len(words)
   if Len > 1:
       lst = []
       for i in range(Len-1):
           for k in range(1,skip+2):
               if i+k < Len:
                   lst.append(join_str.join([words[i],words[i+k]]))
   else:
       #set it as unigram
       lst = create_unigram(words)
   return lst
```

### FeatureSelection.py

```python
def features(sentence, index):
   """ sentence: [w1, w2, ...], index: the index of the word """
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
       'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
       'has_hyphen': '-' in sentence[index],
       'is_numeric': sentence[index].isdigit(),
       'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
}
```

### Clasificaton.py

```python
n-grams & tfidf confusion matrix and F1 scores

#Naive bayes
 [841 3647]
 [427 5325]
 f1-Score: 0.723262051071

#Logistic regression
 [1617 2871]
 [1097 4655]
 f1-Score: 0.70113000531

#svm
 [2016 2472]
 [1524 4228]
 f1-Score: 0.67909201429

#sgdclassifier
 [  10 4478]
 [  13 5739]
 f1-Score: 0.718731637053

#random forest
 [1979 2509]
 [1630 4122]
 f1-Score: 0.665720333284
```

### Prediction.py
Pengklasifikasi kami yang terakhir dipilih adalah  Logistic Regression yang kemudian disimpan di disk dengan nama final_model.sav. Setelah Anda menutup repositori ini, model ini akan disalin ke mesin pengguna dan akan digunakan oleh file predict.py untuk mengklasifikasikan berita palsu. Dibutuhkan artikel berita sebagai masukan dari pengguna kemudian model digunakan untuk keluaran klasifikasi akhir yang ditampilkan kepada pengguna beserta probabilitas kebenarannya.

```python
import pickle

#function to run for prediction
def detectingFakeNews(var):
    #retrieving the best model for prediction call
    loadModel = pickle.load(open('model/final_model.sav', 'rb'))
    prediction = loadModel.predict([var])
    prob = loadModel.predict_proba([var])

    return prediction, prob
```
import os
from textblob import TextBlob
import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gensim import matutils, models
import scipy.sparse as sparse

#read data from files
def read_file(file):
  with open('/tmp/' + file, 'r') as f:
    data = f.read()
  return data

#clean out address lines from emails
def clean_addresses(df):
  for j in range(0, len(df)):
    string1 = df.iloc[j]['text_data']
    for i in range(0, len(string1)):
      if string1[i] == '>':
        df.iloc[j]['text_data'] = string1[i+1:]
        clean_addresses(df)

#remove various characters & excess spaces
def clean_chars(text):
  text = text.lower()
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = re.sub('\n', ' ', text)
  text = re.sub('  ', ' ', text)
  return text

#create the final corpus
def load_corpus(files):
  texts = map(read_file, files)
  texts_df = pd.DataFrame(list(texts), columns=["text_data"])
  cleaned_add = clean_addresses(texts_df)
  clean_func = lambda x: clean_chars(x)
  return(pd.DataFrame(texts_df.text_data.apply(clean_func)))

#create a document term matrix from a corpus
def create_dtm(corpus):
  cv = CountVectorizer(stop_words='english')
  cv_data = cv.fit_transform(corpus.text_data)
  dtm = pd.DataFrame(cv_data.toarray(), columns=cv.get_feature_names())
  return dtm

#create a new corpus in the right format for topic modeling
def create_new_corp(dtm):
  sparse_matrix = sparse.csr_matrix(dtm) #transpose?
  corpus = matutils.Sparse2Corpus(sparse_matrix)
  return corpus

#create a dictionary to map the corpus to the dtm
def create_dict(corpus, dtm):
  new_cv = CountVectorizer(stop_words='english')
  new_cv.fit(list(dtm))
  dict_words = dict((v,k) for k, v in new_cv.vocabulary_.items())
  return dict_words

#begin loading files & process data
all_files = os.listdir('/tmp')
corpus = load_corpus(all_files)
dtm = create_dtm(corpus)
new_corpus = create_new_corp(dtm)

############ SENTIMENT ANALYSIS & CLUSTERING
########
####
sentiment_list = []
po1_list = [] 
subj_list = []
for i in range(0, len(corpus)):
  sent = TextBlob(corpus['text_data'][i]).sentiment
  po1_list.append(sent.polarity)
  subj_list.append(sent.subjectivity)
sent_dict = {'Subjectivity':subj_list, 'Polarity':po1_list}
sent_df = pd.DataFrame(sent_dict)
sent_np = np.asarray(sent_df)
#begin clustering
model = KMeans(2)
model.fit(sent_np)
#model.labels_

# Plot Sentiments
centroids = model.cluster_centers_
labels = model.labels_
colors = ["g.", "r."]
for i in range(len(sent_np)):
  plt.plot(sent_np[i][0], sent_np[i][1], colors[labels[1]], markersize=8)
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s=100, linewidths=6)
plt.xlabel("Subjectivity")
plt.ylabel("Polarity")
plt.show()

############ TOPIC MODELING
########
####
dict_words = create_dict(new_corpus, dtm)
lda = models.LdaModel(corpus= new_corpus, id2word= dict_words, num_topics=3, passes=80)
lda.print_topics()

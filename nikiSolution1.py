import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("."))

import matplotlib.pyplot as plt
import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.naive_bayes import GaussianNB
import re
import spacy
from sklearn.utils import shuffle

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, confusion_matrix, recall_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import keras

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

train_raw.shape

train_raw['Star Rating'].value_counts()

plt.hist(train_raw['Star Rating'])
plt.show()


train_raw.dropna(subset=['Review Text'],inplace=True)
all_reviews = []
for item in train_raw['Review Text']:
    temp = item
    temp = temp.lower()
    cleanr = re.compile('<.*?>')
    temp = re.sub(cleanr, ' ', temp)
    temp = re.sub(r'[?|!|\'|"|#]', r'', temp)
    temp = re.sub(r'[.|,|)|(|\|/]', r'', temp)
    #     temp = [word for word in temp.split(' ') if word not in set(stopwords.words('english'))]
    all_reviews.append(temp)

test_raw.dropna(subset=['Review Text'],inplace=True)
all_reviews_test = []
for item in test_raw['Review Text']:
    temp = item
    temp = temp.lower()
    cleanr = re.compile('<.*?>')
    temp = re.sub(cleanr, ' ', temp)
    temp = re.sub(r'[?|!|\'|"|#]', r'', temp)
    temp = re.sub(r'[.|,|)|(|\|/]', r'', temp)
    #     temp = [word for word in temp.split(' ') if word not in set(stopwords.words('english'))]
    all_reviews_test.append(temp)

X = all_reviews
Y = train_raw['Star Rating']
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

x_test = all_reviews_test
y_test = test_raw['Star Rating']

vectorizer = TfidfVectorizer(ngram_range=(1,2))

x_train_vectors = vectorizer.fit_transform(x_train)
x_valid_vectors = vectorizer.transform(x_valid)
x_test_vectors = vectorizer.transform(x_test)

selector = SelectKBest(f_classif, k=min(1000, x_train_vectors.shape[1]))
selector.fit(x_train_vectors, y_train)
x_train_vectors = selector.transform(x_train_vectors).astype('float32')

x_valid_vectors = selector.transform(x_valid_vectors).astype('float32')
x_test_vectors = selector.transform(x_test_vectors).astype('float32')

round(train_raw['Star Rating'].value_counts() * 10 / train_raw.shape[0])
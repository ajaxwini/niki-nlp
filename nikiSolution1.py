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
# y_test = test_raw['Star Rating']

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

few_classes = list(map(lambda x: x in list(range(2, 5)), y_train.tolist()))

more_classes = list(map(lambda x: x in [1, 5], y_train.tolist()))

np.sum(few_classes)

few_classes = np.array(few_classes)
more_classes = np.array(more_classes)

x_train_few = x_train_vectors[few_classes]
y_train_few = y_train[few_classes]

x_train_more = x_train_vectors[more_classes]
y_train_more = y_train[more_classes]

all_x_train = list(x_train_few)
all_y_train = list(y_train_few)

for i in range(2):
    all_x_train.extend(x_train_few)
    all_y_train.extend(y_train_few)

all_x_train.extend(x_train_more)
all_y_train.extend(y_train_more)


all_x_train = np.concatenate([x_train_more.toarray(), x_train_few.toarray(), x_train_few.toarray(), x_train_few.toarray()], axis=0)
all_y_train = np.concatenate([y_train_more, y_train_few, y_train_few, y_train_few], axis=0)

all_y_train.shape

new_x_train, new_y_train = shuffle(all_x_train, all_y_train)

new_x_train.shape

##MODEL

clf = ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight='balanced')

clf.fit(x_train_vectors, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(x_train_vectors, y_train) * 100))
# print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(x_test_vectors, y_test) * 100))

y_pred = clf.predict(x_test_vectors) ##RESULTS
# precision_score(y_test, y_pred, average='macro')





# MLPClassifier



# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:32:27 2019

@author: Sharaaf Nazeer
"""

import pymongo
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt

#nltk.download('all')

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb

categoryId = 2

placeCollection = db.places

placeData = placeCollection.find({'category_id': categoryId})

reviewData = []
for place in placeData:
    
    if 'reviews' in place:
        for review in place["reviews"]: 
            reviewData.append(review)

placeReviews = pd.DataFrame(reviewData, columns=["place_id", "text", "rating"])
placeReviews.shape
placeReviews.head
#print(placeReviews)

placeReviews["is_good"] = np.where(placeReviews['rating']>=3, 1, 0)

pd.crosstab(index = placeReviews['is_good'], columns="Total count")

from sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(placeReviews['text'], placeReviews['is_good'], random_state=0)


print(X_test, y_test)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
# Fit the CountVectorizer to the training data
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
def tokenize(text):
    #text = stemmer.stem(text)               #stemming
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations and Underscores
    tokens = nltk.word_tokenize(text)       #tokenizing
    return tokens

# Fit the CountVectorizer to the training data specifiying a 
# minimum document frequency of 5 and extracting 1-gram and 2-gram
vect = CountVectorizer(min_df=5, ngram_range=(1,2), tokenizer=tokenize).fit(X_train)
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression(solver='lbfgs')


model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(predictions)

# These reviews are treated the same by our current model
#print(model.predict(vect.transform(['bad','not good'])))

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


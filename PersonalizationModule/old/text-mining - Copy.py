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

#placeReviews = placeReviews[placeReviews['rating']!= 3]
placeReviews["is_good"] = np.where(placeReviews['rating']>=3, 1, 0)

pd.crosstab(index = placeReviews['is_good'], columns="Total count")

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
def tokenize(text):
    #text = stemmer.stem(text)               #stemming
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations and Underscores
    tokens = nltk.word_tokenize(text)       #tokenizing
    return tokens

countvec = CountVectorizer(min_df=5, tokenizer=tokenize, stop_words=stopwords.words('english'))
print(placeReviews)




dtm = pd.DataFrame(countvec.fit_transform(placeReviews['text']).toarray(), columns=countvec.get_feature_names(), index=None)
#Adding label Column
dtm['is_good'] = placeReviews['is_good']
dtm.head()
###Building training and testing sets
df_train = dtm[:55]

#dtm1 = pd.DataFrame(countvec.fit_transform(['this place is too good', 'this is horrible']).toarray(), columns=countvec.get_feature_names(), index=None)
#df_test = dtm1
df_test = dtm[55:]

print(df_train)
print(df_test)

################# Building Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
X_train= df_train.drop(['is_good'], axis=1)
#Fitting model to our data
clf.fit(X_train, df_train['is_good'])

#Accuracy
X_test= df_test.drop(['is_good'], axis=1)
clf.score(X_test,df_test['is_good'])

#Prediction
pred_sentiment=clf.predict(X_test)
print(pred_sentiment)


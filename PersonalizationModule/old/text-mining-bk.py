# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:32:27 2019

@author: Sharaaf Nazeer
"""

import pymongo
import pandas as pd
import numpy as np
import nltk
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

placeReviewsTiny = placeReviews[0:3]
placeReviewsTiny.columns.values

from nltk.tokenize import sent_tokenize, word_tokenize

exampleText = placeReviewsTiny["text"][0]
print(exampleText)

sentTokens = sent_tokenize(exampleText)
print(sentTokens)

wordTokens = word_tokenize(exampleText)
print(wordTokens)


from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
print(len(stopWords))
print(stopWords)

#Removing stopwords from resource
wordTokens = [word for word in wordTokens if not word in stopWords]
print(wordTokens)

stopWords.update(['.', '..', '...', ',', '"', '?', '!', ';', ':', '(', ')', '[', ']', '{', '}'])
print(len(stopWords))

wordTokens = [word for word in wordTokens if not word in stopWords]
print(wordTokens)


#Lemmatizing

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizedTokens = [lemmatizer.lemmatize(word) for word in wordTokens]
print(lemmatizedTokens)
 
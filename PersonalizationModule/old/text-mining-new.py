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
placeReviews["sentiment"] = np.where(placeReviews['rating']>3, 1, 0)

reviewData = list(zip(placeReviews["text"], placeReviews["sentiment"]))

# 80% for training
train_X, train_y = zip(*reviewData[:30])
# Keep 20% for testing
test_X, test_y = zip(*reviewData[30:])

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
 
 
lemmatizer = WordNetLemmatizer()
 
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
 
def clean_text(text):
    text = text.replace("<br />", " ")
    #text = text.decode("utf-8")
 
    return text
 
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
    text = clean_text(text)
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0
 
 
from sklearn.metrics import accuracy_score
pred_y = [swn_polarity(text) for text in test_X]
 
print (accuracy_score(test_y, pred_y)) # 0.6518
 
print(pred_y)
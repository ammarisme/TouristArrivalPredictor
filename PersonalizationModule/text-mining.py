# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:32:27 2019

@author: Sharaaf Nazeer
"""
import sys
import pymongo
import pandas as pd
import numpy as np
import random
import json
#import nltk
#nltk.download('all')

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb
reviewCollection = db.reviews

reviewData = reviewCollection.find()

def main(argv):
    
    placeReviews = pd.DataFrame(reviewData, columns=["place_id", "text", "rating"])
    placeReviews["sentiment"] = np.where(placeReviews['rating']>=3, 1, 0)
    
    placeReviews.shape
    placeReviews.head(20)
    
    placeReviews.isnull().sum()
    placeReviews.dropna(inplace=True)
    placeReviews["sentiment"].value_counts()
    
    
    from sklearn.model_selection import train_test_split
    
    X = placeReviews["text"]
    y = placeReviews["sentiment"]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    
    #count_vect = CountVectorizer()
    
    #count_vect.fit(X_train)
    #X_train_counts = count_vect.transform(X_train)
    
    #X_train_counts = count_vect.fit_transform(X_train)
    
    #X_train_counts
    
    #X_train.shape
    
    from sklearn.pipeline import Pipeline
    
    text_clf = Pipeline([
                         ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=word_tokenize)), 
                         ('clf', LinearSVC())])      
    text_clf.fit(X_train, y_train)
    
    predictions = text_clf.predict(X_test)
    
    
    #from sklearn.metrics import confusion_matrix, classification_report
    
    #print(confusion_matrix(y_test, predictions))
    
    
    #print(classification_report(y_test, predictions))
    
    from sklearn import metrics
    
    metrics.accuracy_score(y_test, predictions)
    
    testText = argv[1]
    
    #testText = "a beautiful place to visit"
    #testText1 = "Not user friendly,too expensive"
    
    predictionRes = text_clf.predict([testText])
    
    countWords = len(testText.split())
    
    rating = 1
    
    if(predictionRes==1):
        if(countWords >= 12):
            rating = 5
        elif(countWords >= 7):
            rating = 4
        else:
            rating = 3
    else:
        if(countWords >= 10):
            rating = 1
        else:
            rating = 2
    
    resultsDict = {
                "text" : testText,
                "rating" : rating,
                "score" : str(predictionRes[0])
            }
    
    results = json.dumps(resultsDict)
    print(results)

if __name__ == "__main__":
    main(sys.argv)

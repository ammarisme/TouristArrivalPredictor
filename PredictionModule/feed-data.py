# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:03:30 2019
"""

import pymongo #import mongo db
import pandas as pd #import pandas for dataset
import json
import googlemaps

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

gmaps = googlemaps.Client(key='AIzaSyCyuexFdP4n0_OaSw3Y0x36qY3iqvHBr70')


def getGoogleResults(placeName):
    results = gmaps.places(placeName, 'textquery')
    
    resultSet = {}
    
    if len(results['results']) != 0:
        placeData = gmaps.place(results['results'][0]["place_id"],'photos,formatted_address,name,rating,reviews')
        placeData = placeData['result']
              
        resultSet= {
                "place_id" : placeData["place_id"],
                "latitude" : placeData["geometry"]["location"]["lat"],
                "longitude" : placeData["geometry"]["location"]["lng"]
                }

    return resultSet

tourData = pd.read_excel('../Data/Project_Dataset.xlsx'); #get dataset
dataframe = pd.DataFrame(tourData,columns=['place', 'category', 'year', 'month', 'no_of_tourists', 'revenue', 'expenses']);
dataframe = pd.DataFrame(tourData);

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb

print("Data feeding initiated!!")

categoryCollection = db.categories #create or get categories
categoryCollection.drop() #drop categories
categoryCollection = db.categories #create or get categories empty table

dataframe['category_id'] = labelencoder.fit_transform(dataframe['category'])
categoryIdArray = dataframe["category_id"].unique();
categoryIdArray = sorted(categoryIdArray)

categoryArray = dataframe["category"].unique();
categoryArray = sorted(categoryArray)

print("Feeding all category data!!")
categoryData = []
index = 0;
for category in categoryArray:        
    myData = { "_id": int(categoryIdArray[index]), "name": category }
    categoryData.append(myData)
    index = index+1

results = categoryCollection.insert_many(categoryData) #insert all category data

categoryDataExtra = pd.read_excel('../Data/category.xlsx'); #get dataset
categoryDataExtra = pd.DataFrame.from_dict(categoryDataExtra) #create list from dataframe
categoryDataExtra = list(json.loads(categoryDataExtra.T.to_json()).values()) #convert to json format

for cat in categoryDataExtra:
    categoryCollection.update_one({'_id' : cat["category_id"]}, {"$set" : {"image" : cat["image"]}})

#places db collection - feed places
placeCollection = db.places
placeCollection.drop()
placeCollection = db.places

dataframe['place_id'] = labelencoder.fit_transform(dataframe['place']) #generate unique id for all places
placeIdArray = dataframe["place_id"].unique();
placeIdArray = sorted(placeIdArray)

placeArray = dataframe["place"].unique();
placeArray = sorted(placeArray)

print("Feeding all place data!!")
placeData = []
fullReviewData = []

key = 0;

tempData = pd.DataFrame(tourData,columns=['place_id', 'category_id', 'place_name']); 
tempData = pd.DataFrame.from_dict(tempData) #create list from dataframe
tempData = json.loads(tempData.T.to_json()).values() #convert to json format

placeDataExtra = pd.read_excel('../Data/important.xlsx'); #get dataset
placeDataExtra = pd.DataFrame.from_dict(placeDataExtra) #create list from dataframe
placeDataExtra = list(json.loads(placeDataExtra.T.to_json()).values()) #convert to json format

#print((placeDataExtra))

for place in placeArray:        
    myData = { "_id": int(placeIdArray[key]), "name": place } 
    
    for placeExtra in placeDataExtra:
        if(int(placeExtra["place_id"]) == int(placeIdArray[key])):
            myData["max_count"] = placeExtra["max_count"]
            myData["ticket_price"] = placeExtra["ticket_price"]
            myData["avg_waste_qty"] = placeExtra["avg_waste_qty"]
            myData["max_profit_needed"] = placeExtra["max_rofit_needed"]            
            myData["waste_qty_by_person"] = placeExtra["waste_qty_by_person"]
            myData["description"] = placeExtra["description"]            
            myData["image"] = placeExtra["image"]
    
    for temp in tempData:
        if int(temp["place_id"]) == int(placeIdArray[key]):
            myData["category_id"] = temp["category_id"]
            myData["original_name"] = temp["place_name"]
            
            
    details = getGoogleResults(myData["original_name"]);
    
    myData["place_id"] = ""
    myData["latitude"] = ""
    myData["longitude"] = ""
    
    if(len(details) != 0):    
        myData["place_id"] = details["place_id"]
        myData["latitude"] = details["latitude"]
        myData["longitude"] = details["longitude"]           
        
    placeData.append(myData)
    key = key+1
else:
    
    placeCollection.insert_many(placeData)
    


print("Feeding reviews!!")

reviewCollection = db.reviews
reviewCollection.drop()
reviewCollection = db.reviews

reviewDataset = pd.read_excel('../Data/Reviews.xlsx'); #get dataset
reviewDataset = pd.DataFrame.from_dict(reviewDataset) #create list from dataframe
reviewDataset = list(json.loads(reviewDataset.T.to_json()).values()) #convert to json format
reviewCollection.insert_many(reviewDataset)

#dataset db collection - feed original dataset
dataCollection = db.dataset
dataCollection.drop()
dataCollection = db.dataset

print("Feeding datasets!!")
dataset = pd.DataFrame(tourData,columns=['place_id', 'year', 'month', 'no_of_tourists', 'revenue', 'expenses']);
dataset = pd.DataFrame.from_dict(dataset)

records = json.loads(dataset.T.to_json()).values()
dataCollection.insert_many(records)

print("Data feeding completed!!")

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:14:46 2019

@author: Sharaaf Nazeer
"""

# Importing the libraries


import pandas as pd
import numpy as np
import sys
import googlemaps
import json
import pymongo
import datetime
from pandas.io.json import json_normalize
import copy

gmaps = googlemaps.Client(key='AIzaSyCyuexFdP4n0_OaSw3Y0x36qY3iqvHBr70')

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb
placeCollection = db.places
bookingCollection = db.bookings

placeCollection = db.places

startLatLong = (6.855948499999999, 79.86296829999999)

destinations = placeCollection.find({ '_id': { '$in': [ 8 , 9, 7] } })


distanceMatrix = [];
all_sets = []
g = {}
p = []

finalRes = [1];

def execute(placeArray):
    n = len(placeArray)
    
    for x in range(1, n):
        g[x + 1, ()] = distanceMatrix[x][0]

    placeCount = []
    key = 0
    for x in range(2, n+1):
        placeCount.append(x)
        key = key+1
        
    placeCount = tuple(placeCount)
    #print(placeCount)
    
    get_minimum(1, placeCount)

    #print('\n\nSolution to TSP: {1, ', end='')
    
    
    solution = p.pop()
    #print(solution[1][0], end=', ')
    
    finalRes.append(solution[1][0])
    #print(n)
    
    for x in range(n-2):      
        for new_solution in p:
            
            
            if tuple(solution[1]) == new_solution[0]:
                solution = new_solution
                
                finalRes.append(solution[1][0])
                #print(finalRes)
                #print(solution[1][0], end=', ')
                break
    #print('1}')
    #finalRes.append(1)
    
    return finalRes

def get_minimum(k, a):

    
    if (k, a) in g:
        # Already calculated Set g[%d, (%s)]=%d' % (k, str(a), g[k, a]))
        return g[k, a]

    values = []
    all_min = []
    for j in a:
        set_a = copy.deepcopy(list(a))
        set_a.remove(j)
        all_min.append([j, tuple(set_a)])
        result = get_minimum(j, tuple(set_a))
        values.append(distanceMatrix[k-1][j-1] + result)

    #print(values)

    # get minimun value from set as optimal solution for
    g[k, a] = min(values)
    p.append(((k, a), all_min[values.index(g[k, a])]))

    return g[k, a]

def main():
    #startLatLong = tuple(argv[1].split(','))
    #destinations  = argv[2]
    #destinations = json.loads(destinations)
    latLongArray = [];
    latLongArray.append(startLatLong)
    placeArray=[];
    placeArray = [-1]
    
    
    for place in destinations:
      #  print(place)
        latLong = (place["latitude"], place["longitude"])
        latLongArray.append(latLong)
        placeArray.append(place['_id'])
    
    results = gmaps.distance_matrix(latLongArray, latLongArray, departure_time=datetime.datetime.now())

    matrixdf = json_normalize(results,['rows','elements'])

    distance = matrixdf["distance"]
    
    key = 1;
    disMet = []
    for dis in distance:
        arrLen = len(placeArray)  
        if key <=arrLen:
            disMet.append(dis['value'])
        
        if key == arrLen:
            distanceMatrix.append(disMet)
            key = 0
            disMet = []       
        
        key = key+1

    #print(distanceMatrix)
    paths = execute(placeArray)    
    routePath = []
    for path in paths:
        key = path-1
        routePath.append(placeArray[key])
    
    #json_dict = json.dumps(routePath)
    #print(json_dict)
    #print(routePath)
    
    #date=argv[3]
    date= '2019-09-07'
    #num_of_days = int(argv[4])
    num_of_days = 1
    #people = int(argv[5])
    people = 12
    
    routePath = routePath[1:]
    
    dateRange = pd.date_range(date, periods=num_of_days, freq='D')
    dateRange = pd.Series(dateRange.format())
    
    placesArr = []
    
    bookingCollection = db.bookings
    bookings = bookingCollection.find()
    
    bookingsArr =[];
    
    
    for booking in bookings:
        bookingDict = {
                    "booking_id": booking["_id"],
                    "people": booking["people"]
                }
        places = booking["places"]
        
        for place in places:
            bookingDict["place_id"] = place["place_id"]
            bookingDict["date"] = place["date"]
        
        bookingsArr.append(bookingDict)    
    
    #print(bookingsArr)
    
    for path in routePath:
        place = placeCollection.find_one({ '_id': path})
        placeDict = {
                    "id" : place["_id"],
                    "google_id" : place["place_id"],
                    "max_count" : place["max_count"],
                    "latitude" : place["latitude"],
                    "longitude" : place["longitude"],
                    "name" : place["name"],
                    "original_name" : place["original_name"]
                }
       
        headCounts = []
        for headCount in place["head_count"]:
            for date in dateRange:
                if(headCount["date"] == date):
                    headCounts.append(headCount)
        
        placeDict["head_count"] = headCounts
        
        
        bookings =[]
        for book in bookingsArr:
            if(book["place_id"] == place["_id"]):
                bookings.append(book)
            
        #print(bookings)    
        placeDict["bookings"] = bookings
        placesArr.append(placeDict)
    
    
    decisonArr = []
    
    for tempPlace in placesArr:
        
#        if (len(dateRange)==1):
#            print
#        else: 
        
            for date in dateRange:
                
                bookingCount =0
                canAllocate = True

                if(len(tempPlace["bookings"])>0): #count number of bookings
                    
                    for tempBooking in tempPlace["bookings"]:
                        if(tempBooking["date"] == date):
                            bookingCount = bookingCount + tempBooking["people"] +people
                        else:
                           bookingCount = bookingCount +people
                    
                    for tempHeadCount in tempPlace["head_count"]:
                        if(tempHeadCount["date"] == date):                    
                            if(bookingCount/tempHeadCount["no_of_tourists"] * 100 > 75):
                                canAllocate = False                    
        
                
        
                tempDict = {
                        "date" : date,
                        "booking_count" : bookingCount,
                        "place_id" : tempPlace["id"],
                        "google_id" : tempPlace["google_id"],
                        "location" : {
                                "latitude" : tempPlace["latitude"],
                                "longitude" : tempPlace["longitude"]
                                },
                        "name" : tempPlace["name"],
                        "original_name" : tempPlace["original_name"],
                        "can_allocate": canAllocate
                        }
                
                decisonArr.append(tempDict)
    
    filteredDescision = list(filter(lambda x: x["can_allocate"] == True, decisonArr))
    filteredDescisionReject = list(filter(lambda x: x["can_allocate"] == False, decisonArr))
    
    x = list(range(0, len(placesArr)))
    splittedArray = list(np.array_split(x, num_of_days));
    
    key =0
    fullfilmentDict = {}
    for splitted in splittedArray:    
        fullfilmentDict[dateRange[key]] = len(splitted)
        key=key+1
        
    #print(fullfilmentDict)    
        
    sortedFinal = {}
    import itertools
    from operator import itemgetter
    sortedFiltered = sorted(filteredDescision, key=itemgetter('date'))
    for key, group in itertools.groupby(sortedFiltered, key=lambda x:x['date']):
        sortedFinal[key] = list(group)
    
    
    
    finalRes = []
    selectedPlaces=[]
    
    for sortedElm in sortedFinal:
        maxCount = fullfilmentDict[sortedElm]
        #print(maxCount)
        expectedResult = [d for d in sortedFinal[sortedElm] if d['place_id'] not in selectedPlaces]
        
        i=1
        for elem in expectedResult:
            if(i>maxCount):
                break       
           
            finalRes.append(elem)
            selectedPlaces.append(elem["place_id"])
            i=i+1
        #print(selectedPlaces)
        #print("----")
    
    rejectedArr = []    
#    rejectedArr.append(7)
#    rejectedArr.append(7)
    for rejected in filteredDescisionReject:
        rejectedArr.append(rejected['place_id'])
#    
#    print(filteredDescisionReject)
#    
        
#    print(rejectedArr)
    rejectedArr =np.unique(rejectedArr)
    
    rejectedFinalArr = [];
    for rejected in filteredDescisionReject:
        if(rejected['place_id'] in rejectedArr):
            rejectedFinalArr.append(rejected)
    
#    print(rejectedFinalArr)
            
    lastFinalResults = {'selected' : finalRes, 'rejected' : rejectedFinalArr}        
            
    json_dict = json.dumps(lastFinalResults)
    print(json_dict)
    sys.stdout.flush()

if __name__ == "__main__":
    main()



     
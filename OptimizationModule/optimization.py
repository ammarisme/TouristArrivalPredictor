# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:52:40 2019
"""

import ga
import pymongo

solPerPop = 8
numGenerations = 200
parentsMating = 4

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb
placeCollection = db.places
placeData = placeCollection.find()

headCountArr={}
datesArray = {}
for place in placeData:
    headCountArr[place["_id"]] = []
    for singleObj in place["predictions"]:
        if not singleObj["date"] in datesArray:
            datesArray[singleObj["date"]] = []
        datesArray[singleObj["date"]].append({
                "_id": place["_id"],
                "max_count":place["max_count"],
                "ticket_price":place["ticket_price"],
                "avg_waste_qty":place["avg_waste_qty"],
                "min_profit":place["max_profit_needed"],
                "waste_qty_pers":place["waste_qty_by_person"],
                "no_of_tourists": singleObj["no_of_tourists"]
                })    

for date, places in datesArray.items():
    firstGene = []
    placesArray = []
    for placeObj in places:
        firstGene.append(placeObj["no_of_tourists"])
        placesArray.append({
                "_id": placeObj["_id"],
                "max_count":placeObj["max_count"],
                "ticket_price":placeObj["ticket_price"],
                "avg_waste_qty":placeObj["avg_waste_qty"],
                "min_profit":placeObj["min_profit"],
                "waste_qty_pers":placeObj["waste_qty_pers"]
                })
    
    newPopulation = ga.newPopulation(firstGene,solPerPop)
    parents=[]
    
    for generation in range(numGenerations):
        fitnessArray = ga.populationFitness(newPopulation,placesArray)
        parents = ga.selectMatingPool(newPopulation,fitnessArray,parentsMating)
        offspringCrossover = ga.crossover(parents,offspringSize=(solPerPop-parents.shape[0], len(placesArray)))
        offspringMutation = ga.mutation(offspringCrossover)
        newPopulation[0:parents.shape[0], :] = parents
        newPopulation[parents.shape[0]:, :] = offspringMutation
        
        
    bestOption = parents[0]
    for index,optimized in enumerate(bestOption):
        headCountArr[placesArray[index]["_id"]].append({"date": date, "no_of_tourists": int(optimized)})
 
for placeId,headCount  in headCountArr.items():
    placeCollection.update_one({'_id' : placeId}, {"$set" : {"head_count" :headCount}})
    

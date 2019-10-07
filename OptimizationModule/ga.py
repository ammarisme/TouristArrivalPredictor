# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:27:06 2019
"""

import numpy


def calculateFitness(touristsCount,placeObj):
    fitness=0
    maxTourists = placeObj['max_count']
    minProfit = placeObj['min_profit']
    ticketPrice=placeObj['ticket_price']
    maxWaste=placeObj['avg_waste_qty']
    avgWaste=placeObj['waste_qty_pers']
    
    if(maxTourists >= touristsCount):
        fitness += 0.3
    else:
        fitness += 0.1
        
    if(minProfit <= touristsCount*ticketPrice):
         fitness += 0.3
    else:
        fitness += 0.1
        
    if(maxWaste >= touristsCount*avgWaste):
         fitness += 0.3
    else:
        fitness += 0.1
        
    return fitness


def newPopulation(firstChromosome, popSize):
    newPop = []
    newPop.append(firstChromosome)
    for popS in range(popSize-1):
        chromosome = []
        for gene in firstChromosome:
            randomValue = numpy.random.randint(-50, 50, 1)
            chromosome.append(gene + randomValue)
        newPop.append(chromosome)
    return numpy.array(newPop)

def populationFitness(population,placesArray):
    fitnessArray=[]
    for chromosome in population:
        totFitness = 0
        for index,gene in enumerate(chromosome):
            totFitness += calculateFitness(gene,placesArray[index])
        fitnessArray.append(totFitness)
    return fitnessArray

def selectMatingPool(population, fitnessArray, numParents):
    parents = numpy.empty((numParents, population.shape[1]))

    for parentNum in range(numParents):

        maxFitnessIdx = numpy.where(fitnessArray == numpy.max(fitnessArray))

        maxFitnessIdx = maxFitnessIdx[0][0]

        parents[parentNum, :] = population[maxFitnessIdx, :]

        fitnessArray[maxFitnessIdx] = -99999999999

    return parents

def crossover(parents, offspringSize):
     offspring = numpy.empty(offspringSize)
     crossoverPoint = numpy.uint8(offspringSize[1]/2)
     for k in range(offspringSize[0]):
         parent1Idx = k%parents.shape[0]
         parent2Idx = (k+1)%parents.shape[0]
         offspring[k, 0:crossoverPoint] = parents[parent1Idx, 0:crossoverPoint]
         offspring[k, crossoverPoint:] = parents[parent2Idx, crossoverPoint:]
     return offspring
 
def mutation(offspringCrossover):
    for idx in range(offspringCrossover.shape[0]):
        randomValue = numpy.random.randint(1, 150, 1)
        offspringCrossover[idx, 4] = offspringCrossover[idx, 4] + randomValue
    return offspringCrossover
    
    
    
    
    
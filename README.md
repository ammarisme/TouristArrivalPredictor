
# Tourist Arrival Predictor

## Project Goal
Efficiently allocate tourists to different places to maximize profit, manage waste, and adhere to capacity constraints using a genetic algorithm.

## Directory Structure
```
|-- .gitignore
|-- Data
    |-- Project_Dataset.xlsx
    |-- Reviews.xlsx
    |-- category.xlsx
    |-- important.xlsx
|-- OptimizationModule
    |-- __pycache__
        |-- ga.cpython-37.pyc
    |-- ga.py
    |-- optimization.py
|-- PersonalizationModule
    |-- old
        |-- test.py
        |-- test.txt
        |-- test2.py
        |-- test3.py
        |-- text-mining - Copy.py
        |-- text-mining-bk.py
        |-- text-mining-new.py
        |-- text-mining.py
    |-- text-mining.py
|-- PredictionModule
    |-- eee.png
    |-- feed-data.py
    |-- fff.png
    |-- final.png
    |-- lll.png
    |-- model
        |-- 1
            |-- checkpoint
            |-- ex_time_series_model.data-00000-of-00001
            |-- ex_time_series_model.index
            |-- ex_time_series_model.meta
        |-- 2
            |-- checkpoint
            |-- ex_time_series_model.data-00000-of-00001
            |-- ex_time_series_model.index
            |-- ex_time_series_model.meta
        |-- 3
            |-- checkpoint
            |-- ex_time_series_model.data-00000-of-00001
            |-- ex_time_series_model.index
            |-- ex_time_series_model.meta
        |-- New folder
            |-- checkpoint
            |-- ex_time_series_model_25.data-00000-of-00001
            |-- ex_time_series_model_25.index
            |-- ex_time_series_model_25.meta
        |-- checkpoint
        |-- ex_time_series_model.data-00000-of-00001
        |-- ex_time_series_model.index
        |-- ex_time_series_model.meta
    |-- prediction.py
    |-- train-model.py
    |-- ttt.png
|-- TravelGuideModule
    |-- test.py
    |-- travel-guide-test.py
    |-- travel-guide.py
|-- conda-module.txt
|-- pip-module.txt
```

## Modules

### OptimizationModule

#### `ga.py`
This file implements a genetic algorithm for optimizing tourist allocation to different places. Key functions include:
- `calculateFitness()`: Evaluates the fitness of a tourist count for a given place.
- `newPopulation()`: Generates a new population of chromosomes.
- `populationFitness()`: Calculates the fitness for a population of chromosomes.
- `selectMatingPool()`: Selects the best-fitting chromosomes for mating.
- `crossover()`: Performs crossover between pairs of chromosomes.
- `mutation()`: Introduces random mutations to the offspring.

#### `optimization.py`
This script uses the genetic algorithm to optimize tourist allocations based on historical data stored in a MongoDB database. It connects to the database, retrieves data, and runs the genetic algorithm over multiple generations to find the best tourist allocation for each place and date.

### PersonalizationModule

#### `old/test.py`
This script demonstrates time series forecasting using a recurrent neural network (RNN) with TensorFlow. It includes data preprocessing, model definition, training, and visualization of results.

#### `old/test2.py`
This script uses an LSTM (Long Short-Term Memory) network for predicting international airline passengers. It includes data loading, normalization, model creation, training, and prediction.

#### `old/test3.py`
This script also uses an LSTM network for predicting stock prices. It includes data loading, visualization, preprocessing, model creation, training, and evaluation.

#### `old/text-mining - Copy.py` and `old/text-mining-bk.py`
These scripts perform text mining on reviews stored in a MongoDB database. They include data loading, preprocessing, feature extraction using CountVectorizer, and sentiment analysis using Naive Bayes and logistic regression models.

### PredictionModule

#### `feed-data.py`
This script feeds data into the MongoDB database from an Excel file. It includes data transformation, encoding, and insertion into various collections like categories, places, reviews, and the main dataset.

#### `prediction.py`
This script predicts the number of tourists for each place using an RNN model. It includes data preparation, scaling, model definition, training, and prediction.

#### `train-model.py`
This script trains an RNN model on tourist data for a specific place. It includes data loading, preprocessing, model creation, training, and prediction of future values.

### TravelGuideModule

#### `test.py` and `travel-guide-test.py`
These scripts implement a solution for the Traveling Salesman Problem (TSP) using a distance matrix and dynamic programming. They calculate the optimal route for a given set of places based on their latitude and longitude.

#### `travel-guide.py`
This script generates a travel guide based on the TSP solution and availability of places. It includes distance matrix calculation, route optimization, and allocation decision-making based on bookings and headcounts.

## Summary
The repository consists of various modules focusing on optimization, personalization, prediction, and travel guide generation using machine learning and genetic algorithms. The scripts utilize TensorFlow, Keras, and MongoDB to process data, train models, and make predictions. Key functionalities include time series forecasting, sentiment analysis, tourist allocation optimization, and route planning for travel guides.

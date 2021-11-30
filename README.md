# CAB PRICE PREDICTOR

A ML regressor predictor to predict costs of different ride sharing apps and determine what factors influence the demand. 

### MOTIVATION
Ridesharing apps have become mainstream in the last decade overtaking taxis and other traditional forms of transportation in some markets. The demand for these services are at an all time high. When the demand increases, dynamic pricing encourages more drivers to handle riders requests. The motivation of this project is to identify the factors that influence dynamic pricing of cab fares.

## DATASET 
[Uber and Lyft Cab Prices](https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices)

## BUILT USING
[Neural Network - Multi-Layer Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) -This model optimizes the squared error using LBFGS or stochastic gradient descent.

## FILE STRUCTURE
### data: folder contains two datasets weather.csv and cab_rides.csv
### test: a sample record set that can be used to predict ride prices
### predictor.py: python file that is used to read the data, preprocess the data, build a model, train the model and predict cab prices
### requirement.txt: a list of required libraries that has to be installed in order to run the program

## Usage
### You need python3+ version installed before running
### Install libraries
To install required library execute below command:
``` 
pip install -r requirements.txt
```
### Run the program
```
python predictor.py
```

## RESULTS
The model has R-squred value around 95%

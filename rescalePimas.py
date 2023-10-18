'''
Program that takes the input attributes from the Pimas Indians dataset and
normalizes the input data and outputs the normalized data.

Extensions implemented are: changing the normalization range from 0-1 to -1-1, create method to log the
values in the dataset, method to exp the values in the dataset to specified value, a method to sqrt the values and a
method to normalize the dataset using the box-cox method.
'''
from csv import reader
from math import sqrt, log
import numpy as np
from loadDataScratch import load_csv, str_column_to_float


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Normalize dataset columns to the range -1 - 1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            # Rescaled to be normalized as values between -1 & 1
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) * 2 - 1


# Create a method to log all values in the dataset
def log_dataset(dataset):
    for row in dataset:
        for i in range(len(row)):
            # Cannot handle a log of 0
            if row[i] == 0:
                row[i] = 0
            else:
                row[i] = log(row[i])


# Create a method to square root all values in the dataset
def sqrt_dataset(dataset):
    for row in dataset:
        for i in range(len(row)):
            # Cannot sqrt negative values, so just return the value as is
            if row[i] < 0:
                row[i] = row[i]
            else:
                row[i] = sqrt(row[i])


# Create method to raise the dataset by a specified value or 2
def exp_dataset(dataset, power=2):
    for row in dataset:
        for i in range(len(row)):
            row[i] = row[i] ** power


# Create a method to normalize the data using the box-cox formula, setting the lambda to be 2 by default
# Happens to be exactly the same as the exp method so actually quite useless
def box_cox_dataset(dataset, ld=2):
    for row in dataset:
        for i in range(len(row)):
            if row[i] == 0:
                row[i] = row[i]
            else:
                row[i] = row[i] ** ld


# Separate the main file so the methods can be imported into other files
if __name__ == '__main__':
    # Load in the Pimas Indians dataset
    file = 'pimas-indians-diabetes.csv'
    dataset = load_csv(file)
    print('Loaded data file %s with %d rows and %d columns' % (file, len(dataset), len(dataset[0])))

    # Have to delete the first row in the data, save it in a var though
    header = dataset[0]
    del dataset[0]

    # Convert string columns to floats
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    print('\n', dataset[0])

    # Create copies of the dataset to log, sqrt, exp and box-cox
    log_copy = np.array(dataset)
    sqrt_copy = np.array(dataset)
    exp_copy = np.array(dataset)
    box_copy = np.array(dataset)

    # Calculate the min and max for each column in the dataset
    minmax = dataset_minmax(dataset)
    # Normalize the data in the dataset
    normalize_dataset(dataset, minmax)
    print('\n', dataset[0])

    # Log the data
    #print(log_copy[0])
    log_dataset(log_copy)
    print('\n', log_copy[0])

    # Sqrt the data
    #print(sqrt_copy[0])
    sqrt_dataset(sqrt_copy)
    print('\n', sqrt_copy[0])

    # Raise the data by a power of 4
    #print(exp_copy[0])
    exp_dataset(exp_copy, power=4)
    print('\n', exp_copy[0])

    # Box cox the dataset with a lambda of -3
    #print(box_copy[0])
    box_cox_dataset(box_copy, ld=-3)
    print('\n', box_copy[0])
'''
Example code loading and editing the data from the Pimas Indians & Iris Flowers
datasets to get a better understanding of what can be done when downloading
csv files.

Extensions implemented are methods to: replace all missing values with 0, a method that takes the outliers
(min & max values) and replaces them with the mean value for each column and converting the output of load_csv
from a list into a numpy array
'''
from csv import reader
import numpy as np


# Create a method to load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        # Read in each row to ensure that no blank rows are added
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Add method to convert string values in a column to floating point values
def str_column_to_float(dataset, column):
    for row in dataset:
        # Strip whitespace before making conversion
        row[column] = float(row[column].strip())


# Create method to convert all string values into an integer
def str_column_to_int(dataset, column):
    # Locate all unique class values
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    # Now assign an integer for each unique class value
    for i, value in enumerate(unique):
        lookup[value] = i

    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup


# Create method to detect and replace missing values with 0
# Replace with '0.0' due to this being done before converting all values to floats in the dataset
def nan_column_to_zero(dataset, column):
    for row in dataset:
        if row[column] == '':
            row[column] = '0.0'


# Create a method to help deal with 'outliers'
# For simplicity will assume the min and max values are 'outliers'
# These values will be replaced with the mean value of the column
def outlier_to_mean(dataset, column):
    # Get the mean, min and max values for the column
    avg = np.mean(dataset[column])
    mn = min(dataset[column])
    mx = max(dataset[column])
    # Now replace the min and max values with the average
    for i in range(len(dataset[column])):
        if dataset[column][i] == mn or dataset[column][i] == mx:
            dataset[column][i] = avg.round(3)


# Separate the main file as the methods in this file are commonly used in other files
if __name__ == '__main__':
    # Now use the load_csv method to load in the Pimas Indians data
    filename = 'pimas-indians-diabetes.csv'
    data = load_csv(filename)
    print('Loaded data file %s with %d rows and %d columns' % (filename, len(data), len(data[0])))

    # Change data type from list to array, commented prints show it works
    array_data = np.array(data)
    #print(array_data)
    #print(array_data.dtype)

    # Need to remove the header to convert column values to floating point
    header = data[0]
    del data[0]

    # The column values are floats:
    print('\nBefore: ', data[0])

    # Replace any missing values with 0
    for i in range(len(data[0])):
        nan_column_to_zero(data, i)

    # Convert string columns to floating point values
    for i in range(len(data[0])):
        str_column_to_float(data, i)

    # Covert the outliers (min and max values) to the average value for each column
    # First need to transpose the dataset
    data_t = list(map(lambda *x: list(x), *data)) # Same as data_t = data.T in pandas
    # Make this -1 due to the last column always being either 0 or 1
    for i in range((len(data_t)) - 1):
        outlier_to_mean(data_t, i)

    # Transpose back
    data = list(map(lambda *x: list(x), *data_t))

    print('\nAfter: ', data[0])

    # Load in the iris dataset
    file = 'IRIS.csv'
    iris_dataset = load_csv(file)
    print('\nLoaded data file %s with %d rows and %d columns' % (file, len(iris_dataset), len(iris_dataset[0])))

    # Need to delete header column for conversion to work
    iris_header = iris_dataset[0]
    del iris_dataset[0]

    print('\n', iris_dataset[0])
    # Convert string columns to float
    for i in range(4):
        str_column_to_float(iris_dataset, i)

    # Convert class column to int
    lookup = str_column_to_int(iris_dataset, 4)
    print('\n', iris_dataset[0])
    print('\n', lookup)
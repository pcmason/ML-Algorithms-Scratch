'''
In this project will create a linear regression method from scratch that will be tested on the Swedish Insurance dataset

Added extension that outputs a graph comparing the actual and predicted values when algorithm's evaluated also extended
to run on the Salary_dataset.csv data that compares years of experience (x) to salary (y) and graphs output vs predicted.
'''
from random import seed, randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt


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


# Split a dataset into train and test splits
def train_test_split(dataset, split=0.6):
    train = list()
    # Create the training dataset based on the split entered or the baseline
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    # While loop to add values to the training dataset until it is full
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    # Return both train and the dataset copy, which is the test split
    return train, dataset_copy


# Calculate the root mean square error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        pred_error = predicted[i] - actual[i]
        # Square the error so the value is positive
        sum_error += (pred_error ** 2)
        mean_error = sum_error / float(len(actual))
    # Take the square root to return to original units
    return sqrt(mean_error)


# Evaluate algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    # Create step size method so this can be used on multiple different datasets
    step_size = max(actual) / len(actual)
    step_size = int(round(step_size))
    # Output the preds compared the the actual
    plt.plot(range(0, int(max(actual)), step_size), predicted, label='Predictions')
    plt.plot(range(0, int(max(actual)), step_size), actual, label='Actual')
    plt.legend()
    plt.show()
    return rmse


# Calculate mean from list of numbers
def mean(values):
    return sum(values) / float(len(values))


# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x - mean)**2 for x in values])


# Now can calculate the covariance between 2 variables
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# Now can calculate b0 & b1 using all the methods above
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


# Put everything together to create the linear regression method
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions


if __name__ == '__main__':
    seed(1)
    # Load and prepare data
    file = 'insurance.csv'
    dataset = load_csv(file)
    # Delete header
    header = dataset[0]
    del dataset[0]
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    # Evaluate algorithm
    split = 0.7
    rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
    print('RMSE: %.3f' % rmse)

    # Load in salary dataset to test Simple Linear model on another dataset
    sal_file = 'Salary_dataset.csv'
    sal_data = load_csv(sal_file)
    # Delete header
    sal_header = sal_data[0]
    del sal_data[0]
    for i in range(len(sal_data[0])):
        str_column_to_float(sal_data, i)

    sal_split = 0.8
    # Evaluate algorithm on the salary dataset
    sal_rmse = evaluate_algorithm(sal_data, simple_linear_regression, sal_split)
    print('\nRMSE: %.3f' % sal_rmse)
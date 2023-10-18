'''
Program that takes the input attributes from the Pimas Indians dataset and
standardizes the input data and outputs the standardized data.

Implemented a spread parameter for the standardization method
'''
from csv import reader
from math import sqrt
from loadDataScratch import load_csv, str_column_to_float


# Calculate mean for standardization
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


# Calculate standard deviation for standardization
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs


# Function to use the means and stdevs to standardize the dataset
def standardize_dataset(dataset, means, stdevs, spread=1.0):
    for row in dataset:
        for i in range(len(row)):
            # Added spread parameter that allows to change how much the stdev should be taken into account
            row[i] = (row[i] - means[i]) / (stdevs[i] * spread)


# Separate the main file
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
    print(dataset[0])

    # Estimate mean and stdev for columns
    means = column_means(dataset)
    stdevs = column_stdevs(dataset, means)

    # Standardize the dataset
    standardize_dataset(dataset, means, stdevs, spread=.4)
    print('\n', dataset[0])

'''
Program that creates simple perceptron method from scratch and uses it to predict the binary classification problem from
the sonar dataset that differentiates between rocks and metal cylinders.

Extended to implement tuning of l_rate & n_epochs and implemented a batch Stochastic Gradient Descent method.
'''
from random import seed, randrange
from csv import reader


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


# Split dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    # Create an empty list for each fold to be filled
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            # Randomly fill each fold from the copied dataset
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        # Append the fold to the list of folds
        dataset_split.append(fold)
    # Return k-1 training sets and 1 test set
    return dataset_split


# Calculate classification accuracy
def class_accuracy(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if pred[i] == actual[i]:
            correct += 1
    return (correct / float(len(actual))) * 100


# Evaluate algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = class_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# Make a prediction with weights
def predict(row, weights):
    # First weight is the bias
    activation = weights[0]
    for i in range(len(row)-1):
        # Use activation to determine if 1 or 0 is returned
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    # Loop over each epoch
    for epoch in range(n_epoch):
        sum_error = 0.0
        # Loop over each row in the data
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            # Implement stochastic gradient descent
            weights[0] = weights[0] + l_rate * error
            sum_error += error ** 2
        # Loop over each weight and update it
        for i in range(len(row)-1):
            weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights


# Train the weights using batch Stochastic Gradient descent
# Estimate Perceptron weights using stochastic gradient descent
def batch_train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    # Loop over each epoch
    for epoch in range(n_epoch):
        sum_error = 0.0
        update_weights = list()
        # Loop over each row in the data
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            # Implement stochastic gradient descent
            update_weights.append(weights[0] + l_rate * error)
            sum_error += error ** 2
        for i in range(len(weights)):
            weights[i] = update_weights[i]
        # Loop over each weight and update it
        for i in range(len(row)-1):
            weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights


# Perceptron algorithm with stochastic gradient descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = batch_train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


# Now test out the perceptron algorithm
if __name__ == '__main__':
    seed(5)
    # Load and prepare data
    filename = 'sonar.csv'
    dataset = load_csv(filename)
    # Delete the header
    header = dataset[0]
    del dataset[0]
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # Convert string class to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # Evaluate algorithm
    n_folds = 5
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Acccuracy: %.3f' % (sum(scores) / float(len(scores))))


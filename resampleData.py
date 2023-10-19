'''
In this file will resample a testing dataset using two methods:
The train test split & K-fold CV. These methods will be implemented from
scratch.

Extensions include: Repeated Train & Test Splits, Leave One Out CV & Stratification for classification problems
'''

from random import seed, randrange

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


# Create extension to above method so it is a Repeated Train and Test split method
def rep_train_test_splits(dataset, split=0.6, repeats=4):
    output = list()
    # Should be same as method above just output multiple training and test splits
    for i in range(repeats):
        train = list()
        # Create training dataset based on the split
        train_size = split * len(dataset)
        dataset_copy = list(dataset)
        # While loop to add values to training dataset
        while len(train) < train_size:
            index = randrange(len(dataset_copy))
            train.append(dataset_copy.pop(index))
        # Add both train and test split to the output list
        output.append([train, dataset_copy])
    # Return the output list
    return output


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


# Leave One Out CV, Similar to kfold CV but k = n
def leave_one_out_split(dataset):
    folds = len(dataset)
    # The rest of the method should be the same as the above kfold CV method
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)) / folds
    # Create empty list to be filled
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            # Randomly fill each fold from the copied dataset
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        # Append fold to list of folds
        dataset_split.append(fold)
    #Return the Leave One Out test splits
    return dataset_split



seed(1)
# Example dataset
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# Test the train/test split
train, test = train_test_split(dataset, 0.7)
print(train)
print(test)

# Test the cross validation method
folds = cross_validation_split(dataset)
print(folds)

# Test repeated train/test split
repeat_splits = rep_train_test_splits(dataset)
print(repeat_splits)

# Test Leave One Out split method
loo_split = leave_one_out_split(dataset)
print('\n', loo_split)
# Should be equivalent to kfold, where folds = len(dataset)
kfold_loo = cross_validation_split(dataset, folds=len(dataset))
print('\n', kfold_loo)
print(len(loo_split) == len(kfold_loo))
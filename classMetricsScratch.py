'''
Program that implements classification accuracy and confusion matrix
manually that is tested on a basic classification dataset.

Extensions implemented includes: Precision, recall, and F1 metrics..
'''


# Calculate classification accuracy
def class_accuracy(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if pred[i] == actual[i]:
            correct += 1
    return (correct / len(actual)) * 100


# Calculate precision for classification
def precision(actual, pred):
    # Create counters for true positives and false positives
    tp = 0
    fp = 0
    # Loop through all values in actual and increment counters as necessary
    for i in range(len(actual)):
        if actual[i] == 1 == pred[i]:
            tp += 1
        if actual[i] == 1 and pred[i] == 0:
            fp += 1
    # Return the precision value
    return tp / (tp + fp)


# Calculate recall for classification
def recall(actual, pred):
    # Create counters for true positive and false negatives
    tp = 0
    fn = 0
    # Loop through values in actual and increment counters
    for i in range(len(actual)):
        if actual[i] == 1 == pred[i]:
            tp += 1
        if actual[i] == 0 and pred[i] == 1:
            fn += 1
    # Return the recall value
    return tp / (tp + fn)


# Now use precision & recall to calculate F1 score
def f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# Calculate a confusion matrix
def confusion_matrix(actual, pred):
    # List of unique class values
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    # Initialize matrix to 0
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    # Loop through predictions and increment value
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[pred[i]]
        matrix[y][x] += 1
    # Return a set of unique class values and the confusion matrix
    return unique, matrix


# Pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
    # Create label for unique class values
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    # Output the confusion matrix
    for i, x in enumerate(unique):
        print('%s | %s' % (x, ' '.join(str(x) for x in matrix[i])))


# Create 2 fake datasets to test the method on
actual = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
preds = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
# Test accuracy
accuracy = class_accuracy(actual, preds)
print('Classification Accuracy: %.1f' % accuracy)

# Test precision
prec = precision(actual, preds)
print('Precision: %.3f' % prec)

# Test recall
rec = recall(actual, preds)
print('Recall: %.3f' % rec)

# Test F1
fScore = f1(prec, rec)
print('F1: %.3f' % fScore)

# Create confusion matrix
uniq, matrix = confusion_matrix(actual, preds)
#print('Unique class values: ', uniq)
#print('Confusion Matrix: ', matrix)
print_confusion_matrix(uniq, matrix)
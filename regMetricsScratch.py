'''
Program that implements mean absolute error and root mean squared error
manually tested on a simple dataset
'''
from math import sqrt


# Calculate mean absolute error
def mae_metric(actual,  predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))


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


# Create simple dataset to test metrics
output = [0.1, 0.2, 0.3, 0.4, 0.5]
pred = [0.11, 0.19, 0.29, 0.41, 0.5]

# Test MAE
mae = mae_metric(output, pred)
print('Mean Absolute Error: %.4f' % mae)

# Test RMSE
rmse = rmse_metric(output, pred)
print('Root Mean Squared Error: %.5f' % rmse)

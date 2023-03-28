import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

N=10
a=np.random.normal((0,0), (2,5), (N,2)) # loc = Mean (“centre”), scale = Standard deviation (spread or “width”), size = int or tuple of ints
b=np.random.normal((10,20), (6,2), (N,2))
ay = np.tile(1, a.shape[0])
by = np.tile(0, b.shape[0])

M = 3
weights = np.random.rand(1,M)
data_sample = np.random.normal((0,0), (2,5), (2,2))

# first function, weights is 1, M shape and data sample is 1, 2 shape, y is 1, shape
def single_prceptron(weights, data_sample):
    bias = 1
    data_sample = np.insert(data_sample, 0, bias)
    result = weights@data_sample
    y_prediction = np.heaviside(result, 1)
    return y_prediction

# second function, 
def update_rule(weights, data_sample, true_ground, learning_rate):
    y_pred = single_prceptron(weights, data_sample)
    bias = 1
    data_sample = np.insert(data_sample, 0, bias)
    weights = weights + learning_rate * (true_ground - y_pred) * data_sample
    return weights

y_pred_arr = np.array([])
learning_rate = 0.1
for i in range(len(a)):
    #y = single_prceptron(weights, data_sample)
    #y_pred_arr = np.append(y_pred_arr, y)
    print("iteration num:", i, "current weights: ", weights)
    data_sample = a[i]
    true_ground = ay[i]
    weights = update_rule(weights, data_sample, true_ground, learning_rate)

# Plot
plt.plot(a[:,0],a[:,1],'r+')
plt.plot(b[:,0],b[:,1],'g+')

plt.show()
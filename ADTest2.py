"""
Test for gradient descent and automatic differentiation for linear regression.
Defining the loss function as the sum of all quadratic errors for all input training examples.
From https://github.com/mattnedrich/GradientDescentExample
"""

import numpy as np
from autograd import grad

points = None
grad_function = None

# y = mx + b
# m is slope, b is y-intercept => w:[m, b]
def loss_function( w ):
	global points
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += ( y - ( np.dot( w, np.array( [x, 1.0] ) ) ) ) ** 2
	return totalError / float(len(points))

def gradient_descent( w, learning_rate, num_iterations ):
	global grad_function
	weights = np.array( w )
	for i in range(num_iterations):
		weights -= grad_function( weights ) * learning_rate
	return weights

def run():
	global grad_function, points
	points = np.genfromtxt( "data.csv", delimiter="," )
	learning_rate = 0.0001
	grad_function = grad( loss_function )
	initial_b = 0.0 								# Initial y-intercept guess.
	initial_m = 0.0 								# Initial slope guess.
	w = np.array( [initial_m, initial_b] )			# Initial weights.
	num_iterations = 1000
	print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format( initial_b, initial_m, loss_function( w ) ) )
	print( "Running..." )
	w = gradient_descent( w, learning_rate, num_iterations )
	print( "After {0} iterations b = {1}, m = {2}, error = {3}".format( num_iterations, w[1], w[0], loss_function( w ) ))

run()
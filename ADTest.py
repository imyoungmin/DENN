"""
Test for gradient descent and automatic differentiation for linear regression.
From https://github.com/mattnedrich/GradientDescentExample
"""

import numpy as np
import autograd

cur_x = 0
cur_y = 0
grad_function = None

# y = mx + b
# m is slope, b is y-intercept => w:[m, b]
def compute_error_for_line_given_points( w, points ):
	global cur_x, cur_y
	totalError = 0
	for i in range(0, len(points)):
		cur_x = points[i, 0]
		cur_y = points[i, 1]
		totalError += loss_function( w )
	return totalError / float(len(points))

def loss_function( w ):
	"""
	Loss function for a single input pair of x, y values.
	:param w: weights vector = [m, b]
	:return: Squared error.
	"""
	global cur_x, cur_y
	return (cur_y - ( w[0] * cur_x + w[1] )) ** 2

def step_gradient( w, points, learningRate ):
	global grad_function, cur_x, cur_y
	w_grad = np.array([0.0, 0.0])
	N = float( len( points ) )
	for i in range( 0, len( points ) ):
		cur_x = points[i, 0]
		cur_y = points[i, 1]
		w_grad += grad_function( w )
	new_w = w - learningRate * w_grad / N
	return new_w

def gradient_descent_runner( points, w, learning_rate, num_iterations ):
	weights = np.array( w )
	for i in range(num_iterations):
		weights = step_gradient( weights, np.array(points), learning_rate )
	return weights

def run():
	global grad_function
	points = np.genfromtxt( "data.csv", delimiter="," )
	learning_rate = 0.0001
	grad_function = autograd.grad( loss_function )
	initial_b = 0.0 								# Initial y-intercept guess.
	initial_m = 0.0 								# Initial slope guess.
	w = np.array( [initial_m, initial_b] )			# Initial weights.
	num_iterations = 1000
	print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format( initial_b, initial_m, compute_error_for_line_given_points( w, points ) ) )
	print( "Running..." )
	w = gradient_descent_runner( points, w, learning_rate, num_iterations )
	print( "After {0} iterations b = {1}, m = {2}, error = {3}".format( num_iterations, w[1], w[0], compute_error_for_line_given_points( w, points ) ))

run()
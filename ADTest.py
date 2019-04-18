"""
Test for gradient descent and automatic differentiation for linear regression.
From https://github.com/mattnedrich/GradientDescentExample and https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
"""

import numpy as np
import autograd
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept => params:[m, b]
def compute_error( params, inputs ):
	totalError = 0
	for i in range( 0, len( inputs ) ):
		x = inputs[i, 0]
		y = inputs[i, 1]
		totalError += ( y - ( np.dot( params, np.array( [x, 1.0] ) ) ) ) ** 2
	return totalError / float( len( points ) )


if __name__ == '__main__':
	points = np.genfromtxt( "data.csv", delimiter="," )
	np.random.shuffle( points )

	# Training parameters.
	batch_size = 20
	num_epochs = 25
	num_batches = int( np.ceil( len( points ) / batch_size ) )
	step_size = 0.1

	def batch_indices( i ):
		idx = i % num_batches
		return slice( idx * batch_size, (idx + 1) * batch_size )

	# Define training objective
	def objective( params, i ):
		idx = batch_indices( i )
		return compute_error( params, points[idx] )

	# Get gradient of objective using autograd.
	objective_grad = autograd.grad( objective )

	print( "     Epoch     |    Train accuracy  " )
	def print_perf( params, i, gradient ):
		if i % num_batches == 0:
			train_acc = compute_error( params, points )
			print( "{:15}|{:20}".format( i // num_batches, train_acc ) )


	# The optimizers provided can optimize lists, tuples, or dicts of parameters.
	optimized_params = adam( objective_grad, np.zeros( 2 ), step_size=step_size,
							 num_iters=num_epochs * num_batches, callback=print_perf )
	print_perf( optimized_params, 0, None )

	plt.plot( points[:,0], points[:,1], "ro" )		# Plot points versus approximated line.
	minx = np.min( points[:,0] )
	maxx = np.max( points[:,0] )
	miny = np.dot( optimized_params, np.array([minx, 1.0]) )
	maxy = np.dot( optimized_params, np.array([maxx, 1.0]) )
	plt.plot( [minx, maxx], [miny, maxy] )
	plt.xlabel( "x" )
	plt.ylabel( "y" )
	plt.show()
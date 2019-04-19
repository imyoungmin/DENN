"""
Solving an ODE using a perceptron of 1 hidden layer.
dy/dx + 1/5*y = exp(-x/5)*cos(x)
Subject to y(0) = 0, and 0 <= x <= 2.
The analytical solution is y_a = exp(-x/5)*sin(x)
The trial solution is y_t = x*N, where N = N(x, params) is a MLP.
"""

import autograd.numpy as np
import autograd
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt


def sigmoid( x ):
	"""
	Sigmoid function.
	:param x: Maybe a vector.
	:return: Element-wise sigmoid function of input vector x.
	"""
	return 1.0 / ( 1.0 + np.exp( -x ) )


def sigmoidPrime( x ):
	"""
	Derivative of sigmoid function.
	:param x: Input vector.
	:return: sigmoid'(x)
	"""
	return sigmoid( x ) * ( 1.0 - sigmoid( x ) )


def nnet( x, params ):
	"""
	Compute output from neural network.
	:param x: Input value.
	:param params: List [(W,b),(V,)], where W is an m-element vector (w_i = weight from unique input to neuron i), b is an m-element bias vector, and V is an m-element weight vector.
	:return: Scalar evaluation of neural network.
	"""
	W = params[0][0]					# Weights towards hidden layer.
	b = params[0][1]					# Biases for hidden layer units.
	V = params[1][0]					# Weights towards output layer.
	z = W * x + b
	sigma = sigmoid( z )				# Become inputs for output layer neuron.
	return np.dot( V, sigma )			# Output neuron is linear.


def dNnet_dx( x, params ):
	"""
	Derivate of the network with respect to the unique input value.
	:param x: Input value.
	:param params: Network parameters (cfr. nnet(.)).
	:return: N_g evaluated at x.
	"""
	W = params[0][0]
	b = params[0][1]
	V = params[1][0]
	z = W * x + b
	sigmaPrime = sigmoidPrime( z )
	return np.dot( V * W, sigmaPrime )


def phi_a( x ):
	"""
	Analytical solution.
	:param x: Input value.
	:return: e^{-x/5}\sin(x)
	"""
	return np.exp( -x / 5.0 ) * np.sin( x )


def phi_t( x, params ):
	"""
	Trial function.
	:param x: Input value.
	:param params: Neural network params (cfr. nnet(.)).
	:return: Approximation to ODE solution.
	"""
	return x * nnet( x, params )


def dPhi_t_dx( x, params ):
	"""
	Derivative of the trial function with respect to the input value
	:param x: Input value.
	:param params: Neural network params (cfr. nnet(.)).
	:return: Derivative of phi_t w.r.t x evaluated at current input value.
	"""
	return nnet( x, params ) + x * dNnet_dx( x, params )


def error( inputs, params ):
	"""
	Compute the average of the squared error.
	:param inputs: Array of input values.
	:param params: Neural network parameters.
	:return: Avg squared error.
	"""
	totalError = 0
	for i in range( 0, len( inputs ) ):
		x = inputs[i]
		totalError += ( dPhi_t_dx( x, params ) + 0.2 * phi_t( x, params ) - np.exp( -x / 5.0 ) * np.cos( x ) ) ** 2
	return totalError / float( len( points ) )


if __name__ == '__main__':
	np.random.seed( 0 )
	N = 17										# Number of training samples.
	MinX = 0
	MaxX = 2
	points = np.random.uniform( MinX, MaxX, N )	# Training dataset.

	# Training parameters.
	H = 6  										# Number of neurons in hidden layer.
	batch_size = 2
	num_epochs = 30
	num_batches = int( np.ceil( len( points ) / batch_size ) )
	step_size = 0.05

	def initParams():
		return [(np.random.rand(H), np.random.rand(H)), (np.random.rand(H),)]

	def batchIndices( i ):
		idx = i % num_batches
		return slice( idx * batch_size, (idx + 1) * batch_size )

	# Define training objective
	def objective( params, i ):
		idx = batchIndices( i )
		return error( points[idx], params )

	# Get gradient of objective using autograd.
	objective_grad = autograd.grad( objective )

	print( "     Epoch     |    Train accuracy  " )
	def printPerf( params, i, _ ):
		if i % num_batches == 0:
			train_acc = error( points, params )
			print( "{:15}|{:20}".format( i // num_batches, train_acc ) )


	# The optimizers provided can optimize lists, tuples, or dicts of parameters.
	optimizedParams = adam( objective_grad, initParams(), step_size=step_size,
							 num_iters=num_epochs * num_batches, callback=printPerf )
	printPerf( optimizedParams, 0, None )

	# Plot analytical solution versus trial solution.
	plt.figure( 1 )
	p = np.linspace( MinX, MaxX, 100 )
	y_a = phi_a( p )
	plt.plot( p, y_a, "r-", label="Analytical solution" )
	y_t = []
	for x in p:
		y_t += [phi_t( x, optimizedParams )]
	plt.plot( p, y_t, "b-", label="Trial solution" )
	plt.plot( [np.min( points ), np.min( points )], [0, 0.8], "g-" )
	plt.plot( [np.max( points ), np.max( points )], [0, 0.8], "g-" )
	plt.xlabel( "x" )
	plt.ylabel( r"$\phi(x)$" )
	plt.legend()
	plt.title( r"Approximating solution to $\frac{d\phi}{dx} + \frac{1}{5}\phi = e^{-x/5}\cos(x)$, $\phi(0) = 0$" )
	plt.show()

	# Plot error.
	plt.figure( 2 )
	plt.plot( p, np.abs( y_a - y_t ) )
	plt.xlabel( "x" )
	plt.ylabel( r"$|\phi_a(x) - \phi_t(x)|$" )
	plt.title( "Error" )
	plt.show()
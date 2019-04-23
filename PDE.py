"""
Solving the Laplace equation using a perceptron of 1 hidden layer.
\frac{d^2y}{dx_1^2} + \frac{d^2y}{dx_2^2} = 0
Subject to y(0,x_2) = y(1,x_2) = y(x_1,0) = 0, y(x_1,1) = \sin(\pi x_1)
The analytical solution is y_a = \frac{1}{e^\pi - e^{-\pi}}\sin(\pi x_1)(e^{\pi x_2} - e^{-\pi x_2})
The trial solution is y_t = x_2 \sin(\pi x_1) + x_1(1 - x_1)x_2(1 - x_2)N, where N = N(x, params) is a MLP, and x = [x_1, x_2]^T.
"""

import autograd.numpy as np
import autograd
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt


def sigmoid( z ):
	"""
	Sigmoid function.
	:param z: Input vector of values.
	:return: Element-wise sigmoid function of input vector z.
	"""
	return 1.0 / ( 1.0 + np.exp( -z ) )


def sigmoidPrime( z ):
	"""
	Derivative of sigmoid function.
	:param z: Input vector of values.
	:return: Element-wise sigmoid'(z).
	"""
	return sigmoid( z ) * ( 1.0 - sigmoid( z ) )


def sigmoidPrimePrime( z ):
	"""
	Second derivative of sigmoid function.
	:param z: Input vector of values.
	:return: Element-wise sigmoid''(z).
	"""
	return sigmoid( z ) * ( 1.0 - sigmoid( z ) ) * ( 1.0 - 2.0 * sigmoid( z ) )


def nnet( x, params ):
	"""
	Compute output from neural network.
	:param x: Input vector with two coordinates: (x_1, x_2).
	:param params: List [(W,b),(V,)], where W is an Hx2 matrix (w_ij = weight from x_j input to neuron i), b is an H-element bias vector, and V is an H-element weight vector.
	:return: Scalar evaluation of neural network.
	"""
	W = params[0][0]					# Weights towards hidden layer.
	b = params[0][1]					# Biases for hidden layer units.
	V = params[1][0]					# Weights towards output layer.
	z = np.dot( W, x ) + b
	sigma = sigmoid( z )				# Become inputs for output layer neuron.
	return np.dot( V, sigma )			# Output neuron is linear.


def dkNnet_dxjk( x, params, j, k ):
	"""
	Compute the kth partial derivate of the nnet with respect to the jth input value.
	:param x: Input vector with two coordinates: (x_1, x_2).
	:param params: Network parameters (cfr. nnet(.)).
	:param j: Input coordinate with respect to which we need the partial derivative (0: x_1, 1: x_2).
	:param k: Partial derivative order (1 or 2 supported).
	:return: \frac{d^kN}{dx_j^k} evaluated at x = (x_1, x_2).
	"""
	W = params[0][0]
	b = params[0][1]
	V = params[1][0]
	z = np.dot( W, x ) + b
	if k == 1:
		sigmaPrime = sigmoidPrime( z )
	else:
		sigmaPrime = sigmoidPrimePrime( z )
	return np.dot( V * (W[:,j] ** k), sigmaPrime )


def phi_a( x ):
	"""
	Analytical solution.
	:param x: Input value vector (x_1, x_2).
	:return: \frac{1}{e^\pi - e^{-\pi}}\sin(\pi x_1)(e^{\pi x_2} - e^{-\pi x_2})
	"""
	return 1.0 / ( np.exp( np.pi ) - np.exp( -np.pi ) ) \
		   * np.sin( np.pi * x[0] ) \
		   * ( np.exp( np.pi * x[1] ) - np.exp( -np.pi * x[1] ) )


def phi_t( x, params ):
	"""
	Trial function.
	:param x: Input value vector (x_1, x_2).
	:param params: Neural network params (cfr. nnet(.)).
	:return: Approximation to ODE solution.
	"""
	return x[1] * np.sin( np.pi * x[0] ) + x[0] * ( 1.0 - x[0] ) * x[1] * ( 1.0 - x[1] ) * nnet( x, params )


def d2Phi_t_dx12( x, params ):
	"""
	Second partial derivative of the trial function with respect to x_1 evaluated at x = (x_1, x_2).
	:param x: Input value vector (x_1, x_2).
	:param params: Neural network params (cfr. nnet(.)).
	:return: \frac{d^2\phi_t}{dx_1^2} evaluated at x = (x_1, x_2).
	"""
	return -( np.pi ** 2 ) * x[1] * np.sin( np.pi * x[0] ) + x[1] * ( 1.0 - x[1] ) \
		   * ( x[0] * ( 1.0 - x[0] ) * dkNnet_dxjk( x, params, 0, 2 ) + 2.0 * ( 1.0 - 2.0 * x[0] ) * dkNnet_dxjk( x, params, 0, 1 ) - 2.0 * nnet( x, params ) )


# TODO: d2Phi_t_dx22


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
	for xp in p:
		y_t += [phi_t( xp, optimizedParams )]
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
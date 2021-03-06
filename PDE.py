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
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


def d2Phi_a_dx12( x ):
	"""
	Second partial derivative of the analytical function with respect to x_1 evaluated at x = (x_1, x_2).
	:param x: Input value vector (x_1, x_2).
	:return: \frac{d^2\phi_a}{dx_1^2} evaluated at x = (x_1, x_2).
	"""
	return -( np.pi ** 2 ) / ( np.exp( np.pi ) - np.exp( -np.pi ) ) * np.sin( np.pi * x[0] ) * ( np.exp( np.pi * x[1] ) - np.exp( -np.pi * x[1] ) )


def d2Phi_a_dx22( x ):
	"""
	Second partial derivative of the analytica function with respect to x_2 evaluated at x = (x_1, x_2).
	:param x: Input value vector (x_1, x_2).
	:return: \frac{d^2\phi_a}{dx_2^2} evaluated at x = (x_1, x_2).
	"""
	return ( np.pi ** 2 ) / ( np.exp( np.pi ) - np.exp( -np.pi ) ) * np.sin( np.pi * x[0] ) * ( np.exp( np.pi * x[1] ) - np.exp( -np.pi * x[1] ) )


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


def d2Phi_t_dx22( x, params ):
	"""
	Second partial derivative of the trial function with respect to x_2 evaluated at x = (x_1, x_2).
	:param x: Input value vector (x_1, x_2).
	:param params: Neural network params (cfr. nnet(.)).
	:return: \frac{d^2\phi_t}{dx_2^2} evaluated at x = (x_1, x_2).
	"""
	return x[0] * ( 1.0 - x[0] ) \
		   * ( x[1] * ( 1.0 - x[1] ) * dkNnet_dxjk( x, params, 1, 2 ) + 2.0 * ( 1.0 - 2.0 * x[1] ) * dkNnet_dxjk( x, params, 1, 1 ) - 2.0 * nnet( x, params ) )


def error( inputs, params ):
	"""
	Compute the average of the squared error.
	:param inputs: List of input vector values.
	:param params: Neural network parameters.
	:return: Avg squared error.
	"""
	totalError = 0
	for x in inputs:
		totalError += ( d2Phi_t_dx12( x, params ) + d2Phi_t_dx22( x, params ) ) ** 2
	return totalError / float( len( points ) )


def plotSurface( XX, YY, ZZ, title, zLabel=r"$\phi$" ):
	"""
	Plot a 3D surface.
	:param XX: Grid of x coordinates.
	:param YY: Grid of y coordinates.
	:param ZZ: Grid of z values.
	:param title: Figure title.
	:param zLabel: Label for z-axis.
	"""
	fig1 = plt.figure()
	ax = fig1.gca( projection='3d' )
	surf = ax.plot_surface( XX, YY, ZZ, cmap=cm.jet, linewidth=0, antialiased=False, rstride=1, cstride=1 )
	fig1.colorbar( surf, shrink=0.5, aspect=5 )
	ax.set_xlabel( r"$x_1$" )
	ax.set_ylabel( r"$x_2$" )
	ax.set_zlabel( zLabel )
	plt.title( title )
	plt.show()


def plotHeatmap( Z, title ):
	"""
	Plot a heatmap of a rectangular matrix.
	:param Z: An m-by-n matrix to plot.
	:param title: Figure title.
	:return:
	"""
	fig1 = plt.figure()
	im = plt.imshow( Z, cmap=cm.jet, extent=(0, 1, 0, 1), interpolation='bilinear' )
	fig1.colorbar( im )
	plt.title( title )
	plt.xlabel( r"$x_1$" )
	plt.ylabel( r"$x_2$" )
	plt.show()


if __name__ == '__main__':
	np.random.seed( 31 )
	random.seed( 11 )
	N = 7									# Number of training samples per dimension.
	MinVal = 0
	MaxVal = 1
	points = []
	for ii in range( N + 1 ):
		for jj in range( N + 1 ):
			points.append( np.array( [ii/N, jj/N] ) )		# Training dataset.
	random.shuffle( points )

	# Training parameters.
	H = 7  									# Number of neurons in hidden layer.
	batch_size = 4
	num_epochs = 50
	num_batches = int( np.ceil( len( points ) / batch_size ) )
	step_size = 0.055

	def initParams():
		return [(np.random.uniform( -1, +1, (H, 2) ), np.random.uniform( -1, +1, H )), (np.random.uniform( -1, +1, H ),)]

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

	# Get maximum error.
	maxError = 0
	for xp in points:
		y_a = phi_a( xp )
		y_t = phi_t( xp, optimizedParams )
		error = np.abs( y_a - y_t )
		if error > maxError:
			maxError = error
	print( "Max error:", maxError )

	# Plot solutions.

	# Make data.
	X = np.linspace( MinVal, MaxVal, 100 )
	Y = np.linspace( MinVal, MaxVal, 100 )
	X, Y = np.meshgrid( X, Y )
	Z_a = np.zeros( X.shape )			# Analytic solution.
	Z_t = np.zeros( X.shape )			# Trial solution with neural network.
	Z_e = np.zeros( X.shape )			# Error surface.
	d2Z_a_x12 = np.zeros( X.shape )		# Second derivatives of analytic solution.
	d2Z_a_x22 = np.zeros( X.shape )
	d2Z_t_x12 = np.zeros( X.shape )		# Second derivatives of trial solution.
	d2Z_t_x22 = np.zeros( X.shape )
	d2Z_e_x12 = np.zeros( X.shape )		# Error surface for second derivatives with respect to x_1.
	d2Z_e_x22 = np.zeros( X.shape )		# Error surface for second derivatives with respect to x_2.
	for ii in range( X.shape[0] ):
		for jj in range( X.shape[1] ):
			v = np.array( [X[ii][jj], Y[ii][jj]] )				# Input values.
			Z_a[ii][jj] = phi_a( v )							# Evalutating solutions.
			Z_t[ii][jj] = phi_t( v, optimizedParams )
			Z_e[ii][jj] = np.abs( Z_a[ii][jj] - Z_t[ii][jj] )

			d2Z_a_x12[ii][jj] = d2Phi_a_dx12( v )				# Evaluating derivatives.
			d2Z_a_x22[ii][jj] = d2Phi_a_dx22( v )
			d2Z_t_x12[ii][jj] = d2Phi_t_dx12( v, optimizedParams )
			d2Z_t_x22[ii][jj] = d2Phi_t_dx22( v, optimizedParams )
			d2Z_e_x12[ii][jj] = np.abs( d2Z_a_x12[ii][jj] - d2Z_t_x12[ii][jj] )
			d2Z_e_x22[ii][jj] = np.abs( d2Z_a_x22[ii][jj] - d2Z_t_x22[ii][jj] )

	# Plotting solutions.
	plotSurface( X, Y, Z_a, r"Analytic solution $\phi_a(\mathbf{x})$" )
	plotSurface( X, Y, Z_t, r"Trial solution $\phi_t(\mathbf{x}, \mathbf{p})$" )

	# Plotting the error heatmap.
	plotHeatmap( Z_e, r"Error heatmap for $|\phi_a(\mathbf{x}) - \phi_t(\mathbf{x})|$" )

	# Plotting second derivatives.
	plotSurface( X, Y, d2Z_a_x12, r"Second derivative of analytical solution with respect to $x_1$", r"$\phi''_a$" )
	plotSurface( X, Y, d2Z_a_x22, r"Second derivative of analytical solution with respect to $x_2$", r"$\phi''_a$" )
	plotSurface( X, Y, d2Z_t_x12, r"Second derivative of trial solution with respect to $x_1$", r"$\phi''_t$" )
	plotSurface( X, Y, d2Z_t_x22, r"Second derivative of trial solution with respect to $x_2$", r"$\phi''_t$" )

	# Plotting the error heatmap for second derivatives.
	plotHeatmap( d2Z_e_x12, r"Error heatmap for second derivatives with respect to $x_1$" )
	plotHeatmap( d2Z_e_x22, r"Error heatmap for second derivatives with respect to $x_2$" )
import numpy as np #the only library to be used
import math


# X is our assumed input. 
X
#Y is our assumed output.
Y


# shape_X = X.shape                          the shape of X
# shape_Y = Y.shape                          the shape of Y

# other activations functions can also be used like tanh , Relu, Leaky Relu etc.  

""" for Forward propogotion""" 
#Sigmoid Function # can also be used in Math library 
def sigmoid (x):
return 1/(1 + np.exp(-x))

""" for back propogotion""" 
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
return x * (1 - x)


#Variable initialization
epoch=2000 #Setting training iterations
alpha=0.1 #Setting learning rate

n_i = X.shape[1] #number of features in data set
n_h = 3 #number of hidden layers neurons
n_o = Y.shape[1] #number of neurons at output layer

#weight and bias initialization
#wh=np.random.uniform(size=(n_i,n_h))  # we are given this weight as  W1 , W2 ,W3
wh = np.array([[W1],[W2],[W3]])
#bh=np.random.uniform(size=(1,n_h))    # we are given this biases as  b1,b2 , b3
bh = np.array([[b1],[b2],[b3]])

#parameters for output Layer

wout=np.random.uniform(size=(n_h,n_o))
bout=np.random.uniform(size=(1,n_o))

for i in range(epoch):

	#Forward Propogation
	h_input1=np.dot(X,wh)       #  vectorized implementation using np.dot
	h_input=h_input1 + bh   # adding the biase
	h_activation = sigmoid(h_input)  #applying the sigmoid activation function


	o_input1=np.dot(h_activation,wout) 
	o_input= o_input1+ bout
	output = sigmoid(o_input)

	#Backpropagation
	E = y-output             #calculating error

	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_sigmoid(h_activation)
	d_output = E * slope_output_layer
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

	#upgradation of the parameters 

	wout += h_activation.T.dot(d_output) *alpha
	bout += np.sum(d_output, axis=0,keepdims=True) *alpha
	wh += X.T.dot(d_hiddenlayer) *alpha
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *alpha

print output

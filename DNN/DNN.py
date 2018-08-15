#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_breast_cancer
from sklearn.model_selection import train_test_split


def init_parameters(layer_dims):
	L = len(layer_dims)
	np.random.seed(3)
	parameters = {}
	for i in range(1, L):
		parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
		parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
	return parameters

def relu(Z):
	return np.maximum(0, Z)

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

def fp(X, parameters):
	A = X
	L = len(parameters) // 2
	caches = [(None, None, None, X)]
	for l in range(1, L):
		A_pre = A
		W = parameters['W' + str(l)]
		b = parameters['b' + str(l)]
		z = np.dot(W, A_pre) + b
		A = relu(z)
		caches.append((W, b, z, A))
	WL = parameters['W' + str(L)]
	bL = parameters['b' + str(L)]
	zL = np.dot(WL, A) + bL
	AL = sigmoid(zL)
	caches.append((WL, bL, zL, AL))
	return AL, caches

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
	cost = np.squeeze(cost)
	return cost

def relu_back(A):
	return np.int64(A > 0)

def bp(AL, Y, caches):
	m = Y.shape[1]
	L = len(caches) - 1
	prev_AL = caches[L - 1][3]
	dzL = 1. / m * (AL - Y)
	dWL = np.dot(dzL, prev_AL.T)
	dbL = np.sum(dzL, axis = 1, keepdims = True)
	gradients = {'dW' + str(L) : dWL, 'db' + str(L) : dbL}
	for i in reversed(range(1, L)):
		post_W = caches[i + 1][0]
		dz = dzL
		dal = np.dot(post_W.T, dz)
		#Al = caches[i][3] 
		#dzl = np.multiply(dal, relu_back(Al)) 
		#使用Al和zl效果相同

		zl = caches[i][2]
		dzl = np.multiply(dal, relu_back(zl))

		prev_A = caches[i -1][3]
		dwl = np.dot(dzl, prev_A.T)
		dbl = np.sum(dzl, axis = 1, keepdims = True)

		gradients['dW' + str(i)] = dwl
		gradients['db' + str(i)] = dbl
		dzL = dzl
	return gradients

def update_param(parameters, gradients, learning_rate):
	L = len(parameters) // 2
	for i in range(L):
		parameters['W' + str(i + 1)] -= learning_rate * gradients['dW' + str(i + 1)]
		parameters['b' + str(i + 1)] -= learning_rate * gradients['db' + str(i + 1)]
	return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, maxCycles):
	costs = []
	parameters = init_parameters(layer_dims)
	for i in range(maxCycles):
		AL, caches = fp(X, parameters)
		cost = compute_cost(AL, Y)
		if i % 1000 == 0:
			print('Cost after iteration {} : {}'.format(i, cost))
			costs.append(cost)
		gradients = bp(AL, Y, caches)
		parameters = update_param(parameters, gradients, learning_rate)
	plt.clf()
	plt.plot(costs)
	plt.xlabel('iterations')
	plt.ylabel('cost')
	plt.show()
	return parameters

def predict(X_test,y_test,parameters):
	"""
	:param X:
	:param y:
	:param parameters:
	:return:
	"""
	m = y_test.shape[1]
	Y_prediction = np.zeros((1, m))
	prob, caches = fp(X_test,parameters)
	for i in range(prob.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if prob[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0
	accuracy = 1- np.mean(np.abs(Y_prediction - y_test))
	return accuracy

def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.01, num_iterations=47000):
	parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
	accuracy = predict(X_test,y_test,parameters)
	return accuracy

if __name__ == "__main__":
	X_data, y_data = load_breast_cancer(return_X_y=True)
	X_train, X_test,y_train,y_test = train_test_split(X_data, y_data, train_size=0.8,random_state=28)
	X_train = X_train.T
	y_train = y_train.reshape(y_train.shape[0], -1).T
	X_test = X_test.T
	y_test = y_test.reshape(y_test.shape[0], -1).T
	accuracy = DNN(X_train,y_train,X_test,y_test,[X_train.shape[0],20, 20, 10, 5, 1])
	print('accuracy reaches %.4f' % accuracy)

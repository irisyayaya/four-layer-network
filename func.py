# -*- coding: utf-8 -*-

import sys
import numpy as np
import math
def read_price(filename):
	try:
		f = open(filename, 'r')
	except IOError:
		print ("Cannot open file %s\n" % filename)
		sys.exit("bye")
	
	# read data
	data = f.readlines()
	f.close()
	
	line0 = data[0].split()
	#print line0
	
	if len(line0) == 0:
		sys.exit("empty first line")
	
	n = int(line0[1])
	T = int(line0[3])
	
	print("n = ", n, ", T = ", T)
	
	matrix_raw = np.zeros((n,T))

	for i in range(n):
		#read line i + 2
		theline = data[i+2].split()
		#print i, " -> ", theline
		for j in range(T):
			if theline[j] == 'NA':
				valueij = np.nan
			else:
				valueij = float(theline[j])
			#print i, j, numberij
			matrix_raw[i][j] = valueij
	return matrix_raw
	
def price_to_return(price_matrix):
	T = price_matrix.shape[1]
	ret_matrix = price_matrix[:,1:T] / price_matrix[:,0:T-1] - 1
	return ret_matrix
	
def train_test_split(data, offset=10):
	n,T = data.shape

	T_train = math.ceil(T/2)
	
	x_train = np.copy(data[:, 0:T_train-offset]) # 0 to T_train-11
	y_train = np.copy(data[:, offset:T_train]) # 10 to T_train-1
	x_test = np.copy(data[:,T_train:T-offset]) # T_train to T-11
	y_test = np.copy(data[:, T_train+offset:T]) # T_train+10 to T-1
	
	return x_train, y_train, x_test, y_test
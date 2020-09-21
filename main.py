# -*- coding: utf-8 -*-

import sys
import numpy as np

from dnn import dnn
from func import read_price, price_to_return, train_test_split



if __name__ == "__main__":
	if len(sys.argv) != 2:  # the program name and the datafile
		# stop the program and print an error message
		sys.exit("usage: eigen.py datafile ")
	
	filename = sys.argv[1]
	print("input", sys.argv[1])
	

	price_mat = read_price(filename)
	ret_mat = price_to_return(price_mat)
	n,T = ret_mat.shape

	x_train, y_train, x_test, y_test = train_test_split(ret_mat)
	
	num_layer = 3 # hiddenlayer = 2; plus output layer = 3
	d = 50
	print("Training with num_layer:", num_layer, 
	       ", number of nodes per layer:", d)
	
	clf = dnn(num_layer,d)
	clf.fit(x_train,y_train)
	y_train_hat = clf.predict(x_train)
	y_train_err = y_train - y_train_hat
	print("in sample avg loss:", np.mean(y_train_err**2))
	
	y_test_hat = clf.predict(x_test)
	y_test_err = y_test - y_test_hat
	print("generalization avg loss:", np.mean(y_test_err**2))
	
import math
import random
import numpy as np
from numpy import random
import pylab



# A function to generate the halfmoon data
# where Input:
#         rad  - central radius of the half moon
#        width - width of the half moon
#            d - distance between two half moon
#       n_samp - total number of the samples
#       Output:
#         data - output data
#data_shuffled - shuffled data
# For example
# halfmoon(10,2,0,1000) will generate 1000 data of 
# two half moons with radius [9-11] and space 0.
def halfmoon(rad,width,d,n_samp):

	if rad < width/2:
		print('The radius should be at least larger than half the width')

	if n_samp%2!=0 :
		print('Please make sure the number of samples is even')
	
	aa    = random.random(size=(2,n_samp/2))
	radius= (rad-width/2) + width*aa[0,:]
	theta = math.pi*aa[1,:]

	x     = radius * np.cos(theta)
	y     = radius * np.sin(theta)
	label = 1*np.ones([1,len(x)])   # label for Class 1n

 	x1    = radius * np.cos(-theta) + rad
	y1    = radius * np.sin(-theta) - d
	label1= -1*np.ones([1,len(x)])  # label for Class 2

	data  = np.vstack(( np.hstack((x, x1)), 
						np.hstack((y, y1)),
						np.hstack((label, label1))
						))
     
	[n_row, n_col] = data.shape
	shuffle_seq = np.random.permutation(n_col)
	data_shuffled = np.zeros((n_row,n_col))

	for i in range(0,n_col):
		data_shuffled[:,i] = data[:,shuffle_seq[i]];

	return [data, data_shuffled]


# annealing - anneal data from 'start' to 'end' with number 'num'
# out = annealing(start,end,num)
# start_data - starting point
# end_data   - ending point
# num        - number of annealing point
# out        - annealed data sequence
def annealing(start_data,end_data,num):
	# Check input parameters
	if start_data == end_data:
		print('Starting point and ending point is the same.')

	if num <= 2:
		print('Number of annealed data point should > = 2.')

	# Linear annealing
	#step = (end_data - start_data)/(num-1);
	out = np.linspace(start_data,end_data,num)
	return out


# a signsum function 
def mysign(d):
#	print 'd=',d[0][0]
	out = 1*(d>=0) + (-1)*(d<0)

	# if distance(d,1) <= distance(d,0):
	#     	out = 1;
	# else
	#     out = 0;

	return out



def perceptron():

	#================== Step 0: Generating halfmoon data =====================
	#load data_shuffled1.mat; shuffled data for half-moon with distance d = 1
	rad    = 10   # central radius of the half moon
	width  = 6    # width of the half moon
	dist   = 0   # distance between two half moons
	num_tr = 1000   # number of training sets
	num_te = 0	  # number of testing sets
	num_samp = num_tr+num_te	# number of samples
	epochs = 50
	print('Perceptron for Classification\n')
	print('_________________________________________\n')
	print('Generating halfmoon data ...\n')
	print('  ------------------------------------\n')
	print('  Points generated: ',num_samp)
	print('  Halfmoon radius : ',rad)
	print('  Halfmoon width  : ',width)
	print('      Distance    : ',dist)
	print('  Number of epochs: ',epochs)
	print('  ------------------------------------\n')

	[data, data_shuffled] = halfmoon(rad,width,dist,num_samp);
#	print('data=',data)

	##============= Step 1: Initialization of Perceptron network ==============
	num_in 	= 2   		# number of input neuron
	b    	= dist/2  	# bias
	err    	= 0    		# a counter to denote the number of error outputs
	#eta  	= 0.95 		# learning rate parameter
	eta 	= annealing(0.9,1E-5,num_tr)
	w    	= np.vstack((b,np.zeros([num_in,1])))	# initial weights

	##=========================== Main Loop ===================================
	## Step 2,3: activation and actual response
	# st = cputime
	print('Training the perceptron using LMS ...\n')
	print('  ------------------------------------\n')
	ee 	= 0*np.zeros(num_tr)
	mse = 0*np.zeros(epochs) 
	for epoch in range(0,epochs):
		shuffle_seq = np.random.permutation(num_tr)
		data_shuffled_tr = data_shuffled[:,shuffle_seq]
#		print('data_shuffle_tr=',data_shuffled_tr)
#		print('data_shuffled_tr[0:2,i]=', data_shuffled_tr[0:2,0:1])
		for i in range( 0,num_tr):
			x = np.vstack((1,data_shuffled_tr[0,i],data_shuffled_tr[1,i])) # fetching data from database
			d = data_shuffled_tr[2,i]   			# fetching desired response from database
			y = mysign(np.dot(w.T,x)[0][0])
#			print 'w*x=', np.dot(w.T,x)[0][0]
			ee[i] = d-y;
#			if d!=y:
#				print 'd-y=', d-y

			## Step 4: update of weight
			w_new = w + eta[i]*(d-y)*x;
			#     if w_new == w, % Stop criteria
			#         break;
			w = w_new
		mse[epoch] = np.mean(np.square(ee))
#		print 'sum of ee=', np.mean(np.square(ee)), 'mse[',epoch,']=',mse[epoch]

	print('  Points trained : %d\n',num_tr)
	print('  w is ', w)
	#print('       Time cost : %4.2f seconds\n',cputime - st)
	print('  ------------------------------------\n')


	##================= Colormaping the figure here ===========================
	##=== In order to avoid the display problem of eps file in LaTeX. =========
	xmin = min(data_shuffled[0,:])
	xmax = max(data_shuffled[0,:])
	ymin = min(data_shuffled[1,:])
	ymax = max(data_shuffled[1,:])
	[x_b,y_b]= pylab.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
	z_b  = 0*np.ones(x_b.shape);
	for x1 in range(0,x_b.shape[0]):
		for y1 in range(0, x_b.shape[1]):
			input_data = np.vstack((1, x_b[x1][y1], y_b[x1][y1]))
			z_b[x1][y1] = np.dot(w.T,input_data)

#	print z_b

	pylab.figure()
	for i in range (0,data_shuffled_tr.shape[1]):
		if data_shuffled_tr[2][i] > 0 :
			pylab.plot(data_shuffled_tr[0][i],data_shuffled_tr[1][i],'x')
		if data_shuffled_tr[2][i] < 0 :
			pylab.plot(data_shuffled_tr[0][i],data_shuffled_tr[1][i],'o')
	

	pylab.contour(x_b,y_b,z_b,0,colors='k', Linewidth=0.5)


	pylab.figure()
	pylab.plot(mse,'k');
	pylab.title('Learning curve');
	pylab.xlabel('Number of epochs');
	pylab.ylabel('MSE');
	
	
	pylab.show()



def main(_):
	perceptron()

if __name__=='__main__':
	perceptron()
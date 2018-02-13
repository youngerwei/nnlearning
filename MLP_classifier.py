# coding=utf-8
import math
import random
import numpy as np
from numpy import random
import pylab
import warnings

#  这是在linux上，哦不，是在Windows上修改的。
# From Home

warnings.filterwarnings("ignore")


# A function to generate the halfmoon data
# where Input:
#         rad  - central radius of the half moon
#        width - width of the half moon
#            d - distance between two half moon
#       n_samp - total number of the samples
#       Output:
#         data - output data
# data_shuffled - shuffled data
# For example
# halfmoon(10,2,0,1000) will generate 1000 data of 
# two half moons with radius [9-11] and space 0.
def halfmoon(rad, width, d, n_samp):
    if rad < width / 2:
        print('The radius should be at least larger than half the width')

    if n_samp % 2 != 0:
        print('Please make sure the number of samples is even')

    aa = random.random(size=(2, n_samp / 2))
    radius = (rad - width / 2) + width * aa[0, :]
    theta = math.pi * aa[1, :]

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    label = 1 * np.ones([1, len(x)])  # label for Class 1n

    x1 = radius * np.cos(-theta) + rad
    y1 = radius * np.sin(-theta) - d
    label1 = -1 * np.ones([1, len(x)])  # label for Class 2

    data = np.mat(np.vstack((np.hstack((x, x1)),
                             np.hstack((y, y1)),
                             np.hstack((label, label1))
                             )))
    [n_row, n_col] = data.shape
    shuffle_seq = np.random.permutation(n_col)
    data_shuffled = np.mat(np.zeros((n_row, n_col)))

    for i in range(0, n_col):
        data_shuffled[:, i] = data[:, shuffle_seq[i]];

    return [data, data_shuffled]


# annealing - anneal data from 'start' to 'end' with number 'num'
# out = annealing(start,end,num)
# start_data - starting point
# end_data   - ending point
# num        - number of annealing point
# out        - annealed data sequence
def annealing(start_data, end_data, num):
    # Check input parameters
    if start_data == end_data:
        print('Starting point and ending point is the same.')

    if num <= 2:
        print('Number of annealed data point should > = 2.')

    # Linear annealing
    # step = (end_data - start_data)/(num-1);
    out = np.linspace(start_data, end_data, num)
    return out


# y = hyperb (x)
# hyperbolic function
# x - input data
# y - output data
def hyperb(x):
    y = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    # y = 1./(1+exp(-x));
    return y


# y = d_hyperb (x)
# differentiation of hyperbolic function
# x - input data
# y - output data
def d_hyperb(x):
    y = (4 * np.exp(2 * x)) / (np.square(1 + np.exp(2 * x)))

    # y = 1./(1+exp(-x));
    return y


# a signsum function
def mysign(d):
    out = 1 * (d >= 0) + (-1) * (d < 0)

    # if distance(d,1) <= distance(d,0),
    #     out = 1;
    # else
    #     out = 0;
    # end

    return out


def MLP_classifier():
    ##=========== Step 0: Generating halfmoon data ============================
    rad = 10  # central radius of the half moon
    width = 6  # width of the half moon
    dist = -7  # distance between two half moons
    num_tr = 1000  # number of training sets
    num_te = 2000  # number of testing sets
    num_samp = num_tr + num_te  # number of samples

    print('Multiple Layer Perceptron for Classification\n')
    print('_________________________________________\n')
    print('Generating halfmoon data ...\n')
    print('  ------------------------------------\n')
    print('  Points generated: %d' % (num_samp))
    print('  Halfmoon radius : %f' % (rad))
    print('  Halfmoon width  : %d' % (width))
    print('      Distance    : %d' % (dist))
    print('  ------------------------------------\n')

    [data, data_shuffled] = halfmoon(rad, width, dist, num_samp)

    ##========== Step 1: Initialization of Multilayer Perceptron (MLP) ========
    print('Initializing the MLP ...\n');
    n_in = 2  # number of input neuron
    n_hd = 20  # number of hidden neurons
    n_ou = 1;  # number of output neuron
    # w = np.zeros((2,1))
    # w1{1} = rand(n_hd,n_in+1)./2  - 0.25; # initial weights of dim: n_hd x n_in between input layer to hidden layer
    w1_1 = np.random.rand(n_hd, n_in + 1)
    dw0_1 = np.zeros((n_hd, n_in + 1))  # rand(n_hd,n_in)./2  - 0.25;%
    # w1{2} = rand(n_ou,n_hd+1)./2  - 0.25; # initial weights of dim: n_ou x n_hd between hidden layer to output layer
    w1_2 = np.random.rand(n_ou, n_hd + 1)
    dw0_2 = np.zeros((n_ou, n_hd + 1))  # rand(n_ou,n_hd)./2  - 0.25;%
    num_Epoch = 50  # number of epochs
    mse_thres = 1E-3  # MSE threshold
    mse_train = np.inf  # MSE for training data
    epoch = 1
    alpha = 0  # momentum constant
    err = 0  # a counter to denote the number of error outputs
    # eta2  = 0.1;         			# learning-rate for output weights
    # eta1  = 0.1;          			# learning-rate for hidden weights
    eta1 = annealing(0.1, 1E-5, num_Epoch)
    eta2 = annealing(0.1, 1E-5, num_Epoch)

    ##========= Preprocess the input data : remove mean and normalize =========
    mean1 = np.vstack((np.mean(data[0:2, :], 1), 0))
    nor_data = np.mat(np.zeros(data_shuffled.shape))
    for i in range(0, num_samp):
        nor_data[:, i] = data_shuffled[:, i] - mean1
    max1 = np.vstack((np.max(np.abs(nor_data[0:2, :]), 1), 1))
    for i in range(0, num_samp):
        nor_data[:, i] = nor_data[:, i] / max1

    ##======================= Main Loop for Training ==========================
    print('Training the MLP using back-propagation ...\n')
    print('  ------------------------------------\n')
    nor_data1 = np.zeros(data_shuffled.shape)
    e = np.zeros(num_tr)
    mse = np.zeros(num_Epoch)
    while mse_train > mse_thres and epoch <= num_Epoch - 1:
        # print('   Epoch #: %d -> '%(epoch))
        ## shuffle the training data for every epoch
        [n_row, n_col] = nor_data.shape
        shuffle_seq = np.random.permutation(num_tr)
        nor_data1 = nor_data[:, shuffle_seq]

        ## using all data for training for this epoch
        for i in range(0, num_tr):
            ## Forward computation
            x = np.vstack((nor_data1[0:2, i], 1))  # fetching input data from database
            # d  = myint2vec(nor_data1(3,i));% fetching desired response from database
            d = nor_data1[2, i]  # fetching desired response from database
            hd = np.vstack((hyperb(w1_1 * x), 1))  # hidden neurons are nonlinear
            o = hyperb(w1_2 * hd)  # output neuron is nonlinear
            e[i] = d - o

            ## Backward computation
            delta_ou = e[i] * d_hyperb(w1_2 * hd)  # delta for output layer
            delta_hd = np.multiply(d_hyperb(w1_1 * x), (w1_2[:, 0:n_hd].T * delta_ou))  # delta for hidden layer
            dw1_1 = eta1[epoch] * delta_hd * x.T;
            dw1_2 = eta2[epoch] * delta_ou * hd.T;

            ## weights update
            w2_1 = w1_1 + alpha * dw0_1 + dw1_1  # weights input -> hidden
            w2_2 = w1_2 + alpha * dw0_2 + dw1_2  # weights hidden-> output

            ## move weights one-step
            dw0_1 = dw1_1
            dw0_2 = dw1_2
            w1_1 = w2_1
            w1_2 = w2_2
        mse[epoch] = np.sum(np.mean(e.T ** 2))
        mse_train = mse[epoch]
        print('   Epoch #: %03d -> MSE = %f' % (epoch, mse_train))
        epoch = epoch + 1

    print('   Points trained : %d' % (num_tr))
    print('  Epochs conducted: %d' % (epoch - 1))
    print('  ------------------------------------\n')

    ##================= Colormaping the figure here ===========================
    ##=== In order to avoid the display problem of eps file in LaTeX. =========
    a = np.array(data_shuffled[0, :].A[0])
    xmin = min(a)
    xmax = max(a)
    a = np.array(data_shuffled[1, :].A[0])
    ymin = min(a)
    ymax = max(a)
    [x_b, y_b] = pylab.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    z_b = 0 * np.ones(x_b.shape);
    for x1 in range(0, x_b.shape[0]):
        for y1 in range(0, x_b.shape[1]):
            input_data = np.vstack(((x_b[x1][y1] - mean1[0]) / max1[0], (y_b[x1][y1] - mean1[1]) / max1[1], 1))
            hd = np.vstack((hyperb(w1_1 * input_data), 1))
            z_b[x1][y1] = hyperb(w1_2 * hd)

    pylab.figure()
    '''
    for i in range (0,data_shuffled.shape[1]):
        if data_shuffled.A[2][i] > 0 :
            pylab.plot(data_shuffled.A[0][i],data_shuffled.A[1][i],'x')
        if data_shuffled.A[2][i] < 0 :
            pylab.plot(data_shuffled.A[0][i],data_shuffled.A[1][i],'o')
    '''

    ##========================== Testing ======================================
    print('Testing the MLP ...\n')
    oo = np.zeros(num_samp)
    for i in range(num_tr, num_samp):
        x = np.vstack((nor_data[0:2, i], 1))
        hd = np.vstack((hyperb(w1_1 * x), 1))
        oo[i] = hyperb(w1_2 * hd)
        xx = max1.A[0:2, :] * x.A[0:2, :] + mean1.A[0:2, :]
        if oo[i] > 0:  # myvec2int(o(:,i)) == 1,
            pylab.plot(xx[0], xx[1], 'ro')
        if oo[i] < 0:  # myvec2int(o(:,i)) == -1,
            pylab.plot(xx[0], xx[1], 'kx')

    # Calculate testing error rate
    err = 0
    for i in range(num_tr, num_samp):
        if abs(mysign(oo[i]) - nor_data.A[2, i]) > 1E-6:
            err = err + 1.0

    print('  ------------------------------------\n')
    print('   Points tested : %d\n' % (num_te))
    print('    Error points : %d %4.2f%%' % (err, (err / num_te) * 100))
    print('  ------------------------------------\n')

    print('Mission accomplished!\n')
    print('_________________________________________\n')

    pylab.contour(x_b, y_b, z_b, 0, colors='k', Linewidth=0.5)

    pylab.figure()
    pylab.plot(mse, 'k');
    pylab.title('Learning curve');
    pylab.xlabel('Number of epochs');
    pylab.ylabel('MSE');

    pylab.show()


if __name__ == '__main__':
    MLP_classifier()

#ECE 763 Project 1
#Author: Luke Martin
#Date: 2/21/2020

#Multiple Different Training Models for estimating data


import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
from scipy.stats import multivariate_normal
import math
from sklearn.decomposition import PCA
import numpy.matlib
from scipy.special import psi, gammaln
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import scipy

def plotavg(mean, size):
	#Function for recovering the average image from the PCA transform.
	plmean = pca_face.inverse_transform(mean)
	plmean = plmean / np.max(plmean)
	plmean = np.reshape(plmean, size)
	#Normalize the image values before presenting
	a = 200.0 / plmean.shape[1]
	dim = (200, int(plmean.shape[0] * a))
	resize = cv2.resize(plmean, dim)
	resize = resize*(255/np.max(resize))
	cv2.imshow("Resize", resize)
	im = np.array(resize, dtype = 'uint8')
	im = Image.fromarray(im, 'RGB')
	plt.imshow(im)
	plt.show()
	

def plotROC(labels, prob, positions):
	#Use the provided labels and results from the pdf function to plot an ROC curve
	prob = np.diagonal(prob)
	fpr, tpr, thresholds = metrics.roc_curve(labels, prob, positions)
	auc = roc_auc_score(labels, prob)
	#Print the area under the curve, overall accuracy
	print(auc)
	plt.plot(fpr, tpr)
	plt.plot([0, 1], [0, 1])
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()

def plotROC2(labels, prob, positions):
	#Same function as plotROC but it does not take the diagonal of the probabilities, used with t-distributions
	fpr, tpr, thresholds = metrics.roc_curve(labels, prob, positions)
	auc = roc_auc_score(labels, prob)
	print(auc)
	plt.plot(fpr, tpr)
	plt.plot([0, 1], [0, 1])
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
def oneGauss(input, mean, covariance):
	#PDF of a gaussian function. Added noise to avoid division by zero
	diff = (input-mean)
	temp1 = np.dot(np.dot(diff,np.linalg.pinv(covariance)), diff.T)
	return (1 / (0.00000000001 + (2 * np.pi) ** (1 / 2) * np.linalg.det((covariance)) ** 0.5) * np.exp(-0.5 * temp1))
def oneGaussLog(input, mean, covariance):
	#Log of the pdf of a gaussian.
	diff = (input-mean).T
	temp1 = np.dot(np.dot(diff.T,np.linalg.pinv(covariance)), diff)
	return -0.5*np.log(np.linalg.det(covariance)) - 0.5*np.log(2*np.pi) -0.5*(temp1)
def bigGam(input, alpha, beta):
	#Form of gamma function, unused in this project
	temp1 = beta**alpha
	temp2 = gamma(alpha)
	temp3 = np.exp(-beta*input)
	temp4 = input**(alpha-1)
	return temp1 / temp2 * temp3 * temp4
def t_pdf(x, mean, covariance, nu):
	#Returns the pdf of the t-distribution given a  set of input images, their mean, covariance and nu value. Based on the algorithm found in the lecture notes.
	(W,L) = np.shape(x)
	c = np.exp(gammaln((nu+L)/2) - gammaln(nu/2))
	det = np.prod(np.diag(covariance))
	c = c / ((nu*np.pi)**(L/2) * np.sqrt(det))
	delta = np.zeros([W,1])
	diff = (x - np.reshape(mean, (1,-1)))
	temp = np.dot(diff, np.linalg.inv(np.diag(np.diag(covariance))))
	for i in range(W):
		delta[i] = np.dot(np.reshape(temp[i,:], (1,-1)), np.transpose(np.reshape(diff[i,:],(1,-1))))
	px = 1+(delta/nu)
	px = px**((-nu-L)/2)
	px = np.dot(px,c)
	return px

def t_cost(Expect, Expect_log, row):
	#Returns new nu values based on the log liklehood of previously calculated expected values of the hidden variables
	nu = np.arange(1,1000)
	value_test = np.zeros([len(nu)])
	for i in range(len(nu)):
		half = nu[i] / 2
		I = row
		value = I * (half*np.log(half)+gammaln(half))
		value = value - (half-1) * np.sum(Expect_log)
		value = value + half * np.sum(Expect)
		value_test[i] = -1 * value
	a = np.where(value_test == value_test.min())
	return nu[a]

def fitting_t(input, precision):
	#This is the main t-distribution functio that creates the model. It uses an EM algorithm as discussed in class and uses the log of some pdf's to avoid overflow/underflow.
	#Returns mu, sigma and nu values
	(W,D) = np.shape(input)
	data_mean = Average(input)
	mu = data_mean
	#Set up variables and get initial values.
	data_variance = np.zeros([D,D])
	diff = input - data_mean
	for i in range(W):
		mat = np.dot((diff.T),diff)
		data_variance = data_variance+mat
	sigma = (data_variance/W)
	
	nu = 1
	iter = 0
	largeInt = 1000000 #Need big value for t to work
	delta = np.zeros([W,1])
	while True:
		#re-calculate nu values.
		
		#Expectation Step
		diff2 = input - mu
		temp = np.dot(diff2, np.linalg.inv(sigma))
		for i in range(W):
			delta[i] = np.dot(np.reshape(temp[i, :],(1,-1)),np.transpose(np.reshape(diff2[i,:],(1,-1))))
		
		nu_and_delta = nu + delta
		#Expected values of the hidden variable
		Expected = ((nu+D)/ nu_and_delta)
		Expected_log = psi((nu+D)/2)- np.log(nu_and_delta/2) #psi is the log derivative of the gamma function
		Expect_sum = np.sum(Expected)
		Expected_Times = Expected * input
		
		#Maximization Step
		#New mean/average
		mu = np.reshape(np.sum(Expected_Times,axis=0),(1,-1))
		mu = (mu/ Expect_sum)
		
		diff = input - mu
		#new Sigma
		sigma = np.zeros([D,D])
		for i in range(W):
			xtemp = np.reshape(diff[i,:],(1,-1))
			sigma = sigma + (Expected[i] * np.dot((xtemp.T),xtemp))
		simga = sigma / Expect_sum
		(U, UU) = np.shape(Expected)
		#Use log liklihood cost function to get new nu value
		nu = t_cost(Expected, Expected_log, U)
		
		temp = np.dot(diff, np.linalg.inv(sigma))
		
		for i in range(W):
			delta[i] = np.dot(np.reshape(temp[i,:],(1,-1)), np.transpose(np.reshape(diff[i,:],(1,-1))))
			
		(sign, loglike) = np.linalg.slogdet(np.array(sigma))
		L = W * (gammaln((nu+D)/2) - (D/2)*np.log(nu*np.pi) - loglike/2 - gammaln(nu/2))
		s = np.sum(np.log(1+(delta/nu)))/2
		L = L - (nu+D)*s
		iter = iter + 1
		#Repeat a set number of times, in this case 10 but can be changed
		if iter == 10:
			break
		largeInt = L/2
	return(mu,sigma,nu)

def getT(input, dof, mean, covariance):
	#Unused function, t-pdf of sorts
	dim = len(covariance[0])
	input=input.flatten()
	mean = mean.flatten()
	gamma1 = gamma((dof+dim)/2)
	gamma2 = gamma(dof/2)
	diff = (input-mean).T
	temp1 = (np.dot(np.dot(diff.T, np.linalg.inv(covariance)), diff))/dof
	temp2 = (1+temp1)**(-(dof+dim)/2)
	temp3 = (dof*np.pi)**(dim/2)
	temp4 = np.sqrt(np.linalg.det(covariance))
	temp5 = gamma1/(temp3*gamma2)
	total = (temp5/temp4)*(temp2)
	
	return total
def Average(image_list):
	#Returns the average value of a list
	Mean = 0
	for i in (image_list):
		new_iteration = i
		Mean = Mean + new_iteration
	return image_list.mean(axis = 0)
def Covariance(image_list, average):
	#Returns the full covariance matrix of a list given the list and its average/mean
	diff = (image_list - average).T
	cov = (np.dot((diff),(diff).T)) / (len(image_list)-1)
	return cov
def gamma(input):
	#Gamma function
	return math.factorial(input)

def gaussian(X, mu, cov):
	#Returns the gaussian pdf, however this one is currently not used since it does not work
    n = X.shape[1]
    diff = (X - mu).T
    diff = diff.flatten()
    return (1 / ((2 * np.pi) ** (1 / 2) * (np.linalg.det(cov)) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
	
def calculateOneGaussian(face_train_images, nonface_train_images):
	#Get the one gaussian model. Takes two lists and will return their means and covariances.
	face_mean = face_train_images.mean(axis = 0)
	face_mean = Average(face_train_images)
	nonface_mean = nonface_train_images.mean(axis = 0)
	dimensions = np.shape(face_train_images)
	covariance = Covariance(face_train_images, face_mean)
	noncovariance = Covariance(nonface_train_images, nonface_mean)
	return face_mean, nonface_mean, covariance, noncovariance

def mixedGaussian(training_face, gaussian_number):
	#The main function for computing the mixed Gaussian Modeling. Takes a list of images and the number of gaussians wanted and will return a list of means, covariances and weights.
	#Uses the EM algorithm to improve the accuracy of the model.
	#initialize values. 
	iterations = 0
	np.random.seed(0)
	input = (training_face)
	(width, height) = np.shape(training_face)
	LargeNumber = 20000
	number = 0
	covariance = [[]] * gaussian_number
	#This function evenly distributes the weight of lambda based on the number input
	lmbda = np.matlib.repmat(1/gaussian_number, gaussian_number, 1)
	#Begin at random datapoints
	random_ints = np.random.permutation(width)
	random_ints = np.array(random_ints[0:gaussian_number])
	mean_collection = np.zeros([gaussian_number, height])
	#Initialize the mean to random datapoints
	ran = np.random.permutation(width)
	ran = np.array(ran[0:gaussian_number])
	for i in range(len(ran)):
		mean_collection[i,:] = (input[ran[i],:])
	#Compute initial mean and variance
	data_mean = Average(input)	
	covar = np.zeros([height, height])
	for i in range(width):
		tmp = input[i,:] - data_mean
		tmp = tmp.T * tmp
		covar = covar + tmp
	
	covar = covar / width
	for i in range(gaussian_number):
		covariance[i] = covar
	
	while True:
		temp = np.zeros([width, gaussian_number])
		temp2 = np.zeros([width, gaussian_number])
		
		#Expectation Step
		for n in range(gaussian_number):
			temp[:, n] = lmbda[n]*(np.diagonal(oneGauss(input, mean_collection[n, :], covariance[n])))
		
		sum = np.sum(temp, axis = 1)
		#Normalization
		for i in range(width):
			temp2[i, :] = temp[i, :] / sum[i]
		
		#Maximization Step
		r = np.sum(temp2, axis = 0)
		r_all = np.sum(r)
		
		for k in range(gaussian_number):
		#Compute new weights, means and covariances based on the equation used in class.
			lmbda[k] = r[k] / r_all
			new_mean = np.zeros([1, height])
			for i in range(width):
				new_mean = new_mean + (temp2[i,k]*input[i,:])
			
			mean_collection[k, :] = (new_mean / r[k])
			
			new_covar = np.zeros([height, height])
			
			covariance[k] = Covariance(input, mean_collection[k,:])

		temp = np.zeros([width, gaussian_number])
		for k in range(gaussian_number):
			temp[:,k] = lmbda[k] * np.diagonal(oneGauss(input, mean_collection[k,:], covariance[k]))
		temp = np.sum(temp, axis = 1)
		temp = np.log(temp)
		N = np.sum(temp)
		iterations = iterations + 1
		#Run until the iterations are reached or a certain level of precision is surpassed
		if(iterations == 100) or (np.abs(N - number) < 0.01):
			break
		number = N
	return (lmbda, mean_collection, covariance)
			
def factAnal(input, factors):
	(row, col) = np.shape(input)
	
	mean = Average(input)
	#Initialize random phi values
	phi = np.random.randn(col, factors)
	
	temp_minus = (input - mean)
	variance = np.sum((temp_minus)**2, axis = 0) / row
	iter = 1
	iter_num = 0
	while True:
		#Expectation Step
		inverse_variance = (1/variance)
		new_phi = (phi.T)* inverse_variance
		temp = np.linalg.inv(np.dot(new_phi, phi) + np.identity(factors))
		#Expectation of the hidden variable
		hiddenE = np.dot(np.dot(temp, new_phi), temp_minus.T)
		
		tempE = [[]]*row
		for i in range(row):
			te = hiddenE[:,i]
			tempE[i] = temp + np.dot(te, te.T)
			
		#Maximization
		#Get new phi and update theta
		phi_2 = np.zeros([col, factors])
		for i in range(row):
			s1 = np.transpose(np.reshape(temp_minus[i,:], (1,-1)))
			s2 = np.transpose(np.reshape(hiddenE[:,i], (-1,1)))
			phi_2 = phi_2 + np.dot(s1, s2)
		
		phi_3 = np.zeros([factors, factors])
		for i in range(row):
			phi_3 = phi_3 + tempE[i]
		phi_3 = np.linalg.inv(phi_3)
		phi = np.dot(phi_2, phi_3)
		
		#Get new variance
		new_var = np.zeros([col,1])
		for i in range(row):
			m = np.transpose(temp_minus[i,:])
			v1 = m*m
			v2 = np.dot(phi, hiddenE[:,i]) * m
			new_var = new_var + v1 - v2
		variance = new_var / row
		variance = np.diag(variance)
		iter_num = iter_num + 1
		if iter_num == iter:
			break
	return (mean, variance, phi)
		
		
			


def mixnumber():
	#unused function for getting a number for the number of gaussians to run
	valid_input = 0 #Checking for valid user inputs  #Loop to confirm the user has made a correct choice
	try:
		val = input('Please input number of Gaussians:')
		user_input = int(val)
	except ValueError:
		print('Invalid Input. Please pick an integer')
	return user_input
def typeRequest():
	#Get the type of Model the user wants to run
	valid_input = 0 #Checking for valid user inputs
	while valid_input == 0:  #Loop to confirm the user has made a correct choice
		try:
			value = input("Please input type of model:\n 1 for single Gaussian\n 2 for Mixed Gaussian\n 3 for t-Distribution\n 4 for Factor Analyzer\n 5 for Mixture of t-distribution\n")
			user_input = int(value)
		except ValueError:
			print('Invalid Input. Please pick an integer')
		if (user_input == 1) or (user_input ==2) or (user_input==3) or (user_input==4) or (user_input == 5):
				valid_input = 1
				return user_input
		else:
			print("Invalid integer entered, please enter 1, 2, 3, or 4\n")
	return user_input



def mixOfT(input, precision, number_gaussians):
	#This function creates the model for the mixture of t-distributions. It only runs through 5 iterations due to the timing needed to run each pass.
	#Mixture of t-distribution 
	(row, col) = np.shape(input)
	#Get the inital values for the covariances of the multiple distributions
	var = [[]]*number_gaussians
	lmbda = np.matlib.repmat(1/number_gaussians, number_gaussians, 1)
	#Get the new weights 
	integers = np.random.permutation(row)
	integers = integers[0:row]
	means = input[integers, :]
	#Covariances
	cov_data = np.cov(input, rowvar = False, bias = 1, ddof = None)
	
	#Small amount of noise to avoid division by zeros
	
	for i in range(number_gaussians):
		var[i] = cov_data
		
	#Initialise DoF to some arbitrary large int
	nu = np.matlib.repmat(10, number_gaussians, 1)
	
	iter = 0
	stop_int = 100000 #Large int to stop from running forever
	while True:
		#Expectation Part
		tau = np.zeros([row, number_gaussians])
		for k in range(number_gaussians):
			#For each gaussian, compute the t-distribution probability
			ttau = mixT_pdf(input, means[k,:], var[k], nu[k])
			tau[:,k] = (lmbda[k]*ttau)
		#Normalize the data
		tau_total = np.sum(tau, axis = 1)
		tau = tau / np.reshape(tau_total, (row, -1))
		
		dlta = np.zeros([row, number_gaussians])
		for i in range(row):
			for k in range(number_gaussians):
				#Compute the distance from each point and the means and variances.
				dlta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(input[i,:],(1,-1)),means[k],np.linalg.inv(var[k]))**2)

		nu_dlta = np.zeros([row, number_gaussians])
		ExpectedHidden = np.zeros([row, number_gaussians])
		Expected_log = np.zeros([row, number_gaussians])
		nu_col = nu + col
		nu_dlta = nu.T + dlta
		for i in range(row):
			for k in range(number_gaussians):
				#Compute the new expected value, and the log liklihood of the expected value
				ExpectedHidden[i,k] = np.divide(nu_col[k],nu_dlta[i,k])
				Expected_log[i, k] = psi((nu_col[k])/2)- np.log(nu_dlta[i,k]/2)
				
		
		#Maximization Part
		#Update Lambda (Weights of each gaussian)
		for k in range(number_gaussians):
			lmbda = np.sum(tau, axis = 0) / row
		
		
		ExpectedHidden2 = ExpectedHidden * tau
		ExpHidSum = np.sum(ExpectedHidden2, axis = 0)
		#New Means 
		new_mean = np.zeros([number_gaussians, col])
		for k in range(number_gaussians):
			for i in range(row):
				new_mean[k, :] = new_mean[k, :] + (ExpectedHidden2[i, k]*input[i,:])
		
		#New Variances
		tau_sum = np.sum(tau, axis = 0)
		for k in range(number_gaussians):
		
			new = np.zeros([col, col])
			for i in range(row):
				ma = np.reshape((input[i,:] - means[k,:]),(1,-1))
				ma = ExpectedHidden2[i,k] * (ma.T * ma)
				new = new + ma
			var[k] = np.divide(new, tau_sum[k])
			var[k] = np.diag(np.diag(var[k]))
		
		#Update Nu, (minimize the cost function)
		for k in range(number_gaussians):
			nu[k] = t_cost(nu[k], ExpectedHidden[:,k], row)
		
		for i in range(row):
			for k in range(number_gaussians):
				#Once again check the new distances between each point and the means and variances.
				dlta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(input[i,:],(1,-1)),means[k,:],np.linalg.inv(var[k]))**2)
		iter = iter + 1
		print('Running Mixture of T. Iteration:' +str(iter))
		if iter == 5:
			break
	return (means, var, nu, lmbda)
	
def mixT_pdf(input, mean, covar, nu):
	#Uses more log functions to reduce the computations needed, otherwise the same as the other pdf for the t-distribution
	L = len(mean)
	
	d = np.exp(gammaln((nu+L)/2) - gammaln(nu/2))
	d = d / (((nu*np.pi)**(L/2)) * np.sqrt(np.linalg.det(covar)))
	N = input.shape[0]
	dlta = np.zeros([N,1])
	subtr = input - mean
	tmp = np.dot(subtr, np.linalg.inv(covar))
	for i in range(N):
		dlta[i] = np.dot(tmp[i,:], np.reshape(subtr[i,:], (-1,1)))
	ret = 1 + (dlta / nu)
	ret = ret**((-nu-L)/2)
	ret = np.dot(ret, d)
	return ret


np.random.seed(0)
face_train_images = []
nonface_train_images = []
face_test_images = []
nonface_test_images = []
for file in glob.glob("faceData/FaceTrain/*.png"):
	img = Image.open(file)
	data = np.array(img, dtype="float")
	face_train_images.append(data)
for file in glob.glob("faceData/FaceTest/*.png"):
	img = Image.open(file)
	data = np.array(img, dtype="float")
	face_test_images.append(data)
for file in glob.glob("nonFaceData/nonFaceTrain/*.png"):
	img = Image.open(file)
	data = np.array(img, dtype="float")
	nonface_train_images.append(data)
for file in glob.glob("nonFaceData/nonFaceTest/*.png"):
	img = Image.open(file)
	data = np.array(img, dtype="float")
	nonface_test_images.append(data)
size = 20
size_pca = 30
img_size = (20,20,3)
input = typeRequest()
pca_face = PCA(30)
temp = []
training_face = np.zeros([len(face_train_images), size*size*3])
training_nonface = np.zeros([len(face_train_images), size*size*3])
testing_face = np.zeros([len(face_test_images), size*size*3])
testing_nonface = np.zeros([len(face_test_images), size*size*3])

#for i in face_train_images:
for i in range(len(face_train_images)):	
	temp = face_train_images[i].flatten()
	training_face[i, :] = temp
training_face_pca = pca_face.fit_transform(training_face)
temp = []
for i in range(len(nonface_train_images)):
	temp = nonface_train_images[i].flatten()
	training_nonface[i, :] = temp
training_nonface_pca = pca_face.fit_transform(training_nonface)
temp = []
for i in range(len(face_test_images)):
	testing_face[i] = face_test_images[i].flatten() 
testing_face_pca = pca_face.fit_transform(testing_face)
temp = []
for i in range(len(nonface_test_images)):
	testing_nonface[i] = nonface_test_images[i].flatten()
testing_nonface_pca = pca_face.fit_transform(testing_nonface)

n1,m1 = testing_face.shape 
n2,m2 = testing_nonface.shape
truth = np.ones((n1,1))
false = np.zeros((n2,1))
new_truth = np.hstack((testing_face, truth))
new_false = np.hstack((testing_nonface,false))
X = np.concatenate((new_truth, new_false), axis=0)
np.random.shuffle(X[0:n1+n2])
labels = X[:,-1]
X_test_roc = X[:,:-1]
X_roc = pca_face.fit_transform(X_test_roc) 

if input == 1:
	mean, nonmean, covar, noncov = calculateOneGaussian(training_face_pca, training_nonface_pca)
	face_pdf = oneGauss(X_roc, mean, covar)
	
	nonface_pdf = oneGauss(X_roc, nonmean, noncov)
	total_roc = face_pdf + nonface_pdf
	plot_roc = face_pdf / total_roc
	plotavg(mean, img_size)
	plotROC(labels, plot_roc,1)
	
	prob_face = oneGauss(testing_face_pca, mean, covar)
	prob_nf = oneGauss(testing_face_pca, nonmean, noncov)
	
	total_face = prob_face + prob_nf
	PROB1 = prob_face / total_face
	count_face = np.sum(PROB1[:] >= 0.5)
	count_nonface = len(testing_face - count_face)
	
	prob_nonface = oneGauss(testing_nonface_pca, mean, covar)
	prob_fnf = oneGauss(testing_nonface_pca, nonmean, noncov)
	total_nonface = prob_nonface + prob_fnf
	PROB2 = prob_nonface / total_nonface
	count2 = np.sum(PROB2[:] >= 0.5)
	count_nf2 = len(testing_face) - count2
	
	FPR = count_nf2 / (count_nf2 + count2)
	FNR = count_nonface / (count_nonface + count_face)
	MC = (count_nf2 + count_nonface) / (len(testing_face))
	
	print('False Positive Rate:' + str(FPR))
	print('False Negative Rate:' + str(FNR))
	print('Misclassification Rate:' + str(MC))
		
elif input == 2:
	gaussian_number = 2
	sum1 = 0
	input = []
	for i in training_face_pca:
		input.append(i)
	for i in training_nonface_pca:
		input.append(i)
	input = np.array(input)
	face_mean = Average(training_face_pca)
	cov = Covariance(training_face_pca, face_mean)
	nonface_mean = Average(training_nonface_pca)
	noncov = Covariance(training_nonface_pca, nonface_mean)
	(lmbda, mean, covariance) = mixedGaussian(input, gaussian_number)
	
	
	for k in range(gaussian_number):
		total1 = (oneGauss(X_roc, mean[k,:], covariance[k]))
		sum1 = sum1+(lmbda[k] * total1)
	print(lmbda)
	
	sum2 = (oneGauss(X_roc, nonface_mean, noncov))
	whole = sum1 / (sum1 + sum2)
	plotROC(labels, whole, 1)
	(lmbda, mean, covariance) = mixedGaussian(training_face_pca, gaussian_number)
	prob_false_pos_face = np.zeros([len(testing_face), 1])
	for k in range(gaussian_number):
		P2 = np.diagonal(oneGauss(testing_face_pca, nonface_mean, noncov))
		P2 = np.reshape(P2, (-1,1))
		prob_false_pos_face = prob_false_pos_face + (lmbda[k] * P2)
	prob_fal_pos_nf = np.diagonal(oneGauss(testing_face_pca, nonface_mean, noncov))
	P_face = prob_false_pos_face / (prob_false_pos_face + np.reshape(prob_fal_pos_nf, (len(testing_face), 1)))
	face = np.sum(P_face[:] >= 0.5)
	falneg = len(testing_face) - face
	
	(lmbda, mean, covariance) = mixedGaussian(testing_nonface_pca, gaussian_number)
	prob_false_neg_face = np.zeros([len(testing_face), 1])
	for k in range(gaussian_number):
		P3 = np.diagonal(oneGauss(testing_nonface_pca, mean[k,:], covariance[k]))
		P3 = np.reshape(P3, (-1,1))
		prob_false_neg_face = prob_false_neg_face + (lmbda[k] * P3)
	
	prob_false_pos_nface = np.diagonal(oneGauss(testing_nonface_pca, face_mean, cov))
	P_nonface = prob_false_neg_face / (prob_false_neg_face + np.reshape(prob_false_pos_nface, (len(testing_face), 1)))
	
	nonF = np.sum(P_nonface[:] >= 0.5)
	falpos = len(testing_face) - nonF
	FPR = falpos / (falpos + nonF)
	FNR = falneg / (falneg + face)
	MCR = (falpos + falneg) / (2*len(testing_face))
	print('False Positive Rate:' + str(FPR))
	print('False Negative Rate:' + str(FNR))
	print('Misclassification Rate:' + str(MCR))
elif input == 3:
	(mean,covariance,nu) = fitting_t(training_face_pca, 0.01)
	(nonFaceMean, noncovariance, non_nu) = fitting_t(training_nonface_pca, 0.01)
	prob1 = t_pdf(X_roc, mean, covariance, nu)
	prob2 = t_pdf(X_roc, nonFaceMean, noncovariance, non_nu)
	prob = prob1 / (prob1 + prob2)
	plotROC2(labels, prob,1)
	
	prob_face_positive = t_pdf(testing_face_pca, mean, covariance, nu)
	prob_nonface_positive = t_pdf(testing_nonface_pca, mean, covariance, nu)
	prob_face_neg = t_pdf(testing_face_pca, nonFaceMean, noncovariance, non_nu)
	prob_nonface_neg = t_pdf(testing_nonface_pca, nonFaceMean, noncovariance, non_nu)
	
	p_face = prob_face_positive / (prob_face_positive + prob_face_neg)
	p_nonface = prob_nonface_positive / (prob_nonface_neg + prob_nonface_positive)
	plotavg(mean, img_size)
	
	total_face = np.sum(p_face[:] >= 0.5)
	total_non = len(testing_face) - total_face
	
	total_face2 = np.sum(p_nonface[:] >=0.5)
	total_non2 = len(testing_face) - total_face2
	
	FPR = total_non2 / (total_non2 + total_non)
	print('False Positive Rate:' + str(FPR))
	FNR = total_non / (total_non2 + total_non)
	print('False Negative Rate:' + str(FNR))
	MR = (total_non + total_non2) / (len(testing_face))
	print('Misclassification Rate:' +str(MR))
	
elif input == 4:
	factors = 4
	
	(mean, variance, phi) = factAnal(training_face_pca, factors)
	mu = pca_face.inverse_transform(mean)
	img = np.reshape(mu, img_size)
	img = img / np.max(mu)
	plt.imshow(img)
	plt.show()
	(nonmean, nonvar, nonphi) = factAnal(training_nonface_pca, factors)
	var = np.dot(phi, (phi.T)) + np.diag(variance)
	nonfvar = np.dot(nonphi, (nonphi.T)) + np.diag(nonvar)
	
	roc_face = oneGauss(X_roc, mean, var)
	roc_nonface = oneGauss(X_roc, nonmean, nonfvar)
	tot_roc = roc_face + roc_nonface
	Prob_roc = roc_face / tot_roc
	plotROC(labels, Prob_roc, 1)
	
	face_pf = oneGauss(testing_face_pca, mean.reshape((size_pca,)), var)
	face_nfp = oneGauss(testing_face_pca, nonmean.reshape((size_pca,)), nonfvar)
	
	tot1 = face_pf + face_nfp
	prob_face = face_pf / tot1
	
	nonface_fp = oneGauss(testing_nonface_pca, mean.reshape((size_pca,)), var)
	nonface_nfp = oneGauss(testing_nonface_pca, mean.reshape((size_pca,)), nonfvar)
	tot2 = nonface_fp + nonface_nfp
	prob_nonface = nonface_fp / tot2
	
	coll1 = np.abs(np.sum(prob_face[:] >= 0.5))
	coll2 = np.abs(len(testing_face) - coll1)
	
	coll3 = np.abs(np.sum(prob_nonface[:] >= 0.5))
	coll4 = np.abs(len(testing_nonface) - coll3)
	
	FPR = (coll4 / (coll4 + coll2))
	FNR = coll2 / (coll1 + coll3)
	MR = (coll4 + coll2) / (len(testing_face))
	print('False Positive Rate:' + str(FPR))
	print('False Negative Rate:' + str(FNR))
	print('Misclassification Rate:' +str(MR))
	
	#Test Data
	r = 0.0001
	(W,D) = np.shape(training_face_pca)
	lin_combo = np.zeros([8,D])
	new = np.zeros([1, len(mean)])
	new2 = np.zeros([1, len(mean)])
	
	for i in range(0,4):
		phi1 = np.transpose(phi[:,i])
		#Mu in +Direction
		v = mean
		new = v+r*phi1
		lin_combo[i,:] = new
		#Mu in -Direction
		new2 = v-r*phi1
		lin_combo[4+i,:] = new2
	lin_mat = [[]]*8
	
	for i in range(8):
		m =pca_face.inverse_transform(lin_combo[i,:])
		m = m/np.max(m)
		m = np.reshape(m, img_size)
		lin_mat[i] = m
	for i in range(8):
		plt.imshow(lin_mat[i])
		plt.show()
	
	
	
elif input == 5:
	number_ts = 3
	input = training_face_pca
	(L, W) = X_roc.shape
	print(W)
	(mean_face, var_face, nu_face, lmbda_face) = mixOfT(input, 0.01, number_ts)
	
	input = training_nonface_pca
	(mean_nonface, var_nonface, nu_nonface, lmbda_nonface) = mixOfT(input, 0.01, number_ts)
	for k in range(number_ts):
		n = pca_face.inverse_transform(mean_face[k])
		s = pca_face.inverse_transform(np.diag(var_face[k]))
		img = np.reshape(n, img_size)
		img = img / np.max(n)
		plt.imshow(img)
		plt.show()
	
	
	temp = np.zeros([L, number_ts])
	tmp2 = np.zeros([L, number_ts])
	
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(X_roc, mean_face[k,:], var_face[k], nu_face[k])
		n = lmbda_face[k]
		temp[:,k] = n * tmp2[:,k]
	
	P_face = np.sum(temp, axis = 1)
	
	temp = np.zeros([L, number_ts])
	tmp2 = np.zeros([L, number_ts])
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(X_roc, mean_nonface[k,:], var_nonface[k], nu_nonface[k])
		n = lmbda_nonface[k]
		temp[:,k] = n * tmp2[:,k]
	P_nface = np.sum(temp, axis = 1)
	
	Pr = P_face / (P_face + P_nface)
	plotROC2(labels, Pr, 1)
	
	
	temp = np.zeros([len(testing_face), number_ts])
	tmp2 = np.zeros([len(testing_face), number_ts])
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(testing_face_pca, mean_face[k,:], var_face[k], nu_face[k])
		n = lmbda_face[k]
		temp[:,k] = n * tmp2[:,k]
	face_fp = np.sum(temp, axis = 1)
	
	temp = np.zeros([len(testing_face), number_ts])
	tmp2 = np.zeros([len(testing_face), number_ts])
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(testing_face_pca, mean_nonface[k,:], var_nonface[k], nu_nonface[k])
		n = lmbda_nonface[k]
		temp[:,k] = n * tmp2[:,k]
	face_nfp = np.sum(temp, axis = 1)
	tot1 = face_fp + face_nfp
	Prob_f = face_fp / tot1
	
	temp = np.zeros([len(testing_face), number_ts])
	tmp2 = np.zeros([len(testing_face), number_ts])
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(testing_nonface_pca, mean_face[k,:], var_face[k], nu_face[k])
		n = lmbda_face[k]
		temp[:,k] = n * tmp2[:,k]
	nonface_fp = np.sum(temp, axis = 1)
	
	temp = np.zeros([len(testing_face), number_ts])
	tmp2 = np.zeros([len(testing_face), number_ts])
	for k in range(number_ts):
		tmp2[:, k] = mixT_pdf(testing_nonface_pca, mean_nonface[k,:], var_nonface[k], nu_nonface[k])
		n = lmbda_nonface[k]
		temp[:,k] = n * tmp2[:,k]
	nonface_nfp = np.sum(temp, axis = 1)
	
	tot2 = nonface_fp + nonface_nfp
	Prob_nf = nonface_fp / tot1
	
	total_face = np.sum(Prob_f[:] >= 0.5)
	face_t = len(testing_face) - total_face
	
	total_nonface = np.sum(Prob_nf[:] >= 0.5)
	nonface_t = len(testing_face) - total_nonface
	
	FPR = nonface_t / (nonface_t + total_nonface)
	FNR = face_t / (face_t + total_face)
	MC = (nonface_t + face_t) / len(testing_face)
	
	print('False Positive Rate = ' + str(FPR))
	print('False Negative Rate =' +str(FNR))
	print('Misclassification Rate:' + str(MC))
	
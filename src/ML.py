'''
File: Implementation of machine learning library
Author: Woodlin Smith
Class: CSC 448
Assignment: Term Project
Due Date: 12/8/2020
'''
import numpy as np
from math import sqrt
from statistics import mode,mean
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import random as rand

def plot_decision_regions(X, y, classifier, resolution=0.02):
    #setup marker generator and color map
    markers=('s','x','o','^','v')
    colors = ('red', 'blue','lightgreen','grey','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot decision surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
    plt.show()


def plot_scatter(X,Y):
    plt.scatter(X,Y)
    plt.show()

def regression_plot(X, Y, reg_model):
    plt.plot(X,reg_model.X.dot(reg_model.weights))
    plt.scatter(X,Y)
    plt.show()

class Perceptron:
    def __init__(self, rate=0.01, niter=10):
        self.rate=rate
        self.niter=niter
        self.weights=[]
        self.errors=[]
    
    def fit(self, x,y):
        
        #weights: create a weights array of correct size
        # initialize to 0
        self.weights=np.zeros(len(x[0])+1)
        

        #number of misclassifications, creates an array
        #to hold number of misclassifications
        for i in range(self.niter):
            self.errors.append(0)
        
        iter_count=0
        converged=False
        #main loop to fit data
        while iter_count<self.niter and not converged:
            #set iteration error=0
            iteration_error=0
            #loop over all objects in X and corresponding y elements
            for xi, target in zip(x,y):
                #add bias term to element
                xi=np.insert(xi,0,1)
                #calculate delta_w, update from previous step
                delta_w=self.rate*(target-self.predict(xi))

                #increase the iteration error if delta_w!=0
                if delta_w != 0:
                    self.weights+=(delta_w*xi)
                    iteration_error+=1
            #update misclassifcation array with # of errors in iteration
            self.errors[iter_count]=iteration_error
            if iteration_error==0:
                converged=True
            iter_count+=1
        return self

    def net_input(self, x):
        #return dot product of x.w + bias
        try:
            return np.dot(self.weights,x)
        #If an array was passed in externally
        except ValueError:
            try:
                ni_array=np.empty(0)
                for elem in x:
                    elem=np.insert(elem,0,1)
                    ni_array=np.append(ni_array,np.dot(self.weights,elem))
                return ni_array
            #if a single data point was passed in externally
            except ValueError:
                x=np.insert(x,0,1)
                return np.dot(self.weights,x)

    def predict(self, x):
        #return the class label after unit step
        return np.where(self.net_input(x)>=0.0,1,-1) 


class LinearRegression:
    def __init__(self):
        self.weights=np.zeros(2)
        self.X=[]
        self.Y=[]
        self.regression_calc=False
    
    def fit(self, X, Y):

        #add column of ones to get bias term
        ones=np.ones(len(X))
        np_x=np.asarray(X)
        np_x=np.vstack((ones,np_x)).T

        np_y=np.asarray(Y)

        self.X=np_x
        self.Y=np_y

        trans_x=np_x.T

        #caclulate A and b
        matr_a=np.linalg.inv(trans_x.dot(np_x))
        vec_b=trans_x.dot(np_y)

        self.weights=matr_a.dot(vec_b)
        
        self.regression_calc=True
        
    def predict(self, x_val):
        try:
            if not self.regression_calc:
                raise ValueError
            return self.weights[1]*x_val+self.weights[0]
        except ValueError:
            print("No model created. Please fit a model first.")
            return -1

    def calc_rsquared(self,x_vec):
        try:
            if not self.regression_calc:
                raise ValueError

            y_mean=np.mean(self.Y)

            sst=0
            ssreg=0
            for i in range(len(x_vec)):
                sst+=(self.Y[i]-y_mean)**2
                ssreg+=(self.predict(x_vec[i])-y_mean)**2
        
            return ssreg/sst
        except ValueError:
            print("No model created. Please fit a model first.")
            return -1

class Interval:
    def __init__(self):
        self.min_thresh=0xffffff
        self.max_thresh=-1

    def find_cutoffs(self,X,Y):
        elems_in_class=[]

        for xi, label in zip(X,Y):
            if label == 1:
                elems_in_class.append(xi)  
        elems_in_class.sort()

        self.min_thresh=elems_in_class[0]
        self.max_thresh=elems_in_class[len(elems_in_class)-1]
    
    def predict_label(self, x):
        try:
            if self.min_thresh<=x and self.max_thresh >= x:
                return 1
            else:
                return -1
        #if an array was passed in
        except ValueError:
            label_array=[]
            for elem in x:
                if self.min_thresh<=elem and self.max_thresh >= elem:
                    label_array.append(1)
                else:
                    label_array.append(-1)
            return label_array

class Threshold:
    def __init__(self):
        self.max_thresh=-0xfffff

    def find_cutoff(self,X,Y):
        elems_in_class=[]
        
        for xi, label in zip(X,Y):
            if label==1:
                elems_in_class.append(xi)
        elems_in_class.sort()

        self.max_thresh=elems_in_class[len(elems_in_class)-1]

    def predict_label(self, x):
        try:
            if self.max_thresh >= x:
                return 1
            else:
                return -1
        #if an array was passed in
        except ValueError:
            label_array=[]
            for elem in x:
                if self.max_thresh >= elem:
                    label_array.append(1)
                else:
                    label_array.append(-1)
            return label_array

class KNN:
    def __init__(self,X,Y):
        self.data=X
        self.labels=Y
        self.dist_dict={"e":self.euclidean_dist,"m":self.manhattan_dist,"s":self.supremum_dist}
        self.choice_dict={"c":self.knn_mode,"r":np.mean}
        self.knn_vec=[]
        self.k_labels=[]

    def knn(self, k, query,distance_method="e",choice="c"):
        self.q_e=query
        self.knn_vec=[]
        self.k_labels=[]
        distance_func=self.dist_dict[distance_method]

        for index, e in enumerate(self.data):
            dist=distance_func(e,self.q_e)

            self.knn_vec.append((dist,index))
        self.knn_vec=sorted(self.knn_vec)

        self.knn_vec=self.knn_vec[:k]

        labels=[]
        for dist, index in self.knn_vec:
            labels=self.labels[index]
        self.k_labels=labels
        return self.choice_dict[choice](self.k_labels)
    
    def predict(self, k, queries, distance_method="e", choice="c"):
        predictions=[]
        if not isinstance(queries[0],list):
            for query in queries:
                predictions.append(self.knn(k,query,distance_method,choice))
        else:
            print("Not a list of vecs")
            predictions.append(self.knn(k,queries,distance_method, choice))
        return predictions
        
    
    def euclidean_dist(self,elem,query):
        dist=0
        for i in range(len(query)):
            dist+=(elem[i]-query[i])**2
        return sqrt(dist)
    
    def manhattan_dist(self,elem,query):
        dist=0
        for i in range(len(query)):
            dist+=abs(elem[i]-query[i])
        return dist

    def supremum_dist(self,elem,query):
        dist_elems=[]
        for i in range(len(query)):
            dist_elems.append(abs(elem[i]-query[i]))
        return max(dist_elems)

    def knn_mode(self,labels):
        (values,counts)=np.unique(labels,return_counts=True)
        return values[np.argmax(counts)]



        
class SupportVectorMachine:
    
    def __init__(self, num_iters=1000, learning_rate=0.001):
        self.iter=num_iters
        self.rate=learning_rate
        self.X=[]
        self.Y=[]
        self.weights=[]
    
    def fit(self, X, Y):
        #add bias term
        ones=np.ones((len(X),1))
        np_x=np.asarray(X)
        np_x=np.hstack((ones,np_x))

        self.X=np_x
        self.Y=Y

        self.weights=np.zeros(len(self.X[0]))

        
        zipped_elems=zip(self.X,self.Y)
        zipped_elems=list(zipped_elems)

        #perform sgd
        for i in range(self.iter):
            rand.shuffle(zipped_elems)
            for example, label in zipped_elems:
                if label*np.dot(example,self.weights)>=1:
                    #set lambda =1/(i+1), which will make the importance of the hinge loss increase as iterations increase
                    self.weights-=self.rate*(2*1/(i+1)*self.weights)
                else:
                    self.weights-=self.rate*(2*1/(i+1)*self.weights-np.dot(example,label))
        
                    
    def predict(self, val):
        try:
            return np.sign(np.dot(np.asarray(val),self.weights))
        except ValueError:
            ones=np.ones((len(val),1))
            val=np.hstack((ones,val))
            return np.sign(np.dot(np.asarray(val),self.weights))


        
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iters=1000):
        self.weights=[]
        self.rate=learning_rate
        self.niters=num_iters
        self.X=[]
        self.Y=[]

    def fit(self, x, y):
        ones=np.ones((len(x),1))
        np_x=np.asarray(x)
        np_x=np.hstack((ones,np_x))

        self.X=np_x
        self.Y=y

        self.weights=np.zeros(len(self.X[0]))

        #perform gradient descent
        for i in range(self.niters):
            n=np.dot(self.X,self.weights)
            h=self.sigmoid(n)
            delf=np.dot(self.X.T,(h-self.Y))/len(self.Y)
            self.weights-=self.rate*delf

       
    def sigmoid(self, n):
        return 1/(1+np.exp(-n))
    
    def predict(self, X):
        ones=np.ones((len(X),1))
        np_x=np.asarray(X)
        X=np.hstack((ones,np_x))
        #using 0.5 as probability threshold
        try:
            n=np.dot(X,self.weights)
            if self.sigmoid(n) > 0.5:
                return 1
            else:
                return 0
        except ValueError:
            predictions=[]
            for elem in X:
                n=np.dot(elem,self.weights)
                
                if self.sigmoid(n)>0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            return np.asarray(predictions)
            





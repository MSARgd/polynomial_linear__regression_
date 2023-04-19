#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:41:01 2023

@author: msa
"""
# target is petal_width   f.iloc[:1,1:2]
# feautres :  petale length
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
df = pd.read_csv("iris.csv")
print(df.head())
print(df.columns)
X = np.hstack((df.iloc[:,2:3]**2,df.iloc[:,2:3]))
X = np.hstack((X,np.ones((X.shape[0],1))))
y = np.array(df.iloc[:,3:4])
# plt.scatter(df.iloc[:,2:3],y)
X_train,y_train = X[:101],y[:101]
X_test , y_test = X[100:150] ,  y[100:150]
'''theta'''
theta = np.random.randn(X.shape[1],1)
print(X.shape)
print(theta.shape)
print(y.shape)
# plt.scatter(df.iloc[:,2:3], y)
# plt.plot(df.iloc[:,2:3], y)
# sys.exit()
# ===========================================
class MultiPloyLinReg :
    def __init__(self,lr=0.001,epoches=5000,batch_size=32) :
        self.lr = lr
        self.epoches = epoches
        self.bias = None
        self.weights = None
    def fit(self,X,O) :
        return X@O
    def predict(self,X,O) :
        return  X.dot(O)
    def grad(self,X, y, theta):
        n_samples,n_feautres = X.shape
        return 1/n_samples * X.T.dot(self.fit(X, theta) - y)
    def cost_function_mse(self,X, y, theta):
        n_samples,n_feautres = X.shape
        return 1/(2*n_samples) * np.sum((self.fit(X, theta) - y)**2)
    # gradient descent
    def gradient_descent(self,X, y, theta):
        cost_history = np.zeros(self.epoches) 
        for i in range(0, self.epoches) : 
            theta = theta - self.lr * self.grad(X, y, theta) 
            cost_history[i] = self.cost_function_mse(X, y, theta) 
        return theta, cost_history 
    
    def show_mse_evaultion(self,cost_history) : 
        plt.xlabel("epoches") 
        plt.ylabel("error") 
        plt.plot(cost_history,range(self.epoches)) 
        plt.show()
    
        
    def plot_model(self,x,y_pred,y,O):
        self.plot_dataset(x, y_pred)
        # plt.plot(x[:,1],y_pred,label="curve model")
        plt.scatter(x[:,1],y, marker='s',color="black",label="Real data")
        plt.title("Model")
        plt.xlabel("petal_width")
        plt.ylabel("petale length")
        plt.legend()
        plt.show()
        
      
        
    def plot_dataset(self,x,y) :
        plt.scatter(x[:,1],y,color="r",)
        # plt.show()
    
    
    
if __name__ == '__main__' : 
    best_theta = np.array([[ 0.02911896],[ 0.21971956],[ 0.5866205 ]])
    mplr = MultiPloyLinReg()
    mplr.plot_dataset(X_train,y_train)
    theta_final, cost_history = mplr.gradient_descent(X_train, y_train, theta)
    print(theta_final)
    print(cost_history[len(cost_history)-1])
    # ===========================================
    predictions = mplr.predict(X_test, theta_final)
    compered_y = np.hstack((predictions,y_test))
    # =========================================== print(compered_y)
    mplr.show_mse_evaultion(cost_history)
    mplr.plot_model(X_train, mplr.predict(X_train, theta_final),y_train,theta_final)
    
 # [[ 0.02078279]
 #  [ 0.26386462]
 #  [-0.1778925 ]]
 # 0.007423533568286428

   
    
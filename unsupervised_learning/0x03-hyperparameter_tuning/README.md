# 0x03-hyperparameter_tuning
This directory contains work with tuning hyperparameters using Gaussian process and Bayesian optimization:

## Mandatory Tasks:
0. [Initialize Gaussian Process](/unsupervised_learning/0x03-hyperparameter_tuning/0-gp.py)
* Create a Python class that represents a noiseless 1D Gaussian process with class constructor and public instance method to calculate the covariance kernel matrix between two matrices.
1. [Gaussian Process Prediction](/unsupervised_learning/0x03-hyperparameter_tuning/1-gp.py)
* Based on previous, update the class to include public instance method that predicts the mean and standard deviation of points in a Gaussian process.
2. [Update Gaussian Process](/unsupervised_learning/0x03-hyperparameter_tuning/2-gp.py)
* Based on previous, update the class to include public instance method that updates a Gaussian Process.
3. [Initialize Bayesian Optimization](/unsupervised_learning/0x03-hyperparameter_tuning/3-bayes_opt.py)
* Create a Python class that performs Bayesian optimization on a noiseless 1D Gaussian process.
4. [Bayesian Optimization - Acquisition](/unsupervised_learning/0x03-hyperparameter_tuning/4-bayes_opt.py)
* Based on previous, update the class to include public instance method that calculates the next best sample location.
5. [Bayesian Optimization](/unsupervised_learning/0x03-hyperparameter_tuning/5-bayes_opt.py)
* Based on previous, update the class to include public instance method that optimizes the black-box function.
6. [Bayesian Optimization with GPyOpt](6-bayes_opt.py)
* Write a Python script that optimizes a machine learning model of choice using GPyOpt.  Then, write a blog post describing the background and approach to this task.

### test_files directory
The test_files directory contains all files used to test output locally.

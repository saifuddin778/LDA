from __future__ import division
import sys
import math
import shelve
import random
import copy
from time import time
sys.dont_write_bytecode = True

#local imports
from tools import fmin, Functions

funcs_ = Functions()

__author__ = "Saifuddin Abdullah"   

#--implementation of linear discriminant analysis for binary classification tasks.
#--Any improvement or such suggestions will be honored. :-)
#--provide features (x's) which can have any dimensions, and a set of 1d y's or observations.
class LDA(object):
    """"Linear discriminant analysis for multiclass/binary classification (x(i), y(i))"""
    def __init__(self, x, y):
        self.funcs_ = funcs_
        #verify the dimensions
        if self.funcs_.verify_dimensions(x):
            if len(x) == len(y):
                self.len_all = len(x)
                self.dmean_ = self.funcs_.mean_nm(x, axis=0)
                self.std_ = self.funcs_.std_nm(x, axis=0)
                #self.x = self.funcs_.normalize_(x)
                self.x = x
                self.y = y
                self.separate_sets()
            else:
                sys.exit()
        else:
            print 'data dimensions are inaccurate..exiting..'
            sys.exit()
    
    #--separating the datasets based on their associated classes or labels
    def separate_sets(self):
        self.groups = {}
        self.group_names = list(set(self.y))
        if len(self.group_names) > 2:
            print 'more than two classes provided...exiting'
            sys.exit()
        #putting all the samples in a regular order so that their
        #grouping can be easier.
        combined  = sorted(zip(self.x, self.y), key = lambda n: n[1])
        #--doing val,key here because (x,y) was zipped
        for val,key in combined:
            if self.groups.has_key(key):
                self.groups[key].append(val)
            else:
                self.groups[key] = []
                self.groups[key].append(val)
        #train on each group
        self.train()

    #--substracts global mean from the group point
    def substract_mean(self, group_point):
        for i, a in enumerate(group_point):
            group_point[i] = group_point[i] - self.mean_global[i]
        
        return group_point

    #--gets the covariance matrix for a dataset/group (t(x).x/len(x))
    def get_cov_matrix(self, matrix):
        cov_mat = []
        for i in zip(*matrix):
            l = []
            for e in zip(*matrix):
                l.append(self.funcs_.dot(i, e)/len(matrix))
            cov_mat.append(l)
        return cov_mat

    #--multiplies the length of dataset with each element of vector and
    #--and divides by the total lenght of the dataset.
    def scalar_mult(self, scalar, vector, divisor):
        new_vector = []
        for  i, a in enumerate(vector):
            new_vector.append((scalar*vector[i])/divisor)
        return new_vector

    #--adds two vectors linearly
    def sum_vectors(self, a, b):
        return [g+h for g,h in zip(a,b)]

    #--calculates the global covariance matrix
    def get_global_cov(self, cov_matrices):
        temp_cov = []
        global_cov = []
        for i, a in enumerate(cov_matrices):
            l = []
            for e, j in enumerate(cov_matrices[a]):
                l.append(self.scalar_mult(self.lens[a], cov_matrices[a][e], self.len_all))
            temp_cov.append(l)
        for a,b in zip(*temp_cov):
            global_cov.append(self.sum_vectors(a,b))
        
        return global_cov
    
    #--processes the groups
    #--finds mean for each of the data group and mean corrects it by substracting
    #--global mean out of it and then finds covariance matrices for them.
    def train(self):
        self.mean_sets = {}
        covariance_sets = {}
        self.lens ={}
        self.probability_vector = {}
        
        self.mean_global = self.funcs_.mean_nm(self.x, axis=0)
        for k, v in self.groups.iteritems():
            self.lens[k] = len(self.groups[k])
            self.probability_vector[k] = self.lens[k]/self.len_all
            self.mean_sets[k] = self.funcs_.mean_nm(self.groups[k], axis=0)
            #mean correcting each set i.e. set - global mean
            self.groups[k] = map(self.substract_mean, self.groups[k])
            covariance_sets[k] = self.get_cov_matrix(self.groups[k])
        self.covariance_global = self.get_global_cov(covariance_sets)
        #--inverse the global covariance matrix
        self.covariance_global = self.funcs_.inv_(self.covariance_global)
        #training completedelf.covariance_global
    
    #prediction or discrimnant function
    def predict(self, v, key_only=False):
        predictions  = {}
        for a in self.group_names:
            predictions[a] = self.funcs_.dot(self.funcs_.prod_2(self.mean_sets[a], self.covariance_global), v) - (self.funcs_.dot(self.funcs_.prod_2(self.mean_sets[a], self.covariance_global), self.mean_sets[a])*0.5)+math.log(self.probability_vector[a])
        if key_only:
            return max(predictions, key=predictions.get)
        else:
            return predictions


"""
A good way to approach multiclass LDA is to get the discriminant output pairwise i.e.
form N(N-1)/2 pairs for the sample dataset, where N is the number of of classes involved
and find classifications for each pair and then combine them to have the final output.
"""
class multiclass_LDA(object):
    def __init__(self, x, y):
        self.funcs_ = funcs_
        if self.funcs_.verify_dimensions(x):
            if len(x) == len(y):
                self.x  = x
                self.y = y
                self.process_sets()
            else:
                sys.exit()
        else:
            print 'data dimensions inaccurate..exiting.'
            sys.exit()
    
    #returns N(N-1)/2 pairs/combinations of the classes involved.
    def get_combinations(self, unique_labels):
        g = []
        for i, a in enumerate(unique_labels):
            current_ = unique_labels[i]
            for e, b in enumerate(unique_labels):
                if e > i:
                    g.append([unique_labels[i], unique_labels[e]])
        return g

    #for separating data based on classes (not used any more)
    def separate_data(self, combined):
        #sort all the classes so that no fuss is there
        combined = sorted(combined, key=lambda n: n[1])
        for i, a in enumerate(combined):
            self.separated_features[combined[i][1]].append(combined[i][0])
            self.separated_labels[combined[i][1]].append(combined[i][1])

    #returns data based on the given binary class combination (a,b)
    def get_x(self, current_):
        result = []
        for a,b in zip(self.x, self.y):
            if b in current_:
                result.append(copy.deepcopy(a))
        return result

    #returns labels based on the given binary class combination (a, b)
    def get_y(self, current_):
        result = []
        for i, a in enumerate(self.y):
            if self.y[i] in current_:
                result.append(copy.deepcopy(self.y[i]))
        return result

    #processes each set
    def process_sets(self):
        self.unique_labels = list(set(self.y))
        #self.separated_features = dict([(k, []) for k in self.unique_labels])
        #self.separated_labels = dict([(k, []) for k in self.unique_labels])
        #self.separate_data(zip(self.x, self.y))
        self.combinations = self.get_combinations(self.unique_labels)
        self.classifiers = {}

        #training a classifier for each pair
        for a in self.combinations:
            import time as qt
            x_ = []
            y_ = []
            current_ = a
            x_ = self.get_x(current_)
            y_ = self.get_y(current_)
            self.classifiers[tuple(a)] = LDA(x_,y_)

    #predictions from each classifier
    def predict(self, v):
        t = []
        for key, value in self.classifiers.iteritems():
            t.append(self.classifiers[key].predict(v))
        return t


        
        
    



# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:13:50 2018

@author: bauer
"""

import pandas as pd
import os
import numpy as np
from scipy.stats import zscore
#Read and set up the data

X = 'Crime_Data_from_2010_to_Present (1).csv'

pwd = os.getcwd()
os.chdir('C:/Users/bauer/Documents/EAE/Práctica python ana')
df = pd.read_csv(os.path.basename(X), engine = 'python')
df_or = pd.read_csv(os.path.basename(X), engine = 'python')
os.chdir(pwd)


#Taking a random sample of the dataframe
df_sample = df.sample(n = 20000, random_state = 4561, axis = 0)
df.isnull().sum()

#Taking another sample to use in an independent validation set

test_sample = df.sample(n = 20000, random_state = 0, axis = 0)
test_sample.isnull().sum()

#Data cleansing and treatment

df_obj = df_sample.select_dtypes(include = 'object').apply(lambda x: x.astype('category').cat.codes)
test_obj = test_sample.select_dtypes(include = 'object').apply(lambda x: x.astype('category').cat.codes)
df_sample.fillna(method = 'ffill', inplace = True)
df_sample.fillna(method = 'bfill', inplace = True)
test_sample.fillna(method = 'ffill', inplace = True)
test_sample.fillna(method = 'bfill', inplace = True)

test = test_sample.select_dtypes(exclude = 'object')

df = df_sample.select_dtypes(exclude = 'object')
df_obj.columns
df[['Date Reported', 'Date Occurred', 'Area Name', 'Crime Code Description','MO Codes', 'Victim Sex', 
    'Victim Descent', 'Premise Description','Weapon Description', 'Status Code', 'Status Description', 
    'Address','Cross Street', 'Location']] = df_obj[['Date Reported', 'Date Occurred', 'Area Name', 'Crime Code Description',
                                         'MO Codes', 'Victim Sex', 'Victim Descent', 'Premise Description',
                                         'Weapon Description', 'Status Code', 'Status Description', 'Address',
                                         'Cross Street', 'Location ']]

#Observe features correlation

df.corr()

#Clean the less correlated features

df.drop(columns = ['Location','Address','Date Occurred','Date Reported','Reporting District',
                   'Area ID','Time Occurred','DR Number', 'Crime Code',
                   'Crime Code 2', 'Crime Code 3','Crime Code 4'], inplace = True)

test[['Date Reported', 'Date Occurred', 'Area Name', 'Crime Code Description',
       'MO Codes', 'Victim Sex', 'Victim Descent', 'Premise Description',
       'Weapon Description', 'Status Code', 'Status Description', 'Address',
       'Cross Street', 'Location']] = test_obj[['Date Reported', 'Date Occurred', 'Area Name', 'Crime Code Description',
       'MO Codes', 'Victim Sex', 'Victim Descent', 'Premise Description',
       'Weapon Description', 'Status Code', 'Status Description', 'Address',
       'Cross Street', 'Location ']]
test.drop(columns = ['Location','Address','Date Occurred','Date Reported','Reporting District',
                   'Area ID','Time Occurred','DR Number', 'Crime Code',
                   'Crime Code 2', 'Crime Code 3','Crime Code 4'], inplace = True)


#Eliminate outliers

'''dfz = df.apply(zscore)
dfz_columns = dfz.columns
for index, row in dfz.iterrows():
    for column_name  in dfz_columns:
        if row[column_name] < -2 or row[column_name] > 2:
            dfz.replace(row[column_name], np.NaN, inplace = True)
        

dfz.isnull().sum()
dfz.mean()
dfz

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis= 1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh   

mad_based_outlier(df)'''


#Defining the dependent variable
y = pd.DataFrame(df_sample['Crime Code'], columns = ['Crime Code'])
test_y = pd.DataFrame(test_sample['Crime Code'], columns = ['Crime Code'])

y.plot(kind = 'hist')

y_cat = y['Crime Code'].astype('category').cat.codes
y_cat = pd.DataFrame(y_cat, columns = ['Crime Codes'])
test_y_cat = test_y['Crime Code'].astype('category').cat.codes

y_cat.plot(kind = 'hist')

'''y_cat = y_cat.sample(n = 11052, random_state = 5469, axis = 0)
test_y_cat = test_y_cat.sample(n = 11052, random_state = 5469, axis = 0)
test = test.sample(n = 11052, random_state = 1, axis = 0)
testz = test.apply(zscore)'''
#There is an unbalanced data problem. To solve it we gonna oversample the minority class

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
x_resampled, y_resampled = ros.fit_sample(df, y_cat)
print(sorted(Counter(y_resampled).items()))
y_resampled = pd.DataFrame(y_resampled, columns = ['Crime Code'])
x_resampled = pd.DataFrame(x_resampled, columns = ['Victim Age', 'Premise Code', 'Weapon Used Code',
                                                   'Crime Code 1',
       'Area Name', 'Crime Code Description', 'MO Codes', 'Victim Sex',
       'Victim Descent', 'Premise Description', 'Weapon Description',
       'Status Code', 'Status Description', 'Cross Street'])

#Let's take a new random sample of the new data, containing the number of rows in the original dataframe
#To avoid oversampling
    
y_sample = y_resampled.sample(n = 20000, random_state = 4561, axis = 0)
x_sample = x_resampled.sample(n = 20000, random_state = 4561, axis = 0)

#Reaobserve the new distribution after oversampling
    
y_resampled.plot(kind = 'hist')

#Split the data in train andtest sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.3, random_state = 9856)

#POWERTUNING
#Cross validation and parameter optimization with several algorithms

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn import cross_validation as cval
from sklearn.model_selection import permutation_test_score
from sklearn.ensemble import RandomForestClassifier


Cval = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 4587)
kf_total = cval.KFold(len(x_train), n_folds=10, shuffle=True, random_state=4)

#Cross validation score after the model is optimized

score, permutation_scores, pvalue = permutation_test_score(
    tuned_clf, x_train, y_train, scoring="f1_micro", cv=Cval, n_permutations=20, n_jobs= 3)

print("Classification score %s (pvalue : %s)" % (score, pvalue))


#Parameters to optimizze

param_grid =  {  
                
                
                
                "criterion": ["gini", "entropy"],
                'max_depth': range(6, 100),
                'max_features': [None, 'auto', 'log2'],
                }

#Utility function to get the score on the best parameters 

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#Model to optimize

n_iter_search = 30

clf = RandomForestClassifier(n_estimators = 200,
                             bootstrap = True, oob_score = True,
                             n_jobs = -1, random_state = 2154, warm_start = False, class_weight = 'balanced')

rand_search= RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = -1, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

#Results and elapsed time

start = time()
rand_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(rand_search.cv_results_)


#Final optimized model

'''tuned_clf = OneVsRestClassifier(BaggingClassifier(RandomForestClassifier(
                                    
                                   criterion = 'gini', n_estimators = 200,
                             bootstrap = True, oob_score = True,
                             n_jobs = 1, random_state = 2154, warm_start = False), n_jobs = 1, random_state = 24,
                                n_estimators = 40, oob_score = True, warm_start = False), n_jobs = -1)'''

tuned_clf = OneVsRestClassifier(RandomForestClassifier(
                                    max_features = 'auto', max_depth = 73,
                                   criterion = 'gini',n_estimators = 300,
                             bootstrap = True, oob_score = True,
                             n_jobs = 1, random_state = 2154, warm_start = False), n_jobs = -1)


tuned_clf.fit(x_train, y_train)
tuned_clf.score(x_test, y_test)

#Repeating the process with different algorithms to compare
from sklearn.ensemble import ExtraTreesClassifier

#Parameters to optimize

param_grid =  {  
                
                
                
                "criterion": ["gini", "entropy"],
                'max_depth': range(6, 100),
                'max_features': [None, 'auto', 'log2'],
                }

#Utility function to get the score on the best parameters 

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#Model to optimize

n_iter_search = 30

tree = ExtraTreesClassifier(n_estimators = 300,
                            bootstrap = True, oob_score = True, n_jobs = -1,
                            random_state = 1245, class_weight = 'balanced')

tree_search= RandomizedSearchCV(tree, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = -1, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

#Get results and elapsed time

start = time()
tree_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(tree_search.cv_results_)


#Optimized final model

tuned_tree = tree = OneVsRestClassifier(ExtraTreesClassifier(max_features = 'auto', max_depth = 73,
                                   criterion = 'gini',n_estimators = 700,
                             bootstrap = True, oob_score = True,
                             n_jobs = 1, random_state = 2154, warm_start = False), n_jobs = -1)



tuned_tree.fit(x_train, y_train)
tuned_tree.score(x_test, y_test)


#Repeating the same process with different algorithms to compare 
from sklearn.ensemble import GradientBoostingClassifier

#Parameters to optimize

param_grid =  {  
                'base_estimator':[GradientBoostingClassifier()],
                'subsample': range(0.1, 1),
                'loss': ['deviance', 'exponential'],
                "criterion": ["gini", "entropy"],
                'max_depth': range(6, 100),
                'max_features': [None, 'auto', 'log2'],
                }

#Utility function to get the score on the best parameters 

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#Model to optimize

n_iter_search = 30

gr = OneVsRestClassifier(GradientBoostingClassifier(loss = 'deviance', n_estimators = 100, random_state = 45,
                                max_features = None, warm_start = True, presort = 'auto', init = None),
                        n_jobs = -1)

gr_search= RandomizedSearchCV(gr, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = -1, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

#Get results and elapsed time

start = time()
tree_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(gr_search.cv_results_)


#Final optimized model

tuned_gr =  GradientBoostingClassifier(loss = 'deviance', n_estimators = 100, random_state = 45,
                                max_features = None, warm_start = True, presort = 'auto', init = None)



gr.fit(x_train, y_train)
gr.score(x_test, y_test)

#Final Validation
from sklearn.metrics import f1_score

#1. ONE VS REST CLASSIFIER CON RANDOM FOREST

tuned_clf.fit(x_sample, y_sample)
prediction = tuned_clf.predict(test)
f1_score(test_y_cat, prediction, average = 'micro')

#2.GRADIENT BOOSTING CLASSIFIER ANIDADO IN ONE VS REST CLASSIFIER

gr.fit(x_sample, y_sample)
prediction = gr.predict(test)
f1_score(test_y_cat, prediction, average = 'micro')


#3.EXTRA TREES CLASSIFIER

tuned_tree.fit(x_sample, y_sample)
prediction = tuned_tree.predict(test)
f1_score(test_y_cat, prediction, average = 'micro')

###################################################################################################################################################

#Plotting a crime heatmap

import gmplot

#let's load the map

gmap = gmplot.GoogleMapPlotter.from_geocode('Los Angeles', 10)

#Isolating the coordinates

coordinates = df_or['Location '].astype(str)
coordinates
coordinates = pd.DataFrame(coordinates)

coordinates.reset_index(drop = True, inplace = True)

#Separate longitude and latitude and storing each one into a new DF

coordinates['Location '][0][1:8]
coordinates
latitude = []
longitude = []



for x in coordinates['Location '].index:
    latitude.append(coordinates['Location '][x][1:8])

for x in coordinates['Location '].index:
    longitude.append(coordinates['Location '][x][11:19])

longitude
    
longitude = pd.DataFrame(longitude, columns = ['longitude'])
longitude = pd.to_numeric(longitude['longitude'], errors = 'coerce')
longitude = pd.DataFrame(longitude, columns = ['longitude'])

latitude = pd.DataFrame(latitude, columns = ['latitude'])
latitude = pd.to_numeric(latitude['latitude'], errors = 'coerce')
latitude = pd.DataFrame(latitude, columns = ['latitude'])
latitude
#latitude = pd.DataFrame(latitude, columns = ['latitude'])

#Clean up NaN's

latitude.isnull().sum()
latitude.dropna(inplace = True)
latitude

longitude.isnull().sum()
longitude.dropna(inplace = True)
longitude

#Convert all longitude values to negative

for x in longitude['longitude'].index:
    if longitude['longitude'][x] > 0:
        longitude['longitude'][x] = -longitude['longitude'][x]


longitude

#Get latitude and longitude together into the same DF

lat_lon = pd.DataFrame()
lat_lon['latitude'] = latitude['latitude'].astype('float')
lat_lon['longitude'] = longitude['longitude'].astype('float')


#Clean up NaN's again

lat_lon.isnull().sum()
lat_lon.dropna(inplace = True)
lat_lon

#Load Coordinates into the heatmap

gmap.heatmap(lat_lon['latitude'], lat_lon['longitude'])

#Save the heatmap plot into an html file from where we will be able to visualize it

gmap.draw('C:/Users/bauer/Documents/EAE/Práctica python ana/crime_map.html')















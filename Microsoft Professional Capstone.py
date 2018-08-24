# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:52:23 2018

@author: bauer
"""

# load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_csv('train_values.csv')
#df = pd.read_csv('C:/Users/bauer/Documents/EAE/Data Science course/Capstone project/dffinal.csv')
test_values = pd.read_csv('test_values.csv')
test_id = pd.read_csv('test_values.csv')
df_labels = pd.read_csv('train_labels.csv')
y = df_labels['damage_grade']
#y = df['damage_grade']
df.columns
#df.drop(columns=['Unnamed: 0','building_id', 'damage_grade'], inplace=True)
df.drop(columns=['building_id'], inplace=True)
test_values.drop(columns=['building_id'], inplace=True)
df.dtypes
df['damage_grade'] = df_labels['damage_grade']
#df.select_dtypes(exclude = 'object').corr().sort_values(by = 'damage_grade', ascending = True)


#corr
corr = pd.DataFrame(df[['ground_floor_type_467b', 'roof_type_67f9', 'foundation_type_467b', 'has_superstructure_cement_mortar_brick',
                       'other_floor_type_67f9', 'has_superstructure_rc_engineered', 'foundation_type_858b',
                       'has_superstructure_rc_non_engineered', 'foundation_type_6c3e',
                       'area', 'other_floor_type_441a', 'secondary_use', 'count_floors_pre_eq', 'other_floor_type_f962',
                       'ground_floor_type_b1b4', 'superstructure', 'has_superstructure_mud_mortar_stone',
                       'foundation_type_337f', 'damage_grade']], columns = ['ground_floor_type_467b', 'roof_type_67f9', 'foundation_type_467b', 'has_superstructure_cement_mortar_brick',
                       'other_floor_type_67f9', 'has_superstructure_rc_engineered', 'foundation_type_858b',
                       'has_superstructure_rc_non_engineered', 'foundation_type_6c3e',
                       'area', 'other_floor_type_441a', 'secondary_use', 'count_floors_pre_eq', 'other_floor_type_f962',
                       'ground_floor_type_b1b4', 'superstructure', 'has_superstructure_mud_mortar_stone',
                       'foundation_type_337f', 'damage_grade'])

#Visualizations

import seaborn as sns
sns.set(style="ticks")


sns.pairplot(corr[['ground_floor_type_467b', 'roof_type_67f9', 'foundation_type_467b',
       'has_superstructure_mud_mortar_stone', 'foundation_type_337f', 'damage_grade'
       ]], hue="damage_grade", diag_kind = 'kde', markers = '+')

corr.dtypes

g = sns.PairGrid(corr,
                 x_vars=['ground_floor_type_467b', 'roof_type_67f9', 'foundation_type_467b',
       'has_superstructure_mud_mortar_stone', 'foundation_type_337f', 'damage_grade'
       ],
                 y_vars= 'damage_grade',
                 aspect=.75, size=3.5)
g.map(sns.swarmplot, palette="pastel");

df

#Some sata transformation

df['superstructure'] = df[['has_superstructure_adobe_mud',
                           'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                           'has_superstructure_cement_mortar_stone',
                           'has_superstructure_mud_mortar_brick',
                           'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                           'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                           'has_superstructure_rc_engineered', 'has_superstructure_other']].astype(str).sum(axis=1)

df['superstructure'] = df['superstructure'].astype('category', ordered = True).cat.codes

'''df.drop(columns=['has_superstructure_adobe_mud',
                      'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                      'has_superstructure_cement_mortar_stone',
                      'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                      'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                      'has_superstructure_rc_engineered', 'has_superstructure_other'], inplace=True)'''

df['secondary_use'] = df[['has_secondary_use',
                          'has_secondary_use_hotel', 'has_secondary_use_rental',
                          'has_secondary_use_institution']].astype(str).sum(axis = 1)

df['secondary_use'] = df['secondary_use'].astype('category', ordered = True).cat.codes

df.columns

'''df.drop(columns=['has_secondary_use', 'has_secondary_use_agriculture',
                 'has_secondary_use_hotel', 'has_secondary_use_rental',
                 'has_secondary_use_institution', 'has_secondary_use_school',
                 'has_secondary_use_industry', 'has_secondary_use_health_post',
                 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
                 'has_secondary_use_other'], inplace=True)'''

df = pd.get_dummies(df, columns = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status'])

'''df['damage_grade'] = df_labels['damage_grade']
df.corr().sort_values(by = 'damage_grade', ascending = True)
df.drop(columns = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status'], inplace = True)'''

'''df.drop(columns = ['position_bcab', 'has_secondary_use_school', 'has_secondary_use_other', 'foundation_type_bb5f', 
                   'has_secondary_use_institution', 'plan_configuration_3fee', 'plan_configuration_6e81', 'has_superstructure_other',
                   'has_secondary_use_gov_office', 'has_secondary_use_agriculture', 'has_secondary_use_use_police',
                   'position_1787', 'land_surface_condition_2f15', 'ground_floor_type_e26c', 'ground_floor_type_b440',
                   'plan_configuration_84cf', 'ground_floor_type_bb5f', 'has_superstructure_mud_mortar_brick', 'plan_configuration_0448',
                   'land_surface_condition_d502', 'legal_ownership_status_bb5f', 'plan_configuration_cb88', 
                   'other_floor_type_9eb0', 'legal_ownership_status_ab03', 'height', 'legal_ownership_status_c8e1',
                   'position_bfba', 'age', 'plan_configuration_a779', 'count_families',
                   'has_secondary_use_industry', 'position_3356', 'land_surface_condition_808e', 'plan_configuration_d2d9',
                   'geo_level_2_id', 'geo_level_3_id', 'plan_configuration_1442', 'has_secondary_use_health_post',
                   'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel', 
                   'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school',
                   'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                   'has_secondary_use_use_police', 'has_secondary_use_other'], inplace = True)'''

df.drop(columns = ['position_1787', 'land_surface_condition_2f15', 'ground_floor_type_e26c', 'ground_floor_type_b440',
                   'plan_configuration_84cf', 'ground_floor_type_bb5f', 'has_superstructure_mud_mortar_brick',
                   'plan_configuration_0448', 'land_surface_condition_d502', 'position_3356', 'land_surface_condition_808e',
                   'plan_configuration_d2d9', 'geo_level_2_id', 'geo_level_3_id', 'plan_configuration_1442',
                   'has_secondary_use_health_post', 'has_secondary_use_health_post', 'has_secondary_use_use_police'
                   ], inplace = True)
df

#Test Stuff
###########################################################################################################################################

'''test_values['superstructure'] = test_values[['has_superstructure_adobe_mud',
                           'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                           'has_superstructure_cement_mortar_stone',
                           'has_superstructure_mud_mortar_brick',
                           'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                           'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                           'has_superstructure_rc_engineered', 'has_superstructure_other']].astype(str).sum(axis=1)

test_values['superstructure'] = test_values['superstructure'].astype('category', ordered = True).cat.codes

test_values.drop(columns=['has_superstructure_adobe_mud',
                      'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                      'has_superstructure_cement_mortar_stone',
                      'has_superstructure_mud_mortar_brick',
                      'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                      'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                      'has_superstructure_rc_engineered', 'has_superstructure_other'], inplace=True)

test_values['secondary_use'] = test_values[['has_secondary_use', 'has_secondary_use_agriculture',
                          'has_secondary_use_hotel', 'has_secondary_use_rental',
                          'has_secondary_use_institution', 'has_secondary_use_school',
                          'has_secondary_use_industry', 'has_secondary_use_health_post',
                          'has_secondary_use_gov_office', 'has_secondary_use_use_police',
                          'has_secondary_use_other']].astype(str).sum(axis = 1)

test_values['secondary_use'] = test_values['secondary_use'].astype('category', ordered = True).cat.codes

test_values.drop(columns=['has_secondary_use', 'has_secondary_use_agriculture',
                 'has_secondary_use_hotel', 'has_secondary_use_rental',
                 'has_secondary_use_institution', 'has_secondary_use_school',
                 'has_secondary_use_industry', 'has_secondary_use_health_post',
                 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
                 'has_secondary_use_other'], inplace=True)'''

'''test_values.drop(columns = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status'], inplace = True)'''

#Data transformation(applying the previous steps on the test data)
#############################################################################################################################

test_values['superstructure'] = test_values[['has_superstructure_adobe_mud',
                           'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                           'has_superstructure_cement_mortar_stone',
                           'has_superstructure_mud_mortar_brick',
                           'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                           'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                           'has_superstructure_rc_engineered', 'has_superstructure_other']].astype(str).sum(axis=1)

test_values['superstructure'] = test_values['superstructure'].astype('category', ordered = True).cat.codes

'''df.drop(columns=['has_superstructure_adobe_mud',
                      'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                      'has_superstructure_cement_mortar_stone',
                      'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                      'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                      'has_superstructure_rc_engineered', 'has_superstructure_other'], inplace=True)'''

test_values['secondary_use'] = test_values[['has_secondary_use',
                          'has_secondary_use_hotel', 'has_secondary_use_rental',
                          'has_secondary_use_institution']].astype(str).sum(axis = 1)

test_values['secondary_use'] = test_values['secondary_use'].astype('category', ordered = True).cat.codes


'''test_values.drop(columns=['has_secondary_use', 'has_secondary_use_agriculture',
                 'has_secondary_use_hotel', 'has_secondary_use_rental',
                 'has_secondary_use_institution', 'has_secondary_use_school',
                 'has_secondary_use_industry', 'has_secondary_use_health_post',
                 'has_secondary_use_gov_office', 'has_secondary_use_use_police',
                 'has_secondary_use_other'], inplace=True)'''

test_values = pd.get_dummies(test_values, columns = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status'])

#test_values['damage_grade'] = df_labels['damage_grade']
#df.corr().sort_values(by = 'damage_grade', ascending = True)
'''test_values.drop(columns = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status'], inplace = True)'''

test_values.drop(columns = ['position_1787', 'land_surface_condition_2f15', 'ground_floor_type_e26c', 'ground_floor_type_b440',
                   'plan_configuration_84cf', 'ground_floor_type_bb5f', 'has_superstructure_mud_mortar_brick',
                   'plan_configuration_0448', 'land_surface_condition_d502', 'position_3356', 'land_surface_condition_808e',
                   'plan_configuration_d2d9', 'geo_level_2_id', 'geo_level_3_id', 'plan_configuration_1442',
                   'has_secondary_use_health_post', 'has_secondary_use_health_post', 'has_secondary_use_use_police'
                   ], inplace = True)




#df.drop(columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'], inplace = True)
#ZSCORES

'''dfz = df.apply(zscore)
dfz_columns = dfz.columns
for index, row in dfz.iterrows():
    for column_name  in dfz_columns:
        if row[column_name] < -2 or row[column_name] > 2:
            dfz.replace(row[column_name], np.NaN, inplace = True)

dfz.dropna(how = 'any', inplace = True)
dfz

y = y.sample(n = 2941, random_state = 2154, axis = 0)'''


'''df['age']
df['age'] = df['age'].astype('category').cat.codes
df['height'] = df['height'].astype('category').cat.codes
test_values['age'] = test_values['age'].astype('category').cat.codes
test_values['height'] = test_values['height'].astype('category').cat.codes
df['area'] = df['area'].astype('category').cat.codes
df['count_floors_pre_eq'] = df['count_floors_pre_eq'].astype('category').cat.codes
df['count_families'] = df['count_families'].astype('category')'''


'''df['geo_level'] = df['geo_level_1_id'] + df['geo_level_2_id'] + df['geo_level_3_id']
df['geo_level'] = df['geo_level'].astype('category').cat.codes
df.drop(columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'], inplace = True)
'''

'''test_values['age']
test_values['age'] = test_values['age'].astype('category').cat.codes
test_values['height'] = test_values['height'].astype('category').cat.codes
test_values['area'] = test_values['area'].astype('category').cat.codes
test_values['count_floors_pre_eq'] = test_values['count_floors_pre_eq'].astype('category').cat.codes
test_values['count_families'] = test_values['count_families'].astype('category')
test_values['geo_level'] = test_values[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']].astype(str).sum(axis = 1)
test_values['geo_level'] = test_values['geo_level'].astype('category').cat.codes
df'''


'''#  Visualization
COL_NUM = 5
ROW_NUM = 8
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12, 12))

for i, (label, col) in enumerate(df.iteritems()):
    ax = axes[int(i / COL_NUM), i % COL_NUM]
    col = col.sort_values(ascending=False)
    col.plot(kind='hist', ax=ax)
    ax.set_title(label)

plt.tight_layout()

df.dtypes'''

'''from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=12)
x_resampled, y_resampled = ros.fit_sample(df, y)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
y_resampled = pd.DataFrame(y_resampled, columns = ['damage_grade'])
x_resampled = pd.DataFrame(x_resampled, columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
       'count_floors_pre_eq', 'age', 'area', 'height', 'count_families',
       'superstructure', 'secondary_use'])
y_resampled.shape
X_resampled.shape'''


'''from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
x_resampled, y_resampled = SMOTE(kind = 'svm').fit_sample(df, y)
print(sorted(Counter(y_resampled).items()))
y_resampled = pd.DataFrame(y_resampled, columns = ['damage_grade'])
x_resampled = pd.DataFrame(x_resampled, columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
       'count_floors_pre_eq', 'age', 'area', 'height', 'count_families',
       'superstructure', 'secondary_use'])'''
    
    
#Oversampling tests
########################################################################################################################################
    
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
x_resampled, y_resampled = SMOTE(kind = 'svm').fit_sample(df, y)
print(sorted(Counter(y_resampled).items()))
y_resampled = pd.DataFrame(y_resampled, columns = ['damage_grade'])
x_resampled = pd.DataFrame(x_resampled, columns = ['geo_level_1_id', 'count_floors_pre_eq', 'age', 'area', 'height',
       'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
       'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'count_families', 'has_secondary_use', 'has_secondary_use_agriculture',
       'has_secondary_use_hotel', 'has_secondary_use_rental',
       'has_secondary_use_institution', 'has_secondary_use_school',
       'has_secondary_use_industry', 'has_secondary_use_gov_office',
       'has_secondary_use_other', 'superstructure', 'secondary_use',
       'foundation_type_337f', 'foundation_type_467b', 'foundation_type_6c3e',
       'foundation_type_858b', 'foundation_type_bb5f', 'roof_type_67f9',
       'roof_type_7e76', 'roof_type_e0e2', 'ground_floor_type_467b',
       'ground_floor_type_b1b4', 'other_floor_type_441a',
       'other_floor_type_67f9', 'other_floor_type_9eb0',
       'other_floor_type_f962', 'position_bcab', 'position_bfba',
       'plan_configuration_3fee', 'plan_configuration_6e81',
       'plan_configuration_8e3f', 'plan_configuration_a779',
       'plan_configuration_cb88', 'legal_ownership_status_ab03',
       'legal_ownership_status_bb5f', 'legal_ownership_status_c8e1',
       'legal_ownership_status_cae1'])    

    
y_resampled    


'''x_resampled, y_resampled = ADASYN().fit_sample(df, y)
print(sorted(Counter(y_resampled).items()))
y_resampled = pd.DataFrame(y_resampled, columns = ['damage_grade'])
x_resampled = pd.DataFrame(x_resampled, columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
       'count_floors_pre_eq', 'age', 'area', 'height', 'count_families',
       'superstructure', 'secondary_use'])'''


#y_resampled = pd.DataFrame(y_resampled, columns = ['Damage_grade'])
"""X_resampled = pd.DataFrame(X_resampled, columns = ['land_surface_condition_2f15', 'land_surface_condition_808e',
       'land_surface_condition_d502', 'foundation_type_337f',
       'foundation_type_467b', 'foundation_type_6c3e', 'foundation_type_858b',
       'foundation_type_bb5f', 'roof_type_67f9', 'roof_type_7e76',
       'roof_type_e0e2', 'ground_floor_type_467b', 'ground_floor_type_b1b4',
       'ground_floor_type_b440', 'ground_floor_type_bb5f',
       'ground_floor_type_e26c', 'other_floor_type_441a',
       'other_floor_type_67f9', 'other_floor_type_9eb0',
       'other_floor_type_f962', 'position_1787', 'position_3356',
       'position_bcab', 'position_bfba', 'plan_configuration_0448',
       'plan_configuration_1442', 'plan_configuration_3fee',
       'plan_configuration_6e81', 'plan_configuration_84cf',
       'plan_configuration_8e3f', 'plan_configuration_a779',
       'plan_configuration_cb88', 'plan_configuration_d2d9',
       'legal_ownership_status_ab03', 'legal_ownership_status_bb5f',
       'legal_ownership_status_c8e1', 'legal_ownership_status_cae1'])"""
    
#Take a random sample of the oversampled new DF with the same numberof rows in the original DF to avoid overfitting
y_sample = y_resampled.sample(n = 10000, random_state = 4561, axis = 0)
x_sample = x_resampled.sample(n = 10000, random_state = 4561, axis = 0)
y_sample.plot(kind='hist')

y_sample
x_sample
x_resampled
y_resampled

#Testing split methods
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
'''x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.30, random_state = 12,
                                                    shuffle = True)'''

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1245)
for train_index, test_index in sss.split(x_sample, y_sample):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x_sample.iloc[train_index], x_sample.iloc[test_index]
    y_train, y_test = y_sample.iloc[train_index], y_sample.iloc[test_index]


'''sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1245)
for train_index, test_index in sss.split(x_resampled, y_resampled):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x_resampled.iloc[train_index], x_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]'''
    
#Defining cross validation methods for test
Cval = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 4587)
kf_total = cval.KFold(len(x_train), n_folds=10, shuffle=True, random_state=4)


#Try different algorithms

#XGBOOST

from xgboost import XGBClassifier

xgb = XGBClassifier(randon_state = 87)
xgb.fit(x_train, y_train)
xgb.score(x_test, y_test)

#VOTING CLASSIFIER
from sklearn.ensemble import VotingClassifier

vot = VotingClassifier(voting = 'soft', n_jobs = 1, estimators = [('rf', for_search), ('gr', rand_search),
                                                                  ('tree', tree_search)])

param_grid =  { 'voting': ['soft', 'hard'],
               'flatten_transform': [None, True, False]
                
}

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
            
n_iter_search = 6


#NEST THE CLASSIFIERS(TEST)
vote = OneVsRestClassifier(VotingClassifier(voting = 'soft', n_jobs = 1, estimators = [('rf', tuned_for), ('gr', tuned_clf),
                                                                  ('tree', tuned_tree)]), n_jobs = -1)

vote_search= RandomizedSearchCV(vote, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = 5, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)


tuned_vote = OneVsRestClassifier(VotingClassifier(voting = 'soft', flatten_transform = False, n_jobs = -1, estimators = [('rf', tuned_for), ('gr', tuned_clf),
                                                                  ('tree', tuned_tree)]), n_jobs = -1)



#GETTING THE TRAINING TIME    
start = time()
vote_search.fit(x_sample, y_sample)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(for_search.cv_results_)







                                                                  
tuned_vote.fit(x_train, y_train)
tuned_vote.score(x_test, y_test)




#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

clffor = RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_depth = 30, min_samples_leaf = 15,
                             bootstrap = False, class_weight = 'balanced', random_state = 4563, n_jobs = -1)

clffor.fit(x_train, y_train)
clffor.score(x_test, y_test)


#PARAMETER OPTIMIZATION WITH CROSS VALIDATION

Cval = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 4587)
kf_total = cval.KFold(len(x_train), n_folds=10, shuffle=True, random_state=4)


param_grid =  { 'n_estimators':[ 200, 300, 600],
                "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': [None, 'auto', 'sqrt'],
                'bootstrap': [True, False]
                
}

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
            
n_iter_search = 100

clffor = RandomForestClassifier()

for_search= RandomizedSearchCV(clffor, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = 5, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

'''for_search = GridSearchCV(clffor, param_grid = param_grid, scoring = 'f1_micro', n_jobs = -1, cv = Cval,
                           return_train_score = True, iid = True, refit = True)'''


tuned_for = OneVsRestClassifier(RandomForestClassifier(n_estimators = 600, min_samples_split = 2, min_samples_leaf = 2,
                                   max_features = 'auto', max_depth = 50, bootstrap = False,
                                   criterion = 'entropy', class_weight = 'balanced', random_state = 87, n_jobs = 5), n_jobs = -1)

start = time()
for_search.fit(x_sample, y_sample)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(for_search.cv_results_)

tuned_for.fit(x_train, y_train)
tuned_for.score(x_test, y_test)
prediction = clf.predict(x_test)
f1_score(y_test, prediction, average = 'micro')


#GRADIENT BOOSTING CLASSIFIER

from sklearn.ensemble import GradientBoostingClassifier

gr_clf = OneVsRestClassifier(BaggingClassifier(GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                    n_estimators=100, subsample=1.0,
                                    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=10,
                                    min_weight_fraction_leaf=0.0, max_depth=30, min_impurity_decrease=0.0,
                                    min_impurity_split=None, init=None, random_state=None,
                                    max_features= 'auto',
                                    verbose=0, max_leaf_nodes=None, warm_start=True, presort='auto'),
                                    oob_score = True, n_jobs = -1, random_state = 1234, n_estimators = 40), n_jobs = 1)

gr_clf1 = BaggingClassifier(OneVsRestClassifier(GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                    n_estimators=200, subsample=1.0,
                                    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=10,
                                    min_weight_fraction_leaf=0.0, max_depth=30, min_impurity_decrease=0.0,
                                    min_impurity_split=None,
                                    verbose=0, max_leaf_nodes=None, warm_start=True, presort='auto'),
                                    n_jobs = 1), n_estimators = 40, oob_score = True, n_jobs = -1, random_state = 1234)

gr = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 300, random_state = 87),n_jobs=-1)


#CROSS VALIDATION
from sklearn.model_selection import RandomizedSearchCV

param_grid = param_grid =  {  
                
                
                "max_depth": range(6, 100),
                "criterion": ["friedman_mse", "mae"],
                'max_features': [None, 'auto', 'log2'],
                'max_leaf_nodes': range(10, 120),
                'warm_start': [True, False]
                }

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




gr.fit(x_train,y_train)
gr.score(x_test, y_test)
prediction = clf.predict(test_values)
test_values
x_sample
#gr_clf.score(x_test, y_test)
f1_score(y_test, prediction, average = 'micro')



#EXTRA TREES
from sklearn.ensemble import ExtraTreesClassifier

Cval = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 4587)
kf_total = cval.KFold(len(x_train), n_folds=10, shuffle=True, random_state=4)


'''tree1 = BaggingClassifier(OneVsRestClassifier(ExtraTreesClassifier(n_estimators = 200, criterion = 'gini', max_features = 'auto',
                            bootstrap = True, oob_score = True, n_jobs = -1,
                            random_state = 1245, class_weight = 'balanced'),n_jobs = 1), n_estimators = 400, oob_score = True, n_jobs = 4,
                             random_state = 1234)'''

param_grid =  { 'n_estimators':[200, 300],
                "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': [None, 'auto', 'sqrt'],
                'bootstrap':[True, False],
                'criterion':['gini', 'entropy']
                
}


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
            
n_iter_search = 60

tree = ExtraTreesClassifier( 
                            oob_score = False, n_jobs = 1,
                            random_state = 1245, class_weight = 'balanced')
            


tree_search = RandomizedSearchCV(tree, param_distributions = param_grid, n_iter = n_iter_search, scoring = 'f1_micro',
                                 n_jobs = 5, iid = False, cv = Cval, random_state = 1254, return_train_score = True
                                 )


tuned_tree = OneVsRestClassifier(ExtraTreesClassifier(oob_score = False, n_jobs = 1,
                            random_state = 0, class_weight = 'balanced', n_estimators = 300,
                            min_samples_split = 5, min_samples_leaf = 2, max_features = None,
                            max_depth = 110, criterion = 'entropy', bootstrap = True), n_jobs = -1)



start = time()
tree_search.fit(x_sample, y_sample)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(tree_search.cv_results_)

tuned_tree.fit(x_train, y_train)
tuned_tree.score(x_test, y_test)


#TEST STUFF
###################################################################################################################################################

param_grid =  { 
                'base_estimator': [tuned_tree],
                
                
}


one =  OneVsRestClassifier(estimator = tuned_tree,n_jobs = 5)
n_iter_search = 1

ada_search = RandomizedSearchCV(ada, param_distributions = param_grid, n_iter = n_iter_search, scoring = 'f1_micro',
                                 n_jobs = 5, iid = False, cv = Cval, random_state = 1254, return_train_score = True
                                 )

start = time()
ada_search.fit(x_sample, y_sample)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(ada_search.cv_results_)




# 1ST SUBMISSION ATEMPT
prediction = pd.DataFrame(prediction, columns = ['damage_grade'])
prediction.set_index(test_id['building_id'], inplace = True)
prediction
prediction.to_csv('prediction20.csv')
#test_values.drop(test_values.index[1000:len(test_values)],inplace = True)
prediction.set_index(test_values['building_id'], inplace = True)
#prediction.drop(prediction.index[1000:len(prediction)], inplace = True)
prediction.to_csv('prediction.csv')
####################################################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn import cross_validation as cval
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


#TRY MORE THINGS

Cval = StratifiedKFold( n_splits = 10, shuffle = True, random_state = 4587)
kf_total = cval.KFold(len(x_train), n_folds=10, shuffle=True, random_state=4)

'''score, permutation_scores, pvalue = permutation_test_score(
    gr_clf, x_sample, y_sample, scoring="f1_micro", cv=Cval, n_permutations=100, n_jobs=8)
print("Classification score %s (pvalue : %s)" % (score, pvalue))'''

param_grid =  { 'n_estimators':[100, 200, 300],
                "max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': [None, 'auto', 'sqrt'],
                'random_state': [245]
                
}

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
            
n_iter_search = 60

clf = GradientBoostingClassifier()

rand_search= RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = 5, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

"""rand_search = GridSearchCV(clf, param_grid = param_grid, scoring = 'f1_micro', n_jobs = 7, cv = kf_total,
                           return_train_score = True)"""

start = time()
rand_search.fit(x_sample, y_sample)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(rand_search.cv_results_)
rand_search.score(x_test, y_test)

tuned_clf = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 300, warm_start = False, min_samples_split = 2,
                                       min_impurity_decrease = 0.2,
                                       max_leaf_nodes = 89, max_features = 'sqrt', max_depth = 10,
                                       min_samples_leaf = 1, criterion = 'friedman_mse', presort = 'auto'), n_jobs = -1)

ada = OneVsRestClassifier(estimator = tuned_tree)
                                                                      
ada.fit(x_train, y_train)
ada.score(x_test, y_test)

tuned_clf = OneVsRestClassifier(rand_search, n_jobs = -1)

tuned_clf.fit(x_train, y_train)
tuned_clf.score(x_test, y_test)
prediction = tuned_clf.predict(test_values)
prediction = rand_search.predict(test_values)

#BAGGING CLASSIFIER

from sklearn.ensemble import BaggingClassifier


param_grid =  {  
                
                
                "base_estimator": [tuned_clf],
                "max_samples": [0.7, 0.8, 1],
                'max_features': [0.8, 1],
                'bootstrap': [True],
                'bootstrap_features': [True, False],
                'oob_score': [True, False],
                'warm_start': [ False]}

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
            
n_iter_search = 24

clf = BaggingClassifier()

rand_search= RandomizedSearchCV(clf, param_distributions=param_grid,
                                   n_iter=n_iter_search, scoring = 'f1_micro', n_jobs = -1, iid= False,
                                   cv = Cval, random_state = 4521, return_train_score= True)

"""rand_search = GridSearchCV(clf, param_grid = param_grid, scoring = 'f1_micro', n_jobs = 7, cv = kf_total,
                           return_train_score = True)"""

start = time()
rand_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(rand_search.cv_results_)







x_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)


x_test


#SUBMISSION SETUP

tuned_clf.fit(x_train, y_train)
tuned_clf.score(x_test, y_test)
results = cval.cross_val_score(gr_clf, x_train, y_train, scoring = 'f1_micro', cv = kf_total, n_jobs = -1)
results
cpred = cval.cross_val_predict(gr_clf, x_test, y_test, cv = kf_total, n_jobs = -1)
cpred
cpred = pd.DataFrame(prediction, columns = ['damage_grade'])
cpred.set_index(test_values['building_id'], inplace = True)
cpred.to_csv('prediction.csv')
tuned_clf.fit(x_sample, y_sample)
tuned_clf.score(x_test, y_test)
prediction = tuned_clf.predict(test_values)
prediction = pd.DataFrame(prediction, columns = ['damage_grade'])
prediction.set_index(test_id['building_id'], inplace = True)
prediction.to_csv('prediction.csv')
x_sample


#MORE TEST STUFF
x_sample.corr()
x_sample.drop(columns = ['secondary_use'], inplace = True)
x_sample.drop(columns = ['count_families'], inplace = True)


x_sample['base'] = x_sample['area'] / x_sample['height']
x_sample['perimeter'] = 2*x_sample['base'] + 2*x_sample['height']

test_values.drop(columns = ['secondary_use'], inplace = True)
test_values.drop(columns = ['count_families'], inplace = True)

test_values['base'] = test_values['area'] / test_values['height']
test_values['perimeter'] = (2*test_values['base']) + (2*test_values['height'])

##########################################################################################

#PREPROCESSING TEST(DISCARDED)
y = pd.get_dummies(y)
y

model = OneVsRestClassifier(tuned_clf, n_jobs = -1)

results = cval.cross_val_score(model, x_train, y_train, scoring = 'f1_micro', cv = kf_total, n_jobs = -1)
results
model.fit(x_train, y_train)
model.score(x_test, y_test)









# Sify the columns to encode then fit and transform
encoder = ce.polynomial.PolynomialEncoder(
    cols=['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
          'position', 'plan_configuration', 'legal_ownership_status', 'legal_ownership_status',
          'has_superstructure_adobe_mud',
          'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
          'has_superstructure_cement_mortar_stone',
          'has_superstructure_mud_mortar_brick',
          'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
          'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
          'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
          'has_secondary_use_agriculture', 'has_secondary_use_hotel',
          'has_secondary_use_rental', 'has_secondary_use_institution',
          'has_secondary_use_school', 'has_secondary_use_industry',
          'has_secondary_use_health_post', 'has_secondary_use_gov_office',
          'has_secondary_use_use_police', 'has_secondary_use_other'])
encoder_test = ce.polynomial.PolynomialEncoder(
    cols=test_values[['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
                      'position', 'plan_configuration', 'legal_ownership_status', 'legal_ownership_status',
                      'has_superstructure_adobe_mud',
                      'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                      'has_superstructure_cement_mortar_stone',
                      'has_superstructure_mud_mortar_brick',
                      'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                      'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                      'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families',
                      'has_secondary_use',
                      'has_secondary_use_agriculture', 'has_secondary_use_hotel',
                      'has_secondary_use_rental', 'has_secondary_use_institution',
                      'has_secondary_use_school', 'has_secondary_use_industry',
                      'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                      'has_secondary_use_use_police', 'has_secondary_use_other']])
encoder.fit(df, verbose=1)
encoder_test.fit(
    test_values[['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
                 'position', 'plan_configuration', 'legal_ownership_status', 'legal_ownership_status',
                 'has_superstructure_adobe_mud',
                 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                 'has_superstructure_cement_mortar_stone',
                 'has_superstructure_mud_mortar_brick',
                 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                 'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
                 'has_secondary_use_agriculture', 'has_secondary_use_hotel',
                 'has_secondary_use_rental', 'has_secondary_use_institution',
                 'has_secondary_use_school', 'has_secondary_use_industry',
                 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                 'has_secondary_use_use_police', 'has_secondary_use_other']], verbose=1)

# Only display the first 8 columns for brevity
trans_df = encoder.transform(df)
trans_df.describe()
encoder_test.transform(
    test_values[['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
                 'position', 'plan_configuration', 'legal_ownership_status', 'has_superstructure_adobe_mud',
                 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                 'has_superstructure_cement_mortar_stone',
                 'has_superstructure_mud_mortar_brick',
                 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                 'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use',
                 'has_secondary_use_agriculture', 'has_secondary_use_hotel',
                 'has_secondary_use_rental', 'has_secondary_use_institution',
                 'has_secondary_use_school', 'has_secondary_use_industry',
                 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
                 'has_secondary_use_use_police', 'has_secondary_use_other']])
    
#SAVE PREPROCESSED DATA    

from sklearn import preprocessing as pr

scaler = pr.MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=[df.columns])
scaled_df.columns
trans_df.describe
scaled_df.to_csv('C:/Users/bauer/Documents/EAE/Data Science course/trans_df.csv')

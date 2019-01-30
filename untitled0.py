# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:51:40 2019

@author: The Freaky Gamer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv("train.csv")

dataset = dataset.drop(['Alley'], axis=1)
dataset = dataset.drop(['Utilities'], axis=1)
dataset = dataset.drop(['Id'], axis=1)
dataset = dataset.drop(['PoolQC'], axis=1)
dataset = dataset.drop(['PoolArea'], axis=1)

#dataset['LotFrontage'] = dataset['LotFrontage'].fillna(0)

#a = dataset[['LotFrontage', 'LotArea']]

'''for index, rows in a.iterrows():
    print("{} {}".format(index, rows))

a = dataset.count()'''
'''
Try to find the value of lot frontage using regression on lot area, but lot of nan values so drop lot frontage'''


'''x = np.zeros([1460, 1])
y = np.zeros([1460, 1])
k=0

for i in range(20):
    print(dataset.iloc[i, 2]==0)
    if dataset.iloc[i, 2] !=0.0:
        x[k] = dataset.iloc[i, 2]
        y[k] = dataset.iloc[i, 3]
        k+=1

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(y, x)

pred = np.empty([1460, 1])

for i in range(1460):
    if dataset.iloc[i, 2] == 0:
        pred[i] = lr.predict(dataset.iloc[i, 3])'''


dataset = dataset.drop(['LotFrontage'], axis=1)
dataset = dataset.drop(['3SsnPorch'], axis=1)
dataset = dataset.drop(['ScreenPorch'], axis=1)
dataset = dataset.drop(['LowQualFinSF'], axis=1)
dataset = dataset.drop(['MiscFeature'], axis=1)


l = ['BsmtQual', 'Fence', 'FireplaceQu', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond']
dataset[l] = dataset[l].replace(np.nan, 'Unknown')

dataset['BsmtExposure'] = dataset['BsmtExposure'].replace(np.nan, 'Unknown')

'''lb = LabelEncoder()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

Xdf = pd.DataFrame(X)
ydf = pd.DataFrame(y)

categ = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 35,
         36, 37, 38, 48, 50, 52, 53, 55, 58, 59, 60, 64, 68, 69]

X[:, categ] = lb.fit_transform(X[:, categ])

ohe = OneHotEncoder(categorical_features=)

X = ohe.fit(X)


temp= X[:, 1:2]
tempdf = pd.DataFrame(temp)
oh2 = OneHotEncoder(categorical_features=[0])
temp = oh2.fit_transform(temp).toarray()'''

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset = pd.DataFrame(X)
num = [53, 55, 58, 59, 21]

dataset[num] = dataset[num].replace(np.nan, 'Unknown')
dataset[[54, 22]] = dataset[[54, 22]].replace(np.nan, 0)
categ = [1,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 35,
         36, 37, 38, 48, 50, 52, 53, 55, 58, 59, 60, 64, 68, 69]




one_hot_encoded = pd.get_dummies(dataset, columns=categ)








test = pd.read_csv('test.csv')

test = test.drop(['Alley'], axis=1)
test = test.drop(['Utilities'], axis=1)
test = test.drop(['Id'], axis=1)
test = test.drop(['PoolQC'], axis=1)
test = test.drop(['PoolArea'], axis=1)
test = test.drop(['LotFrontage'], axis=1)
test = test.drop(['3SsnPorch'], axis=1)
test = test.drop(['ScreenPorch'], axis=1)
test = test.drop(['LowQualFinSF'], axis=1)
test = test.drop(['MiscFeature'], axis=1)

l = ['BsmtQual', 'Fence', 'FireplaceQu', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond']
test[l] = test[l].replace(np.nan, 'Unknown')

test['BsmtExposure'] = test['BsmtExposure'].replace(np.nan, 'Unknown')

num = [53, 55, 58, 59, 21, 1, 19, 20, 68, 48, 50]
X = test.iloc[:, :].values
test = pd.DataFrame(X)
test[num] = test[num].replace(np.nan, 'Unknown')
test[[54, 22, 30, 32, 33, 34, 51]] = test[[54, 22, 30, 32, 33, 34, 51]].replace(np.nan, 0)
categ = [1,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 35,
         36, 37, 38, 48, 50, 52, 53, 55, 58, 59, 60, 64, 68, 69]


test[[30,32,33,34, 42, 43, 56, 57]] = test[[30,32,33,34, 42, 43, 56, 57]].replace(np.nan, 0)

one_hot_encoded2 = pd.get_dummies(test, columns=categ)
final_train, final_test = one_hot_encoded.align(one_hot_encoded2,join='left', axis=1)


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()
reg.fit(final_train, y)

final_test = final_test.fillna(0)

y_pred = reg.predict(final_test)


df_out = pd.DataFrame({'Id':test['Id'], 'SalePrice':y_pred})

df_out.to_csv("out.csv")
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import getsizeof
import xgboost as xgb
# 对字段的类型进行处理，使其占用内存更少
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
def getData():
    dataIterator = pd.read_csv('G:/tempdata/playersPay/tap4fun/tap_fun_train.csv',iterator=True)
    data = dataIterator.get_chunk(10000)
    data_int = data.select_dtypes(include=['int64'])
    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
    data[converted_int.columns] = converted_int
    dtypes = data.dtypes
    dtype_index = dtypes.index
    dtype_type = [i.name for i in dtypes]
    columns_type = dict(zip(dtype_index,dtype_type))

    data = pd.read_csv('G:/tempdata/playersPay/tap4fun/tap_fun_train.csv',dtype=columns_type)
    data.drop(['register_time'],axis=1,inplace=True)
    return data
data = getData()
def getLable(x):
    if x > 0:
        return 1
    else:
        return 0
# print(data.columns)
data['prediction_pay_price'] = data['prediction_pay_price'].apply(getLable)
y = data['prediction_pay_price'].astype('int')
x = data.drop('prediction_pay_price',axis=1)
colum = ['ivory_add_value','ivory_reduce_value','general_acceleration_reduce_value',
         'bd_dolmen_level','training_acceleration_reduce_value','pvp_lanch_count',
         'stone_add_value','treatment_acceleraion_add_value','sr_training_speed_level',
         'stone_reduce_value','building_acceleration_add_value','wood_reduce_value',
         'pay_price','avg_online_minutes','reaserch_acceleration_add_value']
x = data[colum]
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
def svmPro():
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train,y_train)
    print(clf.score(x_train, y_train))  # 精度
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

def xgbclassfier():
    import operator
    from sklearn.tree import DecisionTreeClassifier
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    param = {'max_depth': 4, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    bst.save_model('xgb.model')
    y_hat = bst.predict(data_test)
    y_hat[y_hat>0.5]=1
    y_hat[y_hat<=0.5]=0

    # 获取特征重要程度
    def getImportFeature():

        importance = bst.get_fscore()
        ceate_feature_map(x_train.columns)
        importance = bst.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df.to_csv("feat_importance.csv", index=False)
        plt.figure()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.savefig('feat_importance.png')
        plt.show()

    result = y_test == y_hat
    print(len(y_hat[y_hat==1]))
    print(len(y_hat[y_hat ==0]))
    print('正确率:\t', float(np.sum(result)) / len(y_hat))

def randomForest():
    from sklearn.tree import DecisionTreeClassifier

def regreseion():
    # train = getData()
    # data = train[train['prediction_pay_price']>0]
    # data.to_csv('havePay.csv')
    data = pd.read_csv('havePay.csv')
    x = data[colum]
    y = data['prediction_pay_price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    params = {
        'min_child_weight': 5,
        'eta': 0.5,
        'colsample_bytree': 0.6,
        'max_depth': 7,
        'subsample': 0.9,
        'alpha': 0.04,
        'gamma': 0,
        'silent': 0,
        'verbose_eval': True,
        'seed': 12
    }
    rounds = 10
    xgtrain = xgb.DMatrix(x_train, label=y_train)
    xgtest = xgb.DMatrix(x_test)
    print('Trainning')
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)
    y_hat = bst.predict(xgtest)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)
# 获取
regreseion()
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

assign2_df_train = pd.read_csv('assignment2train.csv')

Y = assign2_df_train['meal']
X = assign2_df_train.drop(columns=['meal','id','DateTime'])

assign2_df_test = pd.read_csv('assignment2test.csv')
xt = assign2_df_test.drop(columns=['meal','id','DateTime'])

assign2_df_truth = pd.read_csv('/Users/hugocorado/Documents/GitHub/econ8310-assignment2/tests/testData.csv')
yt = assign2_df_truth['meal']

num_classes_train = len(np.unique(Y))

model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.5, objective='multi:softmax', num_class=num_classes_train)

modelFit = model.fit(X, Y)

pred = modelFit.predict(xt)
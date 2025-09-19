import pandas as pd
import numpy as np
from xgboost import XGBClassifier

assign2_df_train = pd.read_csv('assignment2train.csv')

Y = assign2_df_train['meal']
X = assign2_df_train.drop(columns=['meal','id','DateTime'])

assign2_df_test = pd.read_csv('assignment2test.csv')
xt = assign2_df_test.drop(columns=['meal','id','DateTime'])

num_classes_train = len(np.unique(Y))

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.5, objective='multi:softmax', num_class=num_classes_train)

modelFit = model.fit(X, Y)

pred = modelFit.predict(xt)

pred = np.array(pred)

pred = np.round(pred).astype(float)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('nba_logreg.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 20].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1340,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

import statsmodels.formula.api as sm
X_opt = X[:, [0, 1, 2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X = X_opt

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='binary')  
# F1 score is approximately 0.70
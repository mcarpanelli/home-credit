from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pipeline import Getdummies, Logarize, DatetoYear, SelectColumns, ReplaceNaN, Square, Interactify, Multiplarify

# Read files
X = pd.read_csv('data/application_train.csv')
X_test = pd.read_csv('data/application_test.csv')
y = X['TARGET']

# Build pipeline
p = Pipeline([
    ('fillna',ReplaceNaN()),
    ('getdummies',Getdummies()),
    ('log',Logarize()),
    ('datetoyear',DatetoYear()),
    ('square', Square()),
    ('multiples', Multiplarify()),
    ('interactions', Interactify()),
    ('select',SelectColumns()),
    ('rf', RandomForestClassifier())
])

# Run model(s)
X = X.reset_index()
X = X.drop(['TARGET'], axis = 1)

params = {'rf__n_estimators':[100, 500], 'rf__max_depth':[5,8], 'rf__max_features': [5,6]}
gscv = GridSearchCV(p, params,
                    cv=3,
                    scoring = 'roc_auc',
                    n_jobs=2)
clf = gscv.fit(X, y)
probabilities = clf.predict_proba(X_test)

# Create Submission
print('Best parameters: {}'.format(clf.best_params_))
print('Best AUC: {}'.format(clf.best_score_))
#  Best parameters: {'rf__max_features': 6, 'rf__n_estimators': 500, 'rf__max_depth': 8}

submit = X_test[['SK_ID_CURR']]
submit['TARGET'] = probabilities[:, 1]
# submit.head()
submit.to_csv('balls_solutions.csv', index = False)


# Code to debug individual pipeline classes

# p.fit(X,y)
# print(p.transform(X))

# g = Getdummies()
# model = g.fit(X,y)
# model.transform(X)

# print(X.columns)
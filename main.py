from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from pipeline import GetDummies, BuildVariables, Logarize, DatetoYear, SelectColumns, ReplaceNaN, Squarify, Interactify, Multiplarify

# Read files
X = pd.read_csv('data/application_train.csv')
X_test = pd.read_csv('data/application_test.csv')
y = X['TARGET']

# Build pipeline
p = Pipeline([
    ('fillna',ReplaceNaN()),
    ('get_dummies',GetDummies()),
    ('build_variables', BuildVariables()),
    ('log',Logarize()),
    ('datetoyear',DatetoYear()),
    ('squares', Squarify()),
    ('multiples', Multiplarify()),
    ('interactions', Interactify()),
    ('select', SelectColumns()),
    ('gb', GradientBoostingClassifier())
    # ('rf', RandomForestClassifier())
])

# Run model(s)
X = X.reset_index()
X = X.drop(['TARGET'], axis = 1)

# params = {'rf__n_estimators':[500], 'rf__max_depth':[8], 'rf__max_features': [6,10]}
params = {'gb__n_estimators':[500], 'gb__max_depth':[3], 'gb__learning_rate': [0.1]}

gscv = GridSearchCV(p, params,
                    cv=3,
                    scoring = 'roc_auc',
                    n_jobs=2)
clf = gscv.fit(X, y)
probabilities = clf.predict_proba(X_test)

# Create Submission
print('Best parameters: {}'.format(clf.best_params_))
print('Best AUC: {}'.format(clf.best_score_))
# Best parameters: {'rf__max_features': 6, 'rf__n_estimators': 500, 'rf__max_depth': 8}
# Best parameters: {'gb__n_estimators': 500, 'gb__max_depth': 3, 'gb__learning_rate': 0.1}

submit = X_test[['SK_ID_CURR']]
submit['TARGET'] = probabilities[:, 1]
# submit.head()
submit.to_csv('GB_07-11-2018.csv', index = False)


# Code to debug individual pipeline classes

# p.fit(X,y)
# print(p.transform(X))

# g = Getdummies()
# model = g.fit(X,y)
# model.transform(X)

# print(X.columns)
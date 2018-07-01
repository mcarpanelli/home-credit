from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

family=['NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
      'NAME_FAMILY_STATUS_Single / not married']

education = ['NAME_EDUCATION_TYPE_Higher education',
      'NAME_EDUCATION_TYPE_Incomplete higher',
      'NAME_EDUCATION_TYPE_Lower secondary',
      'NAME_EDUCATION_TYPE_Secondary / secondary special']

housing = ['NAME_HOUSING_TYPE_Municipal apartment',
      'NAME_HOUSING_TYPE_Office apartment',
      'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents']

contract = ['NAME_CONTRACT_TYPE_Revolving loans']

gender = ['CODE_GENDER_M']

selected_for_dummies = {
    'CODE_GENDER': gender,
    'NAME_EDUCATION_TYPE':education,
    'NAME_FAMILY_STATUS':family,
    # 'NAME_TYPE_SUITE':[],
    # 'NAME_INCOME_TYPE':[],
    'NAME_CONTRACT_TYPE': contract,
    'NAME_HOUSING_TYPE':housing,
    # 'OCCUPATION_TYPE':[],
    'FLAG_OWN_CAR':['FLAG_OWN_CAR_Y'],
    'FLAG_OWN_REALTY':['FLAG_OWN_REALTY_Y']
    }

class SelectColumns(BaseEstimator, TransformerMixin):
    """Only keep columns that we want to keep.
    """
    keep_cols = list(set(['DAYS_EMPLOYED_year', 'DAYS_BIRTH_year',
                'DAYS_REGISTRATION_year', 'DAYS_BIRTH_year_square', 'FLAG_OWN_CAR_Y', 
                'FLAG_OWN_REALTY_Y', 'AMT_CREDIT_log', 'AMT_GOODS_PRICE_log', 'AMT_ANNUITY_log', 'FLAG_OWN_CAR_Y_OWN_CAR_AGE', 'AMT_INCOME_TOTAL_log',
                'AMT_CREDIT_ratio', 'AMT_GOODS_PRICE_ratio', 'AMT_ANNUITY_ratio',
                'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] + family + education + housing + gender + contract))
#    list(selected_for_dummies.keys())
    # drop_cols = []

    def fit(self, X, y):
        return self

    def transform(self, X):
        # X = X.loc[:, self.keep_cols]]]
        X = X[self.keep_cols]
        return X


class Getdummies(BaseEstimator, TransformerMixin):

    def fit(self,X,y):
    #X is a dataframe
        return self

    def transform(self,X):
        for col in selected_for_dummies.keys():
            dummies = pd.get_dummies(X[col],prefix=col)
            # selected_cols = selected_for_dummies[col] #['Male']
            # selected_dummies = dummies[selected_cols]
            # X[selected_dummies.columns] = selected_dummies
            X[dummies.columns] = dummies
        return X


class Logarize(BaseEstimator, TransformerMixin):

    columns_to_log = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_log:
            X[col + '_log'] = np.log(X[col])
        return X


class Square(BaseEstimator, TransformerMixin):

    columns_to_square = ['DAYS_BIRTH_year']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_square:
            X[col + '_square'] = X[col]**2
        return X


class Multiplarify(BaseEstimator, TransformerMixin):

    numerators = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.numerators:
            X[col + '_ratio'] = X[col] / X['AMT_INCOME_TOTAL']
        return X

class DatetoYear(BaseEstimator, TransformerMixin):

    columns_to_convert = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_REGISTRATION']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_convert:
            X[col + '_year'] = X[col] / (-365)
        return X


# class DataType(BaseEstimator, TransformerMixin):
#     col_types = {}
#     def fit(self,X,y):
#         return self
#     def transform(self,X):
#         for col_type, column in self.col_types.items():
#             X[column] = X[column].astype(col_type)
#         X



class ReplaceNaN(BaseEstimator, TransformerMixin):
    """Replace NaNs
    """
    num_col_name = ['OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']
    cat_col_name = ['NAME_HOUSING_TYPE']


    def fit(self, X, y):
        ''' which columns are numerical and which columns are categorical'''

        self.dict = {}
        num_median = X[self.num_col_name].median().values.flatten()
        cat_mod = X[self.cat_col_name].mode().values.flatten()
        for col_name, value in zip(self.cat_col_name, cat_mod):
            self.dict[col_name] = value
        for col_name, value in zip(self.num_col_name, num_median):
            self.dict[col_name] = value

        # print(self.dict)
        return self

    def transform(self, X):
        # print(X.columns)
        X.fillna(value=self.dict, inplace=True)
        # print(X.columns)
        return X

class Interactify(BaseEstimator, TransformerMixin):
    ''' Interactions '''

    interactifier1 = ['FLAG_OWN_CAR_Y']
    interactifier2 = ['OWN_CAR_AGE']

    # def __init__(self, list1, list2):
    #     self.interactifier1 = list1
    #     self.interactifier2 = list2
    #     self.super()

    def fit(self, X, y):
        return self

    def transform(self, X):
        for i, j in zip(self.interactifier1, self.interactifier2):
            X[i+"_"+j] = X[i] * X[j]

        return X


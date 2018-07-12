from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

'''
Select features
'''

family=['NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
      'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow']

education = ['NAME_EDUCATION_TYPE_Higher education',
      'NAME_EDUCATION_TYPE_Incomplete higher',
      'NAME_EDUCATION_TYPE_Lower secondary',
      'NAME_EDUCATION_TYPE_Secondary / secondary special']

housing = ['NAME_HOUSING_TYPE_Municipal apartment',
      'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_House / apartment',
      'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents']

suite = ['NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
       'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B',
       'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied']

contract = ['NAME_CONTRACT_TYPE_Revolving loans']

gender = ['CODE_GENDER_M']

occupation = ['OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
            'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
            'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
            'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
            'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
            'OCCUPATION_TYPE_Medicine staff',
            'OCCUPATION_TYPE_Private service staff',
            'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
            'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
            'OCCUPATION_TYPE_Waiters/barmen staff']

income_type = ['NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Pensioner',
       'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
       'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working']

selected_for_dummies = {
    'CODE_GENDER': gender,
    'NAME_EDUCATION_TYPE':education,
    'NAME_FAMILY_STATUS':family,
    'NAME_TYPE_SUITE':suite,
    'NAME_INCOME_TYPE':income_type,
    'NAME_CONTRACT_TYPE': contract,
    'NAME_HOUSING_TYPE':housing,
    'OCCUPATION_TYPE':occupation,
    'FLAG_OWN_CAR':['FLAG_OWN_CAR_Y'],
    'FLAG_OWN_REALTY':['FLAG_OWN_REALTY_Y']
    }

class ReplaceNaN(BaseEstimator, TransformerMixin):
    """
    Fill in NaNs with mode for cateforicals, and median for continuous variables.
    Input: 2 lists, one for categorical, and another one for continuous variables.
    Output: Dataframe X with NaNs filled in accordingly.
    """
    num_col_name = ['OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                'AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL',
                'REGION_RATING_CLIENT_W_CITY']
    cat_col_name = list(selected_for_dummies.keys())

    def fit(self, X, y):
        '''
        Which columns are numerical and which columns are categorical
        '''
        self.dict = {}
        num_median = X[self.num_col_name].median().values.flatten()
        cat_mod = X[self.cat_col_name].mode().values.flatten()
        for col_name, value in zip(self.cat_col_name, cat_mod):
            self.dict[col_name] = value
        for col_name, value in zip(self.num_col_name, num_median):
            self.dict[col_name] = value

        return self

    def transform(self, X):
        X.fillna(value=self.dict, inplace=True)
        return X

class GetDummies(BaseEstimator, TransformerMixin):
    '''
    Create dummy variables from categorical variables.
    Input: Dictionary mapping original to new variables.
    Output: Dataframe X with new dummy variables.
    '''

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


class BuildVariables(BaseEstimator, TransformerMixin):
    '''
    Creates new variables.
    Input: 2 lists: One of variables to be used as input, another with names of new variables.
    Output: Dataframe X with new variables.
    '''

    columns_to_input = ['DAYS_EMPLOYED']
    columns_new_names = ['employed']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for i, j in zip(self.columns_to_input, self.columns_new_names):
            X[j] = X[i]!=365243*1
        return X

class Logarize(BaseEstimator, TransformerMixin):
    '''
    Create log variables (usually for $ variables).
    Input: List of variables to be logarized.
    Output: Dataframe X with new log variables.
    '''

    columns_to_log = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_log:
            X[col + '_log'] = np.log(X[col])
        return X


class Squarify(BaseEstimator, TransformerMixin):
    '''
    Create squared variables (usually for time variables).
    Input: List of variables to be squared.
    Output: Dataframe X with new squared variables.
    '''

    columns_to_square = ['DAYS_BIRTH_year']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.columns_to_square:
            X[col + '_square'] = X[col]**2
        return X


class Multiplarify(BaseEstimator, TransformerMixin):
    '''
    Create financial multiples in the shape of ratios.
    Input: List of variables to be passed as the numerators of the ratios.
    Output: Dataframe X with new variables.
    '''

    numerators = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']

    def fit(self,X,y):
        return self

    def transform(self,X):
        for col in self.numerators:
            X[col + '_ratio'] = X[col] / X['AMT_INCOME_TOTAL']
        return X

class DatetoYear(BaseEstimator, TransformerMixin):
    '''
    Turn negative days into positive years.
    Input: List of variables measured in negative days.
    Output: Dataframe X with new variables measured in positive years.
    '''

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

class Interactify(BaseEstimator, TransformerMixin):
    '''
    Create interactions among variables.
    Input: 2 lists with variables to interact
    Output: Dataframe X with new interaction variables for each value pair
    '''

    interactifier1 = ['FLAG_OWN_CAR_Y', 'DAYS_EMPLOYED_year']
    interactifier2 = ['OWN_CAR_AGE', 'employed']

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

class SelectColumns(BaseEstimator, TransformerMixin):
    """
    Select columns to pass to model.
    Input: List with the names of the columns.
    Output: Filtered X (dataframe to pass to model)
    """
    keep_cols = list(set(['DAYS_EMPLOYED_year_employed', 'DAYS_BIRTH_year',
                'DAYS_REGISTRATION_year', 'DAYS_BIRTH_year_square',
                'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y', 'FLAG_OWN_CAR_Y_OWN_CAR_AGE',
                'REGION_RATING_CLIENT_W_CITY',
                'AMT_CREDIT_log', 'AMT_GOODS_PRICE_log', 'AMT_ANNUITY_log',  'AMT_INCOME_TOTAL_log',
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

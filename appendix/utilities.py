import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from category_encoders import TargetEncoder

'''
helper function that imputes missing values in numeric features groupwise
'''
def impute_groupwise(X):
    df = X.copy() # create a copy to avoid mutating the original feature set

    # impute by geographic region
    if 'PostalCode' in df.columns:
        # postal code gives the best neighborhood proxy for estimating built years
        df['YearBuilt'] = df.groupby('PostalCode')['YearBuilt'].transform(lambda x: x.fillna(x.median()))
    if 'CountyOrParish' in df.columns:
        df['LivingArea'] = df.groupby('CountyOrParish')['LivingArea'].transform(lambda x: x.fillna(x.median()))
        df['LotSizeSquareFeet'] = df.groupby('CountyOrParish')['LotSizeSquareFeet'].transform(lambda x: x.fillna(x.median()))
        df['AssociationFee'] = df.groupby('CountyOrParish')['AssociationFee'].transform(lambda x: x.fillna(x.median()))
        # fall back on county if postal code is not available for some missing built years
        df['YearBuilt'] = df.groupby('CountyOrParish')['YearBuilt'].transform(lambda x: x.fillna(x.median()))

    # impute by living space
    if 'LivingArea' in df.columns:
        step = 500 # parameter to tune
        min_area = df['LivingArea'].min()
        max_area = df['LivingArea'].max()
        # create 500-sqft tiers and put those records missing living area into a separate tier called "Unknown"
        area_tiers = pd.cut(df['LivingArea'], bins=np.arange(min_area, max_area + step, step)).astype(str).fillna('Unknown')
        # recall that we don't have missing values in bedrooms
        df['BathroomsTotalInteger'] = df.groupby(area_tiers)['BathroomsTotalInteger'].transform(lambda x: x.fillna(x.median()))
        df['Stories'] = df.groupby(area_tiers)['Stories'].transform(lambda x: x.fillna(x.median()))
        df['ParkingTotal'] = df.groupby(area_tiers)['ParkingTotal'].transform(lambda x: x.fillna(x.median()))

    # impute by global medians if any of the imputation steps above fails
    fallback_imputer = SimpleImputer(strategy='median')
    cols_to_check = ['LivingArea', 'LotSizeSquareFeet', 'AssociationFee', 
                     'BathroomsTotalInteger', 'Stories', 'ParkingTotal']
    df[cols_to_check] = fallback_imputer.fit_transform(df[cols_to_check])

    # enforce integer constraints
    df['YearBuilt'] = df['YearBuilt'].round().astype(int)
    df['BedroomsTotal'] = df['BedroomsTotal'].round().astype(int)
    df['BathroomsTotalInteger'] = df['BathroomsTotalInteger'].round().astype(int)
    df['Stories'] = df['Stories'].round().astype(int)
    df['ParkingTotal'] = df['ParkingTotal'].round().astype(int)

    return df

'''
helper function that generates a generic, reusable preprocessing pipeline

parameters:
----------
scale_skewed (bool) determines whether numeric features are scaled: 
    if True, apply monotonic transformations and z-score standardization so features follow a Gaussian-like distribution and have equal weights during modelin;
    otherwise, pass raw features through
'''
def get_preprocessor(high_cardinality_cols, boolean_cols, numerical_cols, scale_skewed=False):
    # scale numeric features (optional)
    if scale_skewed:
        numerical_processor = Pipeline([
            ('skew_correct', PowerTransformer(method='yeo-johnson', standardize=True)) # choose Yeo-Johnson because it works with both negative and positive values
            # ('standardize', StandardScaler())
        ])
    else:
        numerical_processor = "passthrough"

    # one hot encode boolean features
    boolean_processor = Pipeline([
        ('bool_imputer', SimpleImputer(strategy='constant', fill_value=False)),
        ('bool_to_int', FunctionTransformer(lambda x: x.astype(int), validate=False))
    ])

    # target encode high-cardinality categorical features
    location_processor = Pipeline([
        # handle missing values safely before encoding
        ('location_imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        # smoothing regularization prevents overfitting in sparse locations by taking a weighted average of local and global target means to encode location features
        ('target_encoding', TargetEncoder(smoothing=10.0))
    ])

    # create parallel column transformer
    col_transformer = ColumnTransformer(transformers=[
        ('numerical_processing', numerical_processor, numerical_cols),
        ('boolean_processing', boolean_processor, boolean_cols),
        ('location_encoding', location_processor, high_cardinality_cols)
    ], remainder='drop')

    # combine sequential feature processors
    return Pipeline([
        ('groupwise_imputation', FunctionTransformer(impute_groupwise, validate=False)),
        ('column_transformations', col_transformer)
    ])
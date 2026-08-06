import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from category_encoders import TargetEncoder

'''
helper function that implements feature-specific groupwise imputation
'''
def impute_groupwise(X):
    df = X.copy()

    def median_impute(df, group, columns):
        existing_cols = [col for col in columns if col in df.columns]
        for col in existing_cols:
            df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
        return df

    # impute by geographic region
    region_impute_cols = ['LivingArea', 'LotSizeSquareFeet', 'AssociationFee', 'YearBuilt', 'property_age', 'floor_area_ratio']
    if 'CountyOrParish' in df.columns:
        df = median_impute(df, 'CountyOrParish', region_impute_cols)

    # impute by living area
    living_area_impute_cols = ['BathroomsTotalInteger', 'Stories', 'ParkingTotal', 'living_area_per_bedroom', 'bath_bed_ratio']
    if 'LivingArea' in df.columns:
        step = 500  # parameter to tune
        min_area = df['LivingArea'].min()
        max_area = df['LivingArea'].max()

        # create 500-sqft tiers and put records with missing living area into a separate tier called "Unknown"
        if pd.notna(min_area) and pd.notna(max_area):
            area_tiers = pd.cut(df['LivingArea'], bins=np.arange(min_area, max_area + step, step)).astype(str).fillna('Unknown')
            df = median_impute(df, area_tiers, living_area_impute_cols)

    # impute any remaining missing values with global medians
    cols_to_check = [col for col in (region_impute_cols + living_area_impute_cols) if col in df.columns]
    fallback_imputer = SimpleImputer(strategy='median')
    df[cols_to_check] = fallback_imputer.fit_transform(df[cols_to_check])

    # enforce integer constraints
    cols_to_int = ['YearBuilt', 'property_age', 'BedroomsTotal', 'BathroomsTotalInteger', 'Stories', 'ParkingTotal']
    for col in cols_to_int:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    return df

'''
helper function that generates a generic, reusable preprocessing pipeline

scale_skewed determines whether numeric features are scaled: 
    if True, apply monotonic transformations and z-score standardization so features follow a Gaussian-like distribution and have equal weights in modeling;
    otherwise, pass raw features through
'''
def get_preprocessor(high_cardinality_cols, boolean_cols, numerical_cols, scale_skewed=False):
    # scale numeric features (optional)
    if scale_skewed:
        numerical_processor = Pipeline([
            ('skew_correct', PowerTransformer(method='yeo-johnson', standardize=True)) # choose Yeo-Johnson because it works with both negative and positive values
        ])
    else:
        numerical_processor = 'passthrough'

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
        ('groupwise_imputation', FunctionTransformer(impute_groupwise, validate=False, feature_names_out='one-to-one')),
        ('column_transformations', col_transformer)
    ])
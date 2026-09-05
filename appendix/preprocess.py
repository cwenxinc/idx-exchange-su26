import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from category_encoders import TargetEncoder

# ------------------------------------------------------------------
# helper class that implements feature-specific groupwise imputation
# ------------------------------------------------------------------
class GroupwiseImputer(BaseEstimator, TransformerMixin):

    def __init__(self, area_step=500):
        self.area_step = area_step

    # -------------------------------------------------------------------
    # helper function that learns imputation rules from training data
    # NOTE: have the function accept the target for sklearn compatibility
    # -------------------------------------------------------------------
    def fit(self, X, y=None):
        df = X.copy()

        # (i) prepare for region-based imputation
        self.region_medians_ = {}
        self.region_impute_cols_ = [
             'LivingArea', 
             'LotSizeSquareFeet', 
             'AssociationFee', 
             'YearBuilt', 
             'property_age', 
             'floor_area_ratio'
        ]
        for col in self.region_impute_cols_:
            if col in df.columns:
                self.region_medians_[col] = (
                    df.groupby('CountyOrParish')[col].median()
                )

        # (ii) prepare for space-based imputation
        self.living_area_impute_cols_ = [
            'BathroomsTotalInteger',
            'Stories',
            'ParkingTotal',
            'living_area_per_bedroom',
            'bath_bed_ratio'
        ]
        self.area_medians_ = {}
        # create living area tiers and put records with missing living area into a separate tier
        self.area_min_ = df['LivingArea'].min()
        self.area_max_ = df['LivingArea'].max()
        if pd.notna(self.area_min_) and pd.notna(self.area_max_):
            self.area_bins_ = np.arange(
                self.area_min_, 
                self.area_max_ + self.area_step, 
                self.area_step
            )

            area_tiers = pd.cut(
                df['LivingArea'],
                bins=self.area_bins_, 
                include_lowest=True
            ).astype(object)
            area_tiers[df['LivingArea'].isna()] = 'Unknown'

            for col in self.living_area_impute_cols_:
                if col in df.columns:
                    self.area_medians_[col] = (
                        # note that some tiers defined in the training data may not be represented in the validation/test set
                        df.groupby(area_tiers, observed=False)[col].median()
                    )
        else:
            self.area_bins_ = None

        # (iii) prepare global medians for fallback
        cols_to_check = [
            col for col in (
                self.region_impute_cols_ 
                + self.living_area_impute_cols_
            )
            if col in df.columns
        ]
        self.fallback_medians_ = df[cols_to_check].median()

        return self

    # ------------------------------------------------------------------------
    # helper function that imputes missing values using training-learned rules
    # ------------------------------------------------------------------------
    def transform(self, X):
        df = X.copy()

        # (i) apply region-based imputation
        for col, medians in self.region_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(
                    # map each missing value to its county median
                    df['CountyOrParish'].map(medians)
                )

        # (ii) apply space-based imputation
        if getattr(self, 'area_bins_', None) is not None:
            area_tiers = pd.cut(
                df['LivingArea'], 
                bins=self.area_bins_,
                include_lowest=True
            ).astype(object)
            area_tiers[df['LivingArea'].isna()] = 'Unknown'

            for col, medians in self.area_medians_.items():
                if col in df.columns:
                    df[col] = df[col].fillna(
                        area_tiers.map(medians)
                    )

        # (iii) apply fallback if imputation fails due to missing groups or out-of-range values
        for col, median in self.fallback_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # (iv) enforce integer constraints
        cols_to_int = [
            'YearBuilt',
            'property_age',
            'BedroomsTotal',
            'BathroomsTotalInteger',
            'Stories',
            'ParkingTotal'
        ]
        for col in cols_to_int:
            if col in df.columns:
                df[col] = df[col].round().astype(int)

        return df


# ---------------------------------------------------------------------------
# helper function that applies a generic preprocessing pipeline to all models
# ---------------------------------------------------------------------------
def get_preprocessor(high_cardinality_cols, boolean_cols, numerical_cols, scale_skewed=False):
    # (i) scale numeric features (model dependent)
    if scale_skewed:
        numerical_processor = Pipeline([
            # chose Yeo-Johnson because it works with both negative and positive values
            ('skew_correct', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
    else:
        numerical_processor = 'passthrough'

    # (ii) one-hot encode boolean features
    boolean_processor = Pipeline([
        # handle missing values safely before encoding
        ('bool_imputer', SimpleImputer(strategy='constant', fill_value=False)),
        ('bool_to_int', FunctionTransformer(lambda x: x.astype(int), validate=False))
    ])

    # (iii) target encode high-cardinality categorical features
    location_processor = Pipeline([
        # handle missing values safely before encoding
        ('location_imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        # smoothing regularization prevents overfitting in sparse locations by taking a weighted average of local and global target means to encode location features
        ('target_encoding', TargetEncoder(smoothing=10.0))
    ])

    # (iv) create parallel column transformer
    col_transformer = ColumnTransformer(transformers=[
        ('numerical_processing', numerical_processor, numerical_cols),
        ('boolean_processing', boolean_processor, boolean_cols),
        ('location_encoding', location_processor, high_cardinality_cols)
    ], remainder='drop')

    # (v) consolidate into a full preprocessing pipeline
    return Pipeline([
        ('groupwise_imputation', GroupwiseImputer()),
        ('column_transformations', col_transformer)
    ])
import streamlit as st
import pandas as pd
import joblib
import preprocess
from pathlib import Path

# locate and load the trained model
# BASE_DIR = Path(__file__).resolve().parent
model = joblib.load('model.pkl')

# configure the page setup
st.set_page_config(
    page_title='California Single-Family Home Value Estimator', 
    layout='centered'
)
st.title('California Single-Family Home Value Estimator')
st.write('Enter the characteristics of a single-family home to generate an estimated sales price.')

# gather user inputs on layout
st.subheader('Layout')
col1, col2 = st.columns(2)
with col1:
    living_area = st.number_input('Living Area (Sq Ft)', min_value=0, max_value=18000, step=1)
    bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, step=1)
    bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, step=1)
with col2:
    stories = st.number_input('Stories', min_value=1, max_value=3, step=1)
    parking_spaces = st.number_input('Parking Spaces', min_value=0, max_value=10, step=1)
    lot_size = st.number_input('Lot Size (Sq Ft)', min_value=0, max_value=20000, step=100)

# gather user inputs on amenities
st.subheader('Amenities')
has_view = st.checkbox('View')
has_pool = st.checkbox('Private Pool')
has_attached_garage = st.checkbox('Attached Garage')
has_fireplace = st.checkbox('Fireplace')

# gather user inputs on construction history
st.subheader('Construction History')
age = st.number_input('Property Age (Years)', min_value=0, max_value=250, step=1)

# gather user inputs on location 
st.subheader('Location')
st.write('The location needs to be in California.')
city = st.text_input('City') 
county = st.text_input('County')
zipcode = st.text_input('Zipcode') 
school_district = st.text_input('School District')

# put some space between input fields and button
st.write('')

# generate sales price prediction
if st.button(
    'Estimate Home Value',
    type='primary',
    use_container_width=False
):
    input_data = pd.DataFrame({
        'LivingArea': [living_area], 
        'BedroomsTotal': [bedrooms], 
        'BathroomsTotalInteger': [bathrooms], 
        'Stories': [stories], 
        'ParkingTotal': [parking_spaces],
        'LotSizeSquareFeet': [lot_size], 
        'ViewYN': [has_view], 
        'PoolPrivateYN': [has_pool], 
        'AttachedGarageYN': [has_attached_garage], 
        'FireplaceYN': [has_fireplace],
        'property_age': [age],
        'City': [city], 
        'CountyOrParish': [county], 
        'PostalCode': [zipcode], 
        'DistrictNa': [school_district]
    })
    # -------------------------
    # DEBUGGING
    # -------------------------

    # Get preprocessing components
    preprocessor = model.regressor_.named_steps['features']

    groupwise_imputer = (
        preprocessor
        .named_steps['groupwise_imputation']
    )

    column_transformer = (
        preprocessor
        .named_steps['column_transformations']
    )

    location_pipeline = (
        column_transformer
        .named_transformers_['location_encoding']
    )

    location_imputer = (
        location_pipeline
        .named_steps['location_imputer']
    )

    target_encoder = (
        location_pipeline
        .named_steps['target_encoding']
    )

    # 1. Check input
    st.write("Input columns:")
    st.write(input_data.columns.tolist())

    st.write("Input dtypes:")
    st.write(input_data.dtypes)

    # 2. Check fitted encoder configuration
    st.write("TargetEncoder columns:")
    st.write(target_encoder.cols)

    st.write("Location transformer columns:")
    st.write(column_transformer.transformers_[2][2])

    # 3. Extract location input
    location_input = input_data[
        ['CountyOrParish', 'City', 'PostalCode', 'DistrictNa']
    ]

    st.write("Location input:")
    st.write(location_input)

    # 4. Test SimpleImputer
    try:
        imputed_location = location_imputer.transform(location_input)

        st.write("1. Location imputer succeeded.")
        st.write("Imputer output type:")
        st.write(type(imputed_location))
        st.write("Imputer output:")
        st.write(imputed_location)

    except Exception as e:
        st.error("1. Location imputer failed.")
        st.exception(e)

    # 5. Test TargetEncoder after restoring column names
    try:
        imputed_location_df = pd.DataFrame(
            imputed_location,
            columns=location_input.columns,
            index=location_input.index
        )

        st.write("Imputed location DataFrame:")
        st.write(imputed_location_df)

        encoded_location = target_encoder.transform(
            imputed_location_df
        )

        st.write("2. TargetEncoder succeeded after restoring column names.")
        st.write(encoded_location)

    except Exception as e:
        st.error("2. TargetEncoder still failed.")
        st.exception(e)

    # 6. Test GroupwiseImputer + ColumnTransformer
    try:
        imputed_input = groupwise_imputer.transform(input_data)

        st.write("Groupwise imputer output:")
        st.write(imputed_input)

        features_output = column_transformer.transform(imputed_input)

        st.write("3. ColumnTransformer succeeded.")
        st.write("Output type:")
        st.write(type(features_output))
        st.write("Output:")
        st.write(features_output)

    except Exception as e:
        st.error("3. ColumnTransformer failed.")
        st.exception(e)

    # 7. Test complete model
    try:
        prediction = model.predict(input_data)

        st.write("4. Full model prediction succeeded.")
        st.success(f'Estimated Sales Price: ${prediction[0]:,.0f}')

    except Exception as e:
        st.error("4. Full model prediction failed.")
        st.exception(e)

    # -------------------------
    # END DEBUGGING
    # -------------------------
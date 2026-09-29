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
    page_title='Single-Family Home Value Estimator', 
    layout='centered'
)
st.title('Single-Family Home Value Estimator')
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
    try:
        # -------------------------
        # DEBUGGING
        # -------------------------

        # 1. Inspect input
        st.write("Input columns:")
        st.write(input_data.columns.tolist())

        st.write("Input dtypes:")
        st.write(input_data.dtypes)

        # 2. Access fitted preprocessing components
        preprocessor = model.regressor_.named_steps['features']

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

        # 3. Confirm what the fitted TargetEncoder expects
        st.write("TargetEncoder columns:")
        st.write(target_encoder.cols)

        st.write("Location transformer columns:")
        st.write(column_transformer.transformers_[2][2])

        # 4. Extract only the location columns
        location_input = input_data[
            ['CountyOrParish', 'City', 'PostalCode', 'DistrictNa']
        ]

        st.write("Location input:")
        st.write(location_input)

        # 5. Test SimpleImputer
        imputed_location = location_imputer.transform(location_input)

        st.write("Imputer output type:")
        st.write(type(imputed_location))

        st.write("Imputer output:")
        st.write(imputed_location)

        # 6. Test TargetEncoder directly
        encoded_location = target_encoder.transform(imputed_location)

        st.write("TargetEncoder succeeded.")
        st.write(encoded_location)

        # 7. Finally test the full model
        prediction = model.predict(input_data)
        st.success(f'Estimated Sales Price: ${prediction[0]:,.0f}')

        # -------------------------
        # END DEBUGGING
        # -------------------------

    except Exception as e:
        st.error('Sorry, the estimation could not be generated.')
        st.exception(e)
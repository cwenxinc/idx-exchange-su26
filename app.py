import streamlit as st
import numpy as np
import pandas as pd
import joblib

# load the trained model
model = joblib.load('model.pkl')

# configure the page setup
st.set_page_config(
    page_title='Single-Family Home Value Estimator', 
    layout='centered'
)
st.title('Single-Family Home Value Estimator')
st.write('Enter the characteristics of a single-family home to generate an estimated sales price.')

# gather user inputs on location 
st.subheader('Location')
loc1, loc2 = st.columns(2)
with loc1: 
    county = st.text_input('County') 
    city = st.text_input('City') 
    zipcode = st.text_input('Zipcode') 
with loc2: 
    school_district = st.text_input('School District')
    mls_area = st.text_input('MLS Area Major')

# gather user inputs on layout
st.subheader('Layout')
layout1, layout2 = st.columns(2)
with layout1:
    living_area = st.number_input('Living Area (Sq Ft)', min_value=0, max_value=18000, step=100)
    bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, step=1)
    bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, step=1)
    stories = st.number_input('Stories', min_value=1, max_value=3, step=1)
with layout2:
    parking_spaces = st.number_input('Parking Spaces', min_value=0, max_value=10, step=1)
    lot_size = st.number_input('Lot Size (Sq Ft)', min_value=0, max_value=20000, step=100)

# gather user inputs on construction history
st.subheader('Construction History')
hist = st.columns(1)
with hist:
    age = st.number_input('Property Age (Years)', min_value=0, max_value=250, step=1)

# gather user inputs on amenities
st.subheader('Amenities')
amen = st.columns(1)
with amen:
    has_view = st.checkbox('View')
    has_pool = st.checkbox('Private Pool')
    has_attached_garage = st.checkbox('Attached Garage')
    has_fireplace = st.checkbox('Fireplace')

# generate sales price prediction
if st.button(
    'Estimate Home Value',
    type='primary',
    use_container_width=False
):
    input_data = pd.DataFrame({
        'MLSAreaMajor': [mls_area], 
        'CountyOrParish': [county], 
        'City': [city], 
        'PostalCode': [zipcode], 
        'DistrictNa': [school_district], 
        'ViewYN': [has_view], 
        'PoolPrivateYN': [has_pool], 
        'AttachedGarageYN': [has_attached_garage], 
        'FireplaceYN': [has_fireplace], 
        'LivingArea': [living_area], 
        'BedroomsTotal': [bedrooms], 
        'BathroomsTotalInteger': [bathrooms], 
        'Stories': [stories], 
        'LotSizeSquareFeet': [lot_size], 
        'ParkingTotal': [parking_spaces], 
        'property_age': [age] 
    })
    try:
        prediction = model.predict(input_data)
        st.success(f'Estimated Sales Price: ${prediction[0]:,.0f}')
    except Exception as e:
        st.error('Sorry, the prediction could not be generated.')
        st.exception(e)
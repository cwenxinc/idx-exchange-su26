import streamlit as st
import pandas as pd
import joblib
import preprocess

# load the trained model
model = joblib.load('model.pkl')

# -------------------------
# tab configuration
# -------------------------
st.set_page_config(
    page_title='California Home Valuation',
    layout='wide'
)

# -------------------------
# page content & structure
# -------------------------
st.image('logo.png', width=120)
st.title('What\'s My Home Worth in California?')
st.write('Get an instant home valuation based on recent California sales, then ' 
         'connect with an IDX Exchange expert to maximize your sales price.')

# -------------------------
# input controls
# -------------------------
st.write('To start off, help us gather some information about your home! \n')
with st.expander('How is this tool built and used?'):
    st.write('This tool is trained on CRMLS sales records from January 2025 through April 2026 and '
             'uses gradient boosting, a tree-based ensemble machine learning method, to generate sales price estimates. '
             'The information you provide below is used solely to prepare your estimate and will not be retained or sold.')

row1_col1, row1_col2 = st.columns(2, gap='large')
with row1_col1:
    with st.container(border=True):
        st.subheader('Location')
        city = st.text_input('City', placeholder='e.g., El Cajon') 
        county = st.text_input('County', placeholder='e.g., San Diego')
        zipcode = st.text_input('ZIP Code', placeholder='e.g., 92021') 
        school_district = st.text_input('School District', placeholder='e.g., Grossmont Union High')
with row1_col2:
    with st.container(border=True):
        st.subheader('Layout')
        living_area = st.number_input('Living Area (Sq Ft)', min_value=0, max_value=18000, step=1)
        bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, step=1)
        bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, step=1)
        stories = st.number_input('Stories', min_value=1, max_value=3, step=1, value=1)
        parking_spaces = st.number_input('Parking Spaces', min_value=0, max_value=10, step=1)
        lot_size = st.number_input('Lot Size (Sq Ft)', min_value=0, max_value=20000, step=100)

row2_col1, row2_col2 = st.columns(2, gap='large')
with row2_col1:
    with st.container(border=True):
        st.subheader('Amenities')
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            has_view = st.checkbox('View')
            has_fireplace = st.checkbox('Fireplace')
        with sub_col2:
            has_pool = st.checkbox('Private Pool')
            has_attached_garage = st.checkbox('Attached Garage')
with row2_col2:
    with st.container(border=True):
        st.subheader('Other Details')
        age = st.number_input('Property Age (Years)', min_value=0, max_value=250, step=1)
            
# put some space between input fields and button
st.write('\n')

# -------------------------
# button & status messages
# -------------------------
# generate sales price prediction
if st.button(
    'Show Me My Home Value',
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
        prediction = model.predict(input_data)
        st.success(f'Estimated Sales Price: ${prediction[0]:,.0f}')

    except Exception as e:
        st.error('Sorry, we weren\'t able to generate an estimate.')
        st.exception(e)
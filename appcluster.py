
import streamlit as st
import numpy as np
import joblib

st.markdown("""
            <style>
            .stTextInput, .stNumberInput, .stRadio, .stCheckbox ,.stSelectbox{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: #000000;  /* Black background for inputs */
            border: 2px solid #ffffff;  /* White border for inputs */
            color: #ffffff;  /* White font for the inputs */
        }
        [data-testid="stAppViewContainer"] {
        background-image: url('https://img.freepik.com/free-photo/type-entertainment-complex-popular-resort-with-pools-water-parks-turkey-with-more-than-5-million-visitors-year-amara-dolce-vita-luxury-hotel-resort-tekirova-kemer_146671-18728.jpg?t=st=1735057833~exp=1735061433~hmac=d30725398f1d3bddf43cc3005ddb16be92a43d6eebc7a5f18c8b43127725fcee&w=1060');
        background-size: cover;
        background-attachment: fixed;
    }
        .custom-success-box {
            padding: 15px;
            margin: 20px;
            border: 2px solid #dc3545;  /* Red border */
            background-color: #dc3545;  /* Light red background */
            color: #fffff;  /* Dark red text */
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }
            .title{
            font-size: 36px;
            font-weight: bold;
            color: #fffff;  /* White text color */
            background-color: #253f4b;  /* Orange background */
            padding: 10px 20px;  /* Add padding around the text */
            border-radius: 10px;  /* Rounded corners for the background */
            text-align: center;  /* Center align the title */
            font-family: 'Arial', sans-serif;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); 

            }
            
    
</style>
            """,unsafe_allow_html=True)



# Load the trained KMeans model and scaler
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define input fields for customer details
st.markdown('<div class="title">Customer Cluster Prediction</div>', unsafe_allow_html=True)


# Input fields
avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, value=100.0, step=0.1)
lead_time = st.number_input('Lead Time (days)', min_value=0, value=50)
no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=1)
no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, value=0)
repeated_guest = st.selectbox('Is Repeated Guest?', [0, 1])
market_segment_type = st.selectbox('Market Segment Type', [0, 1, 2, 3])  # Assuming encoded market segments

# Collect the input data into a numpy array
customer_data = np.array([[avg_price_per_room, lead_time, no_of_special_requests,
                           no_of_previous_bookings_not_canceled, repeated_guest, market_segment_type]])

# Button to predict the cluster
if st.button('Predict Cluster'):
    # Scale the input data
    customer_data_scaled = scaler.transform(customer_data)

    # Predict the cluster
    cluster = kmeans_model.predict(customer_data_scaled)[0]

    # Assign cluster labels
    if cluster == 0:
        cluster_label = 'Budget'
    elif cluster == 1:
        cluster_label = 'Luxury'
    else:
        cluster_label = 'Frequent'

    # Display the predicted cluster
       
    st.markdown(f"""
        <div class="custom-success-box">
            The predicted customer cluster is: <strong>{cluster_label}</strong>
        </div>
    """, unsafe_allow_html=True)


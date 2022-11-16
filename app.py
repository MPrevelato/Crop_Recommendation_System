import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown('Crop Recommendation System')

st.markdown("<h1 style='text-align: center; color: black;'>Crop Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: black;'>Here you can insert the features! This way the system will predict the best crop to plant!</h5>", unsafe_allow_html=True)


col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

with col3:
    n_input = st.number_input('Insert N (kg/ha) value:', min_value= 0, max_value= 140)
    p_input = st.number_input('Insert P (kg/ha) value:', min_value= 5, max_value= 145)
    k_input = st.number_input('Insert K (kg/ha) value:', min_value= 5, max_value= 205)
    temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value= 9., max_value= 43., step = 1., format="%.2f")

with col5:
    hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value= 15., max_value= 99., step = 1., format="%.2f")
    ph_input = st.number_input('Insert pH value:', min_value= 3.6, max_value= 9.9, step = 0.1, format="%.2f")
    rain_input = st.number_input('Insert Avg Rainfall (mm) value:', min_value= 21.0, max_value= 298.0, step = 0.1, format="%.2f")


df = pd.read_csv('Crop_recommendation.csv')

X = df.drop('label', axis = 1)
y = df['label']


gnb = GaussianNB()
gnb.fit(X,y)

predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input]]




with col5:
    predict_btn = st.button('Get Your Prediction')

if predict_btn:
    gnb_predicted_value = gnb.predict(predict_inputs)
    st.text('Crop suggestion: {}'.format(gnb_predicted_value[0]))
        
    df_pred = df[df['label'] == gnb_predicted_value[0]]
    st.dataframe(df_pred.describe(), use_container_width=True)
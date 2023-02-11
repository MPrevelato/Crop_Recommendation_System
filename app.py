import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import joblib

pd.options.display.max_colwidth = 2000
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-color:#ffc107;

}}
[data-testid="stSidebar"] {{
background-color:#8C564B;

}}
[data-testid="stHeader"] {{
background-color:#ffc107;
}}
[data-testid="stToolbar"] {{
background-color:#ffc107;

}}
</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

def load_bootstrap():
        return st.markdown("""<link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">""", unsafe_allow_html=True)

with st.sidebar:
    
    load_bootstrap()
    st.markdown("""<h4 style='text-align: center; color: black;'>
    This Application was developed by Mateus Prevelato Athayde. 
    You can find my Linkedin and GitHub profiles below:</h4>""",unsafe_allow_html=True)
    st.markdown(f"""<h4 style='text-align: center; color: black;'>
     <a style='text-align: center; color: black;' type="button" class="btn btn-warning btn-lg" 
     href = "https://github.com/MPrevelato/Crop_Recommendation_System">GitHub</a> <a style='text-align: center; color: black;' 
     type="button" class="btn btn-warning btn-lg" 
     href = "https://www.linkedin.com/in/mateus-prevelato/">Linkedin</a></h4>""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: black;'>Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>This Application predict what is the best crop to plant based on NPK values and Weather Conditions!</h5>", unsafe_allow_html= True)

colx, coly, colz = st.columns([1,4,1], gap = 'medium')
with coly:
    st.markdown("""
  
    
      <h6 style='text-align: center;'>
        This Application is a Random Forest model that give 
        recommendations to farmers based on an Indian Crop 
        Recommendation <a style='text-align: center; color: blue;' 
        href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset">Dataset</a>.
        This way, by inserting N, P, K and pH based on your soil conditions
        or the desired values of these 4 values (changed by fertilizers), and by giving
        the weather conditions, such as temperature, humidity, rainfall, it is possible
        to generate a recommendation of which crop to plant! Below you can see a bar chart
        that, in a simple way, explain of how much each of these features
        impacts the model.
      </h6>
  
        """, unsafe_allow_html=True)

df = pd.read_csv('Crop_recommendation.csv')

rdf_clf = joblib.load('final_rdf_clf.pkl')

X = df.drop('label', axis = 1)
y = df['label']

df_desc = pd.read_csv('Crop_Desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')

st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model:</h5>", unsafe_allow_html=True)


importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
importance['Importance(%)'] = importance['Importance(%)'] * 100

colx, coly, colz = st.columns([1,4,1], gap = 'medium')
with coly:
    color_discrete_sequence = '#609cd4'
    fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
    fig.update_traces(marker_color="#8C564B")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width= True)

st.markdown("<h5 style='text-align: center;'>Here you can insert the features! This way the system will predict the best crop to plant!</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>In the (?) marks you can get some help about each feature.</h5>", unsafe_allow_html=True)


col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

with col3:
    n_input = st.number_input('Insert N (kg/ha) value:', min_value= 0, max_value= 140, help = 'Insert here the Nitrogen density (kg/ha) from 0 to 140.')
    p_input = st.number_input('Insert P (kg/ha) value:', min_value= 5, max_value= 145, help = 'Insert here the Phosphorus density (kg/ha) from 5 to 145.')
    k_input = st.number_input('Insert K (kg/ha) value:', min_value= 5, max_value= 205, help = 'Insert here the Potassium density (kg/ha) from 5 to 205.')
    temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value= 9., max_value= 43., step = 1., format="%.2f", help = 'Insert here the Avg Temperature (ºC) from 9 to 43.')

with col5:
    hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value= 15., max_value= 99., step = 1., format="%.2f", help = 'Insert here the Avg Humidity (%) from 15 to 99.')
    ph_input = st.number_input('Insert pH value:', min_value= 3.6, max_value= 9.9, step = 0.1, format="%.2f", help = 'Insert here the pH from 3.6 to 9.9')
    rain_input = st.number_input('Insert Avg Rainfall (mm) value:', min_value= 21.0, max_value= 298.0, step = 0.1, format="%.2f", help = 'Insert here the Avg Rainfall (mm) from 21 to 298')




predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input]]

with col5:
    predict_btn = st.button('Get Your Recommendation!')


cola,colb,colc = st.columns([2,10,2])
if predict_btn:
    rdf_predicted_value = rdf_clf.predict(predict_inputs)
    #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
    with colb:
        st.markdown(f"<h3 style='text-align: center;'>Best Crop to Plant: {rdf_predicted_value[0]}.</h3>", 
        unsafe_allow_html=True)
    col1, col2, col3 = st.columns([9,4,9])
    with col2:
        df_desc = df_desc.astype({'label':str,'image':str})
        df_desc['label'] = df_desc['label'].str.strip()
        df_desc['image'] = df_desc['image'].str.strip()
        

        df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
        df_image = df_pred_image['image'].item()
        
        st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)
        

    
    st.markdown(f"""<h5 style='text-align: center;'>Statistics Summary about {rdf_predicted_value[0]} 
            NPK and Weather Conditions values in the Dataset.</h5>""", unsafe_allow_html=True)
    df_pred = df[df['label'] == rdf_predicted_value[0]]
    st.dataframe(df_pred.describe(), use_container_width = True)        
    

    
    

    
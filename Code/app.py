# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0


import numpy as np
import pandas as pd
import streamlit as st # For building the web app
from train_Kmeans import train_model as train
from train_gmm import train_model as train_gmm
from train_db import train_model as train_db
from train_optics import train_model as train_optics
from train_birch import train_model as train_birch
from classification import classify
from ingest_transform import preprocess_test

st.set_page_config(page_title="Unsupervised Clustering", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Unsupervised Clustering </h1>", unsafe_allow_html=True)
st.divider()


tab1, tab2, tab3=  st.tabs(["Model Config", "Model Training & Evaluation", "Classification"])

with tab1:  
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_csv("MOCK_DATA.csv")

    st.write("Here is a preview of your data:")
    st.write(df.head())

with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()

    # K-Means

    model_name = 'K-Means'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    num_clusters = st.number_input('Number of clusters:', min_value=2, max_value=10, value=3, step=1)
    if st.button(f"Train {model_name} Model", use_container_width=True):
        with st.status("Training K-Means Model..."):
            score = train(df, num_clusters)
    
            st.write(f"Training complete! The score is: {score}")

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    st.divider()
        
    # Gaussian Mixture Model
    
    model_name = 'Gaussian Mixture Model'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    n_component = st.slider('n_component:', min_value=0, max_value=10, step=1)
    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_gmm(df,n_component)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    st.divider()
        
        
    # DBSCAN
    
    model_name = 'DBSCAN'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    eps = st.slider('EPS (distance threshold):', min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    min_sm = st.slider('Min Samples', min_value=0, max_value=50, step=2)
    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_db(df, eps, min_sm, minmax = True)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    st.divider()

        
    # OPTICS
    
    model_name = 'OPTICS'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    min_sample = st.slider('min_samples:', min_value=0, max_value=20, step=1)
    xi = st.slider('xi:', min_value=0.0, max_value=1.0, step=0.01)
    cluster = st.slider('min_cluster_size:', min_value=0.0, max_value=1.0, step=0.05)
    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_optics(df, min_sample, xi, cluster,minmax = True)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    st.divider()

        
    # BIRCH
    model_name = 'BIRCH'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

    n_clus1 = st.number_input('Number of clusters:', min_value=2, max_value=10, value=3, step=1, key='clusters_option_1')
    threas = st.slider('min_cluster_size:', min_value=0.0, max_value=1.0, step=0.05, key='thres_option_2')
    if st.button(f"Train {model_name} Model", use_container_width=True):
        
        with st.status(f"Training {model_name}..."):
            score= train_birch(df, n_clus1, threas, minmax = True)

        st.success(f"{model_name} Trained Sucessully")
        st.markdown("<h4 style='text-align: center; color: white;'>Model Evaluation </h4>", unsafe_allow_html=True)

        st.write(f"Silhouette: {score}")
        st.image(f"Code/saved images/{model_name}.jpg", caption=f"Cluster graph of {model_name}", width=600)

    st.divider()


with tab3:
    algorithm = st.selectbox("Select algorithm:", ("KMeans", "DBSCAN", "Gaussian Mixture Model", "OPTICS", "BIRCH"))
    
    
    with st.form(key="clustering_form"):
        st.subheader("Clustering")
        st.write("Enter customer details for clustering.")

        age = st.number_input('Enter age (18-80):', min_value=18, max_value=80, value=30)

        income = st.number_input('Enter annual income (30000.00-100000.00):', min_value=3000.00, max_value=300000.00, value=50000.00, step=0.01, format="%.2f")

        purchase_history = st.number_input('Enter Purchase history :', min_value=100, max_value=50000, value=500)

        customer_spending_score = st.number_input('Enter Customer Spending Score :', min_value=0, max_value=100, value=50)

        freq_of_visit = st.number_input('Enter frequency of visit:', min_value=0, max_value=100, value=50)

        gender_opt = ['Male', 'Female', 'Agender', 'Genderqueer', 'Polygender', 'Genderfluid', 'Non-binary', 'Bigender']

        gender = preprocess_test(st.radio('Choose an option:', gender_opt,horizontal=True))

        region = preprocess_test(st.radio('Choose an option:', ['East', 'West', 'North', 'South'],horizontal=True))

        customer_type = preprocess_test(st.radio('Choose an option:', ['budget', 'regular', 'premium'], horizontal=True))

        items = np.array([[age, income, purchase_history, customer_spending_score, freq_of_visit, gender, region, customer_type]])

        if st.form_submit_button("Cluster", use_container_width=True):
            new_cluster = classify(algorithm, items)
            st.write(f"The Data belong to {new_cluster}")

    

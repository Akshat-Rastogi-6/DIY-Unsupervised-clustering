# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (04 July 2024)
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


import joblib
import pandas as pd
from sklearn.cluster import KMeans
from ingest_transform import preprocess_test, scale_back


def classify(algorithm, items):
    scaled_data = scale_back(items)
    if(algorithm == 'KMeans'):
        model = joblib.load('Code\saved model\kmeans.pkl')
        clusters = model.predict(scaled_data)
        return clusters
    
    elif(algorithm == 'Gaussian Mixture Model'):
        model = joblib.load('Code\saved model\gmm.pkl')
        clusters = model.predict(scaled_data)
        return clusters
    
    elif(algorithm == 'DBSCAN'):
        model = joblib.load('Code\saved model\dbscan.pkl')
        clusters = model.predict(scaled_data)
        return clusters
    
    elif(algorithm == 'OPTICS'):
        model = joblib.load('Code\saved model\optics.pkl')
        clusters = model.predict(scaled_data)
        return clusters
    
    elif(algorithm == 'BIRCH'):
        model = joblib.load(r'Code\saved model\birch.pkl')
        clusters = model.predict(scaled_data)
        return clusters
    
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


import pandas as pd # For data manipulation and analysis
import joblib # For loading the trained model
from sklearn.cluster import DBSCAN, OPTICS
from ingest_transform import preprocess_data, scale_data

# Importing helper functions from the local .py files
# from load import load_train
from evaluate import evaluate_model

def train_model(df, min_sample, xi, cluster, minmax):
    X = preprocess_data(df)
    X_pca = scale_data(X, minmax)
    optics = OPTICS(min_samples=min_sample, xi=xi, min_cluster_size=cluster).fit(X_pca)
    # optics = OPTICS(min_samples=10, xi=0.02, min_cluster_size=0.25).fit(X_pca)
    labels = optics.labels_
    evals = evaluate_model(X_pca, labels, 'OPTICS')
    joblib.dump(optics, 'Code\saved model\optics.pkl')

    return evals
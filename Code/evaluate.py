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

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def evaluate_model(X_pca, labels, algorithm_name, centroids = None):
    silhouette_avg = silhouette_score(X_pca, labels)

    plt.figure(figsize=(8, 6))

    # Scatter plot of the data points colored by cluster label
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=100, label="Data points")

    # Plot the centroids on the same plot
    if(centroids is not None):
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x', label="Centroids")

    # Adding title and labels
    plt.title(f'{algorithm_name} Clustering on Dataset')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.legend()
    plt.grid(True)
    plt.colorbar(label='Cluster Label')
    plt.savefig(f'Code/saved images/{algorithm_name}.jpg', format="jpg", dpi=300)
    plt.show()
    return silhouette_avg

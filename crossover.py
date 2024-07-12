import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Define the directory path where your data is located
data_dir = '/Users/abhijeetkumar/CrossOver/my_dataset.csv'

# Load the data
data = pd.read_csv(data_dir)

# Extract relevant data
user_data = data[['Customer ID', 'StockCode', 'Quantity', 'InvoiceDate']]

# Convert 'Customer ID' to numeric index
user_data = user_data.reset_index(drop=True)
user_data['user_id'] = user_data.index

# Handle missing values and data cleaning
user_data = user_data.dropna()

# Create user-item interaction matrix
user_item_matrix = user_data.pivot_table(index='user_id', columns='StockCode', values='Quantity', fill_value=0)

# Perform SVD to learn user and item embeddings
svd = TruncatedSVD(n_components=50)
user_embeddings = svd.fit_transform(user_item_matrix)
item_embeddings = svd.components_.T

def collaborative_filtering_recommendation(user_id, top_n=10):
    """
    Provide product recommendations using collaborative filtering.
    """
    user_vector = user_embeddings[int(user_id)]
    item_scores = user_vector @ item_embeddings.T
    top_items = np.argsort(item_scores)[-top_n:]
    return [int(item) for item in top_items]

def content_based_recommendation(user_id, top_n=10):
    """
    Provide product recommendations using content-based filtering.
    """
    user_vector = user_embeddings[int(user_id)]
    item_similarities = 1 - user_item_matrix.T.apply(lambda x: cosine(x, user_vector))
    top_items = item_similarities.nlargest(top_n).index
    return [int(item) for item in top_items]

def hybrid_recommendation(user_id, top_n=10):
    """
    Provide product recommendations using a hybrid approach.
    """
    cf_recommendations = collaborative_filtering_recommendation(int(user_id), top_n)
    cb_recommendations = content_based_recommendation(int(user_id), top_n)
    return list(set(cf_recommendations + cb_recommendations))[:top_n]

def offline_evaluation(test_data, top_n=10):
    """
    Evaluate the recommendation engine using offline metrics.
    """
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []

    for user_id in test_data['user_id'].unique():
        user_interactions = test_data[test_data['user_id'] == user_id]['StockCode'].tolist()
        recommendations = hybrid_recommendation(user_id, top_n)

        precision = len(set(recommendations) & set(user_interactions)) / top_n
        recall = len(set(recommendations) & set(user_interactions)) / len(user_interactions)
        ndcg = ndcg_score(user_interactions, recommendations, top_n)

        precision_at_k.append(precision)
        recall_at_k.append(recall)
        ndcg_at_k.append(ndcg)

    return {
        'Precision@k': np.mean(precision_at_k),
        'Recall@k': np.mean(recall_at_k),
        'NDCG@k': np.mean(ndcg_at_k)
    }

def ndcg_score(true_items, recommended_items, k):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) score.
    """
    dcg = 0
    idcg = 0

    for i, item in enumerate(recommended_items[:k]):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)

    for i, item in enumerate(true_items[:k]):
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg else 0

# Example usage
test_data = user_data.sample(frac=0.2, random_state=42)
evaluation_metrics = offline_evaluation(test_data, top_n=10)
print(evaluation_metrics)
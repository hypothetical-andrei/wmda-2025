import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# 2. Custom silhouette scorer for DBSCAN
#    NOTE: The signature must be (estimator, X, y=None) to avoid "missing y_true" errors.
def dbscan_silhouette_scorer(estimator, X, y=None):
    # After GridSearchCV calls estimator.fit(X_train), the cluster labels are in estimator.labels_
    labels = estimator.labels_  # DBSCAN stores the cluster assignments here

    # Exclude outliers (label == -1)
    mask = labels != -1
    # If there are fewer than 2 clusters among the non-outliers, silhouette is undefined
    if len(set(labels[mask])) < 2:
        return -1

    return silhouette_score(X[mask], labels[mask])

# 3. Define the parameter grid
param_grid = {
    'eps': [0.2, 0.3, 0.4, 0.5],
    'min_samples': [3, 5, 7]
}

# 4. Use a "single-split" CV that uses the entire dataset as both train and test
#    Unsupervised scenario often does not have a separate test set
indices = np.arange(len(X))
single_split_cv = [(indices, indices)]

# 5. Set up the GridSearchCV
grid_search = GridSearchCV(
    estimator=DBSCAN(),
    param_grid=param_grid,
    scoring=dbscan_silhouette_scorer,  # pass the callable directly
    cv=single_split_cv,
    n_jobs=-1
)

# 6. Fit and find the best hyperparameters
grid_search.fit(X)

# 7. Report results
print("Best Params:", grid_search.best_params_)
print("Best Silhouette Score:", grid_search.best_score_)

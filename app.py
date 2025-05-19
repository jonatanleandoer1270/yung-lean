
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

data = pd.read_csv('/content/drive/MyDrive/songs_normalize.csv')
df = pd.DataFrame(data)
print(df.head(10))

df['primary_genre'] = df['genre'].str.split(', ').str[0]
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['primary_genre'])

diddler = df[['energy', 'loudness', 'liveness', 'valence', 'genre_encoded', 'tempo']]
scaler = StandardScaler()
Ye = scaler.fit_transform(diddler)

y = df['song']
pca = PCA(n_components=3)
Yungleanxdiddy = pca.fit_transform(Ye)

X_train, X_test, y_train, y_test = train_test_split(Yungleanxdiddy, y, test_size=0.2, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
print("Test Set Score:", grid_search.score(X_test, y_test))

y_pred = grid_search.predict(X_test)

def recommend_songs_knn(song_name, top_n=5):
    if song_name not in df['song'].values:
        return f"{song_name} not found in dataset."
    song_index = df[df['song'] == song_name].index[0]
    song_vector = Yungleanxdiddy[song_index].reshape(1, -1)
    best_model = grid_search.best_estimator_
    distances, indices = best_model.named_steps['knn'].kneighbors(song_vector, n_neighbors=top_n + 1)
    recommended_indices = indices[0][1:]
    return df.iloc[recommended_indices]['song'].tolist()

print("Top KNN Recommendations:")
print(recommend_songs_knn("Comfortably Numb", top_n=5))

plt.plot([0, 1, 2, 3, 4, 5], pca.components_[0])
plt.title("Relative contribution of each metric in the musics")
plt.xlabel("Metrics")
plt.ylabel("Relative contribution")
plt.show()

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
therevengeofpuff = kmeans.fit_predict(Yungleanxdiddy)

plt.figure(figsize=(12, 6))
plt.scatter(Yungleanxdiddy[:, 0], Yungleanxdiddy[:, 1], c=therevengeofpuff, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('Clusters using K-means')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()

plt.scatter(Yungleanxdiddy[:, 1], Yungleanxdiddy[:, 2], c=therevengeofpuff, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='red', label='Centroids')
plt.title('Clusters using K-means')
plt.xlabel('PCA Feature 2')
plt.ylabel('PCA Feature 3')
plt.legend()
plt.show()

plt.scatter(Yungleanxdiddy[:, 0], Yungleanxdiddy[:, 2], c=therevengeofpuff, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='red', label='Centroids')
plt.title('Clusters using K-means')
plt.xlabel('PCA Feature 2')
plt.ylabel('PCA Feature 3')
plt.legend()
plt.show()

silhouette_avg = silhouette_score(Yungleanxdiddy, therevengeofpuff)
print(f"Silhouette Score: {silhouette_avg}")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

metrics = {
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'test_score': grid_search.score(X_test, y_test),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}
joblib.dump(metrics, 'metrics.pkl')

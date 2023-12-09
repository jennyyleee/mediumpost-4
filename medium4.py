import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("Hot 100 Audio Features.csv")
print(df)

numerical_columns = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'spotify_track_popularity']
music_data = df[numerical_columns]

music_data.dropna(inplace=True)



# Elbow Method to find optimal k-value
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(music_data)
    wcss.append(kmeans.inertia_)

#plot for elbow method
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Num of Clusters')
plt.ylabel('WCSS')
plt.show()

# using KMeans with k=4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusterlabels = kmeans.fit_predict(music_data)

music_data['cluster'] = clusterlabels
cluster_summary = music_data.groupby('cluster').mean()
print(cluster_summary)



#finding count for each cluster
cluster0_count = sum(clusterlabels == 0)
cluster1_count = sum(clusterlabels == 1)
cluster2_count = sum(clusterlabels == 2)
cluster3_count = sum(clusterlabels == 3)


print(f"Num songs in Cluster 0: {cluster0_count}")
print(f"Num songs in Cluster 1: {cluster1_count}")
print(f"Num songs in Cluster 2: {cluster2_count}")
print(f"Num songs in Cluster 3: {cluster3_count}")


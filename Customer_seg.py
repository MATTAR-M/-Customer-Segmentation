#Importing necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
#loading the dataset from my local machine 
df = pd.read_csv(r"C:\Users\DELL\Downloads\Mall_Customers.csv")
df.head()
#selecting the desired two features for the clustring 
x=df[['Annual Income (k$)','Spending Score (1-100)']]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
#using elbow method to get numbers of clusters 
inertia = []
k= range(1,11)
for i in k:
    imeans = KMeans(n_clusters=i,random_state=42)
    imeans.fit(x_scaled)
    inertia.append(imeans.inertia_)
plt.plot(k,inertia,marker='o')
plt.xlabel('cluster number')
plt.ylabel('inertia')
plt.title('Elbow method')
plt.grid(True)
plt.show()
#Apply KMeans clustering with the number of clusters
imeans = KMeans(n_clusters=5,random_state=42)
df['cluster']=imeans.fit_predict(x_scaled)
#map cluster numbers to meaningful customer segment names
cluster_names = {
    0: 'Standard Spenders',
    1: 'Budget Cautious',
    2: 'Luxury Seekers',
    3: 'Potential Big Spenders',
    4: 'High Value Customers'
}
df['segment'] = df['cluster'].map(cluster_names)
#Visualize the KMeans clustering using a scatter plot
plt.figure(figsize=(9,6))
sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue='segment',palette='Set2')
plt.title('Customer segment by k-means')
plt.legend()
plt.grid(True)
plt.show()
#print average income and spending per cluster to understand the segments
cluster_summary = df.groupby('cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)
#Bonus: Apply DBSCAN to detect clusters and outliers
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(x_scaled)
DBSCAN_cluster_names = {
    -1: 'Noise',
     0: 'Cluster A',
     1: 'Cluster B',
     2: 'Cluster C',
     3: 'Cluster D'
}
df['DBSCAN_Segment'] = df['DBSCAN_Cluster'].map(DBSCAN_cluster_names)
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Segment', palette='tab10')
plt.title('Customer Segments by DBSCAN')
plt.show()
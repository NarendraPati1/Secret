import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv('sales_data_sample.csv', encoding='latin1')

# 2. Select columns and remove missing values
X = df[['SALES', 'QUANTITYORDERED', 'PRICEEACH']].dropna()

# 3. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 5. Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# 6. Apply K-Means (assuming optimal K = 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)

# 7. Display results
print(df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'Cluster']].head())

# 8. Visualize clusters
plt.scatter(X['QUANTITYORDERED'], X['SALES'],
            c=df.loc[X.index, 'Cluster'])

plt.xlabel('Quantity Ordered')
plt.ylabel('Sales')
plt.title('K-Means Clustering')
plt.show()

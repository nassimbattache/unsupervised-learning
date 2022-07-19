from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering,SpectralClustering,DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.cluster import hierarchy
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import time

#Dictionnaire des score de toutes les méthodes
silhouette_dict = {}

#Calcul des temps d'exécutions

temps = {}

#Chargement de la dataset 
df = pd.read_csv('Wholesale_customers_data.csv')

#Récupération des données
data = df.values

print(data)
data = normalize(data)
print(data)


# plot des données en utilisant PCA dimensions (3D)
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("Les trois dimensions de PCA")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()




# Choix du nombre de clusters en utilisant l'inertia

nb_clusters = [2, 3 ,4 ,5 ,6 ,7 ,8, 9 ,10]
iner = []
for clus in nb_clusters:
    kmeans = KMeans(n_clusters=clus, random_state=10).fit(data)
    iner.append(kmeans.inertia_)

    
plt.figure(2)
plt.plot(nb_clusters,iner)
plt.title("Choix du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertia du model avec i clusters")
plt.show()


#Utilisation de PCA pour décomposer les données
pca = PCA(n_components=3).fit_transform(data)


#mesure de temps d'exécution
start = time.time()
# Application de K-means
kmeans = KMeans(n_clusters=2, random_state=10).fit(data)
y = kmeans.fit_predict(data)

end = time.time()

temps['k-means']= round((end - start)/2,4)
#Calcul de silhouette
silhouette_dict['k-means'] =  silhouette_score(pca, y)

# plot des données en utilisant K-means
fig = plt.figure(3, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("K-means")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()


#mesure de temps d'exécution
start = time.time()
# Application de GMM

gmm = GaussianMixture(n_components=2,covariance_type = 'spherical').fit(data)
y=gmm.predict(data)

end = time.time()

temps['GMM']= round((end - start)/2,4)
#Calcul de silhouette
silhouette_dict['GMM'] =  silhouette_score(pca, y)  

# plot des données en utilisant GMM
fig = plt.figure(4, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("GMM")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()




#mesure de temps d'exécution
start = time.time()
# Application de CAH

cah = AgglomerativeClustering(distance_threshold=None, n_clusters=2, linkage='ward').fit(data)
y = cah.fit_predict(data)


end = time.time()

temps['CAH']= round((end - start)/2,4)
#Calcul de silhouette
silhouette_dict['CAH'] =  silhouette_score(pca, cah.labels_)

# plot des données en utilisant CAH
fig = plt.figure(5, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("CAH")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()



#mesure de temps d'exécution
start = time.time()
#Application de DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=8, metric='euclidean').fit(data)
y = dbscan.fit_predict(data)


end = time.time()

temps['DBSCAN']= round((end - start)/2,4)
#Calcul de silhouette
silhouette_dict['DBSCAN'] =  silhouette_score(pca, y)

# plot des données en utilisant DBSCAN
fig = plt.figure(6, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("DBSCAN")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()



#mesure de temps d'exécution
start = time.time()
#Application de SpectralClustering
spect_model = SpectralClustering(n_clusters=2, n_init=10, gamma=-0.2, affinity='nearest_neighbors', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1).fit(data)
y = spect_model.fit_predict(data)

end = time.time()

temps['SpectralClustering']= round((end - start)/2,4)
#Calcul de silhouette
silhouette_dict['SpectralClustering'] =  silhouette_score(pca, y)

# plot des données en utilisant DBSCAN
fig = plt.figure(7, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("Spectral Clustering")
ax.set_xlabel("dim 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("dim 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("dim 3")
ax.w_zaxis.set_ticklabels([])
plt.show()

print(silhouette_dict)
print(temps)
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:47:55 2022

@author: vaishnav
"""
#=================================================================================================================================================
#importing the data

import pandas as pd
df = pd.read_csv(r"C:\anaconda\New folder (2)\wine.csv")
df

#=================================================================================================================================================

df["Type"].value_counts().plot(kind='bar')


df.describe()

#=================================================================================================================================================

df1 = df.drop('Type',axis=1)
df1

df1.describe()
df1.info()

#=================================================================================================================================================
#normalizing the data
from sklearn.preprocessing import scale

df1 = scale(df1)
df1

#=================================================================================================================================================
#Applying PCA Fit Transform to dataset

from sklearn.decomposition import PCA

pca=PCA(n_components=13)

df1_pca=pca.fit_transform(df1)
df1_pca


pca.components_

pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

#=================================================================================================================================================

import numpy as np
#cummulative variane
var=np.cumsum(np.round(pca.explained_variance_ratio_,4)*100)

import matplotlib.pyplot as plt
# Variance plot for PCA components obtained 
plt.plot(var,color='blue')

#=================================================================================================================================================

# Final Dataframe
df2=pd.concat([df['Type'],pd.DataFrame(df1_pca[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
df2

import seaborn as sns
#visualization of pc

fig=plt.figure(figsize=(10,8))
sns.scatterplot(data=df2)

sns.scatterplot(data=df2, x='PC1', y='PC2', hue='Type')

#=================================================================================================================================================
#clustering
#Agglomerative/Hierarchical 

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title('customer dendogram')
dend = shc.dendrogram(shc.linkage(df2,method='complete'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward")
Y=cluster.fit_predict(df2)
Clusters=pd.DataFrame(Y,columns=['Clusters'])


wine=df.copy()
wine['h_clusterid'] = cluster.labels_
wine.h_clusterid.value_counts().plot(kind='bar')

#===============================================================================================================
#Kmeans Clustering

from sklearn.cluster import KMeans
kmeans=KMeans().fit(df2)
# let's find the optimum number of clusters;
score=[]
K=range(1,7)

for i in K:
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=3)
    kmeans.fit(df2)
    score.append(kmeans.inertia_)

#visualize;

plt.plot(K,score,color="red")
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()



from yellowbrick.cluster import KElbowVisualizer
# for K-elbow;

kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,7))
visualizer.fit(df2)
visualizer.poof()
plt.show()

cluster=KMeans(n_clusters=3,init="k-means++").fit(df2)
#add tag values;

cluster.labels_

wine2 = df.copy()
wine2["ClusterID"] = cluster.labels_
wine2.ClusterID.value_counts().plot(kind='bar')









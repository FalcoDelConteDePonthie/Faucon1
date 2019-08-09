import numpy as np
import pylab as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# Metodo che serve per leggere il file CSV
def leggiCsv(nomeFile,separatore):
    file = pd.read_csv(nomeFile,separatore,dtype='float64')
    return file
#https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
data = leggiCsv("shuttle.csv",";")
scaler = MinMaxScaler(feature_range=[0, 1])
data = np.array(data)
print(data)
data_rescaled = scaler.fit_transform(data[1:, 0:8])
#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Shuttle Dataset Explained Variance')
plt.show()
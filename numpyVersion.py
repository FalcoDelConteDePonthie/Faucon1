import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
# This method select random initial centroid from dataset
# and memorize it in "centroidi" vector for returned it
def calcolaCentroidiRandom(data):
    centroidi = []
    i = 0
    # Finche non arrivo in posizione colonna clustering
    while i < int(Numclust):
        # Estrae il punto dal data point in modo randomico
        idx = int(int(0) + (random.random() * (int(N) - int(0))))
        i += 1
        centroidi.append(data[idx].tolist())   
    centroidi = np.array(centroidi)
    return centroidi   

# Metodo che serve per leggere il file CSV
def leggiCsv(nomeFile,separatore):
    file = pd.read_csv(nomeFile,separatore,dtype='float64')
    return file


def normalizzaDati(dati):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(dati)
    df_normalized = pd.DataFrame(np_scaled)
    return np_scaled,df_normalized


def dist(x, l): 
    dSum = 0.0	
    for j in range(0,D):
        dSum += pow(x[j] - Z[l][j], 2.0)
    return dSum

def getListPair(Numclust,dati):
    lista = pd.DataFrame(columns=list('AB')) 
    vD = []
    for i in range(0,N):
        dMin = float('inf')
        s = -1
        for j in range(0,Numclust):
            dDist = dist(dati[i],j)
            if dMin > dDist:
                dMin = dDist
                s = j
        vD.append(s)
        lista = lista.append({'A': dMin,'B': i}, ignore_index=True)
    lista = lista.sort_values(by=['A'],ascending=False)
    return lista,vD

def getOutlier(nOutliers,CM,listPair,lstcluster):
    for j in range(0,nOutliers):
        i = int(listPair.iloc[j]['B'])
        if int(CM[i]) != int(Numclust):
            lstcluster[int(CM[i])].remove(data[i].tolist())
            lstcluster[Numclust].append(data[i].tolist())          
            CM[i] = Numclust
    return lstcluster,CM


def getNotOutlier(nOutliers,CM,listPair,lstcluster,vd):
    for l in range(nOutliers,N):
        i = int(listPair.iloc[l]['B'])
        s = int(vd[i])
        if int(CM[i]) != int(s):
            lstcluster[int(CM[i])].remove(data[i].tolist())
            lstcluster[s].append(data[i].tolist())
            CM[i] = s
    return lstcluster,CM        
           
def updateU(dati,dAvgDist,lstcluster,cm):
    listPair, vd = getListPair(Numclust,dati)        
    nOutliers = 0
    for i in range(0,N0):
        if listPair.iloc[i]['A'] <= dAvgDist:
            break
        nOutliers=i+1
    
    lstcluster,cm = getOutlier(nOutliers,cm,listPair,lstcluster)
            
    lstcluster,cm = getNotOutlier(nOutliers,cm,listPair,lstcluster,vd)

    
    return lstcluster, CM 
       
       
def calculateObj(data,lstCluster,Z):    
    dSum1 = 0.0
    for i in range(0,N):
        j = CM[i]
        if j < Numclust:
            dSum1 += dist(data[i], int(j))

    dAvgDist = dSum1 * Gamma / (N - len(lstCluster[Numclust]))
    dobj = dSum1 + dAvgDist * len(lstCluster[Numclust])
    return dAvgDist, dobj  
       
def updateZ(lstcluster,Z):
    dTemp = 0.0
    for k in range(0,Numclust): 
        for j in range(0,D):
            dTemp = 0.0
            for i in range(0,len(lstcluster[k])):
                rec = lstcluster[k][i]
                dTemp += rec[j]
            if len(lstcluster[k]) != 0:
                Z[k][j] = dTemp / len(lstcluster[k])
            else:
                Z[k][j] = Z[k][j]
    return Z



# Primo Step inizializzare in modo randomico i centroidi dopo aver letto e normalizzato i dati
dati = leggiCsv("shuttle.csv",";")
# Per convenzione uso la maiuscola per indicare che sono costanti
N = np.shape(dati)[0]
D = np.shape(dati)[1]    
data,dataframeNorm = normalizzaDati(dati)
print(type(data))
Gamma = 9
Delta = pow(10,-6)
Numclust = 3
Maxiter = 100
P0 = 0.1
N0 = int(P0*N)
TOT = 0
# Secondo Step per ogni centroide prendo il data point e cerco per quale dei k
# centroidi la distanza è minore. A quello assegno il punto
for i in range(0,1):
    CM = []
    CM = np.array(CM, dtype="float64")
    Z = []
    Z = np.array(Z, dtype="float64")
    Z =  calcolaCentroidiRandom(data)

    print(Z)
    lstCluster = []
    for i in range(0,Numclust+1):
        app = []
        lstCluster.append(app)
    s = -1
    for i in range(0,N):
        dMin = float('inf')
        s=-1
        for j in range(0,Numclust):
            dDist = dist(data[i],j)
            if dDist < dMin:
                s = j
                dMin = dDist
        lstCluster[s].append(data[i].tolist())
        CM = np.append (CM, s)

    dAvgDist, dobj = calculateObj(data,lstCluster,Z)
    numiter = 1
    while(1==1):	
        lstCluster, CM = updateU(data,dAvgDist,lstCluster,CM)
        Z = updateZ(lstCluster,Z)
        dObjPre = dobj
        dAvgDist, dobj = calculateObj(data,lstCluster,Z)
        if abs(dObjPre - dobj) < Delta:
            break
        numiter+=1
        if numiter > Maxiter:
            break
        print(len(lstCluster[-1]))
    from sklearn.decomposition import PCA
    import pylab as pl
    for i in range(0,Numclust):        
        dataframeNorm.loc[len(dataframeNorm)] = Z[i]
        CM = np.append(CM,9)
    X = dataframeNorm.iloc[:, 0:9].values
    y = CM
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    print(pca.explained_variance_ratio_)
    #print(pca.explained_variance_)
    for i in range(0, pca_2d.shape[0]):
        if y[i] == 0:#Primo Cluster
            c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
        elif y[i] == 1:#Secondo Cluster
            c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
        elif y[i] == 2:#Terzo Cluster
            c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='y', marker='x')
        elif y[i] == 3:#Outlier
            c4 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
        elif y[i] == 9:#Centroidi
            c5 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='k', marker='X')
    
    pl.legend([c1, c2, c3, c4, c5], ['Cluster0','Cluster1','Cluster2','Outlier','Centroide'])
    pl.title('Classificazione outlier con tre centroidi Shuttle normalizzato')
    pl.show()    

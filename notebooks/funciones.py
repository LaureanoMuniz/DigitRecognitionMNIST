import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn import metrics
import time
import metnum

def construir_dataset(numeroElementos = 42000):
	df_train = pd.read_csv("../data/train.csv")
	df_train = sk.utils.shuffle(df_train, random_state = 42)
	cutting_point = len(df_train) - (len(df_train)//5)
	df_test = df_train[cutting_point:]
	df_train = df_train[:cutting_point]
	# Uso values para mandar todo a arrays de numpy
	X_test = df_test[df_test.columns[1:]].values
	y_test = df_test["label"].values.reshape(-1, 1)
	df_train = df_train[:min(numeroElementos,cutting_point)]
	X = df_train[df_train.columns[1:]].values
	y = df_train["label"].values.reshape(-1, 1)
	return(X, y, X_test, y_test)

def get_KFold_sets(x,y,K=5):
    X_trains = []
    Y_trains = []
    X_vals = []
    Y_vals = []
    bucket_size = len(x)//K
    for i in range(K):
        low = bucket_size*i
        high = bucket_size * (i+1)
        X_vals.append(x[low :high])
        Y_vals.append(y[low :high])
        X_train,Y_train = x[:low], y[:low]
        X_train = np.concatenate((X_train,x[high:]),axis=0)
        Y_train = np.concatenate((Y_train,y[high:]),axis=0)
        X_trains.append(X_train)
        Y_trains.append(Y_train)
    return (X_trains,Y_trains,X_vals,Y_vals)


def encontrarParOptimo(X_trains,Y_trains,X_vals,Y_vals,conPeso=False):
	accspar = []
	for k in tqdm(range(1,16,1)):
	    for alpha in range(25,36):
	        acc = 0
	        for i in range(len(X_trains)):
	            pca = metnum.PCA(alpha)
	            pca.fit(X_trains[i])
	            X_train_transformada = pca.transform(X_trains[i])
	            X_val_transformada = pca.transform(X_vals[i])
	            clf = metnum.KNNClassifier(k, conPeso)
	            clf.fit(X_train_transformada,Y_trains[i])
	            y_pred = clf.predict(X_val_transformada)
	            acc += sk.metrics.accuracy_score(Y_vals[i], y_pred)
	        acc = acc/len(X_trains)
	        accspar.append((acc,k,alpha)) 

	accspar.sort(reverse = True)
	mejores5pares = accspar[:5]
	minimosf1 = minimizarF1(mejores5pares,X_trains,Y_trains,X_vals,Y_vals,conPeso)
	return minimosf1[0]

def minimizarF1(mejores5,X_trains,Y_trains,X_vals,Y_vals,conPeso=False):
	minimosf1 = []
	digitosConPeoresF1 = []
	for tupla in mejores5:
		f1_scorePorClase = np.zeros(10)
		for i in range(len(X_trains)):
		    pca = metnum.PCA(tupla[2])
		    pca.fit(X_trains[i])
		    X_train_transformada = pca.transform(X_trains[i])
		    X_val_transformada = pca.transform(X_vals[i])
		    clf = metnum.KNNClassifier(tupla[1])
		    clf.fit(X_train_transformada, Y_trains[i])
		    y_pred = clf.predict(X_val_transformada)
		    f1_scorePorClase += np.array(sk.metrics.f1_score(Y_vals[i],y_pred, average = None))
		f1_scorePorClase = f1_scorePorClase/len(X_trains)
		minimosf1.append((np.amin(f1_scorePorClase),tupla[0],tupla[1],tupla[2]))
	minimosf1.sort(reverse = True)
	return(minimosf1)

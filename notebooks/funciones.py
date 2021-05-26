import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn import metrics
import time

def construir_tp():
	!cd .. && mkdir build
	!cd ../build/ && rm -rf *
	!cd ../build && cmake \
	  -DPYTHON_EXECUTABLE="$(which python)" \
	  -DCMAKE_BUILD_TYPE=Release ..
	!cd ../build && make install

def construir_dataset(numeroElementos = 42000):
	df_train = pd.read_csv("../data/train.csv")
	df_train = sk.utils.shuffle(df_train, random_state = 42)
	df_train = df_train[:numeroElementos]
	cutting_point = len(df_train) - (len(df_train)//5)
	df_test = df_train[cutting_point:]
	df_train = df_train[:cutting_point]
	# Uso values para mandar todo a arrays de numpy
	X = df_train[df_train.columns[1:]].values
	y = df_train["label"].values.reshape(-1, 1)
	X_test = df_test[df_test.columns[1:]].values
	y_test = df_test["label"].values.reshape(-1, 1)
	return(X, Y, X_test, y_test)

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
    return X_trains,Y_trains,X_vals,Y_vals




def encontrarParOptimo(X_trains,Y_trains,X_vals,Y_vals,conPeso=False):
	accspar = []
	for k in tqdm(range(1,16,1)):
	    for alpha in range(25,36):
	        acc = 0
	        X_trains_transformadas = []
	        X_vals_transformadas = []
	        for i in range(len(X_trains)):
	            pca = metnum.PCA(alpha)
	            pca.fit(X_trains[i])
	            X_trains_transformadas.append(pca.transform(X_trains[i]))
	            X_vals_transformadas.append(pca.transform(X_vals[i]))
	            clf = metnum.KNNClassifier(k, conPeso = True)
	            clf.fit(X_trains_transformadas[i],Y_trains[i])
	            y_pred = clf.predict(X_vals_transformadas[i])
	            acc += sk.metrics.accuracy_score(Y_vals[i], y_pred)
	        acc = acc/len(X_trains)
	        accspar.append((acc,k,alpha)) 

	accspar.sort(reverse = True)
	mejores5pares = accspar
	mejores5pares = mejores5pares[:5]
	print(mejores5pares)

	minimosf1 = []
	digitosConPeoresF1 = []
	for tupla in mejores5pares:
	    clf = metnum.KNNClassifier(tupla[1])
	    clf.fit(X, y)
	    y_pred = clf.predict(X_test)
	    f1_scorePorClase = sk.metrics.f1_score(y_test,y_pred, average = None)
	    digitosConPeoresF1.append(np.where(f1_scorePorClase == np.amin(f1_scorePorClase))[0][0])
	    minimosf1.append((np.amin(f1_scorePorClase),(tupla[1],tupla[2])))
	    
	minimosf1.sort(reverse = True)
	print(minimosf1)
	print(digitosConPeoresF1)
	print("Nuestra par ideal es ", (minimosf1[0][1]))
# %load mnist_loader.py

import matplotlib.pyplot as plt
import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
  

import pickle

with open("mnist.pkl", "rb") as f:
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

def ReLU(x):
    return np.maximum(0,x)

def ReLU_pochodna(x):
    return np.greater(x, 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_pochodna(s):
    s = s.ravel() 
    jacobian = np.diag(s) - np.outer(s, s)
    return jacobian

def oblicznie_nodów(wartosci_wejscia, bias, wagi):
    return wagi @ wartosci_wejscia + bias

nody_wejście = 784 # 784 pikseli na obraz
nody_warstwa_1 = 32
nody_warstwa_2 = 32
nody_wyjście = 10 # 10 możliwych wyników cyfr 0-9

# wagi i bias

bias_1 = np.zeros((nody_warstwa_1, 1)) # (20,1)
wagi_1 = np.random.randn(nody_warstwa_1, nody_wejście) * 0.1 # (20, 784)

bias_2 = np.zeros((nody_warstwa_2, 1)) # (20,1)
wagi_2 = np.random.randn(nody_warstwa_2, nody_warstwa_1) * 0.1 # (20, 20)

bias_3 = np.zeros((nody_wyjście, 1)) # (10,1)
wagi_3 = np.random.randn(nody_wyjście, nody_warstwa_2) * 0.1 # (10, 20)

###############################

epoki = 20

for e in range(epoki):

  dobrze_zaklasyfikowane = 0

  for k in range(training_data[0].shape[0]):

    pierwszy_obraz = training_data[0][k].reshape(-1,1)  # (784,1)

    oczekiwana_liczba = training_data[1][k] # wartość
    oczekiwany_wynik = np.zeros((nody_wyjście,1)) # (10, 1)
    oczekiwany_wynik[oczekiwana_liczba] = 1

    # obliczenia na warstwach

    wyniki_1 = ReLU(oblicznie_nodów(pierwszy_obraz, bias_1, wagi_1)) # (20, 1)
    wyniki_2 = ReLU(oblicznie_nodów(wyniki_1, bias_2, wagi_2)) # (20, 1)
    wyniki_3 = softmax(oblicznie_nodów(wyniki_2, bias_3, wagi_3)) # (10, 1)

    ######

    wyznaczona_liczba = np.argmax(wyniki_3) # wartość

    if wyznaczona_liczba == oczekiwana_liczba:
          dobrze_zaklasyfikowane += 1

    # błąd średniokwadratowy = 1/10 sum(1:10)(wynik_i-oczekiwany_i)^2
    koszt = np.mean(np.square(wyniki_3 - oczekiwany_wynik)) # wartość

    learning_rate = 0.1 # wartość

    # dla jednego mamy pochodną = 2(wynik-oczekiwana)
    pochodna_kosztu = 2 * (wyniki_3 - oczekiwany_wynik) / nody_wyjście  #(10,1)

    # wagi_3 = wagi_3 - learning_rate * pochodna_kosztu/pochodna_wyniki_3
    # pochodna funkci aktywacyjnej * pochodna kosztu * wyniki_(i-1)
    # (10,10) * (10,1) = (10,1)  || (10,1) * (20, 1).T = (10,1) * (1, 20) = (10,20)
    #pochodna_wagi_3 = softmax_pochodna(wyniki_3) @ pochodna_kosztu @ wyniki_2.T # (10, 20)

    c_w_3_pochodna = softmax_pochodna(wyniki_3) @ pochodna_kosztu # (10, 1)
    wagi_3_nowe = wagi_3 - learning_rate * (c_w_3_pochodna @ wyniki_2.T) # (10, 20) poprawny wymiar wagi_3
    bias_3_nowe = bias_3 - learning_rate * c_w_3_pochodna # (10, 1) | nie mnożymy razy wyniki_2, bo tutaj pochodna wynosi 1

    # (10,20).T * (10,1) = (20,10) * (10,1) = (20,1)
    c_w_2_pochodna = wagi_3.T @ c_w_3_pochodna * ReLU_pochodna(wyniki_2) # (20,1)
    wagi_2_nowe = wagi_2 - learning_rate * (c_w_2_pochodna @ wyniki_1.T) # (20, 20) poprawny wymiar wagi_2
    bias_2_nowe = bias_2 - learning_rate * c_w_2_pochodna # (20,1)

    # analogicznie jak dla wagi_2
    c_w_1_pochodna = wagi_2.T @ c_w_2_pochodna * ReLU_pochodna(wyniki_1) # (20,1)
    wagi_1_nowe = wagi_1 - learning_rate * (c_w_1_pochodna @ pierwszy_obraz.T) # (20, 784) poprawny wymiar wagi_1
    bias_1_nowe = bias_1 - learning_rate * c_w_1_pochodna # (20,1)

    wagi_1 = wagi_1_nowe
    wagi_2 = wagi_2_nowe
    wagi_3 = wagi_3_nowe

  print(f"Epoka {e+1}/{epoki} - {dobrze_zaklasyfikowane/training_data[0].shape[0]}")
###############################

przejścia = 0
dobrze_zaklasyfikowane_test = 0

for k in range(test_data[0].shape[0]):

  pierwszy_obraz = test_data[0][k].reshape(-1,1)
  oczekiwana_laleiczba = test_data[1][k]
  oczekiwany_wynik = np.zeros((nody_wyjście,1))
  oczekiwany_wynik[oczekiwana_liczba] = 1

  wyniki_1 = ReLU(oblicznie_nodów(pierwszy_obraz, bias_1, wagi_1)) # (20, 1)
  wyniki_2 = ReLU(oblicznie_nodów(wyniki_1, bias_2, wagi_2)) # (20, 1)
  wyniki_3 = softmax(oblicznie_nodów(wyniki_2, bias_3, wagi_3)) # (10, 1)

  wyznaczona_liczba = np.argmax(wyniki_3) # wartość
  prawd_wyznaczonej_liczby = max(wyniki_3) # wartość

  if wyznaczona_liczba == oczekiwana_liczba:
        dobrze_zaklasyfikowane_test += 1

  przejścia +=1

  if przejścia % 500 == 0:
    print(przejścia, dobrze_zaklasyfikowane_test/przejścia)

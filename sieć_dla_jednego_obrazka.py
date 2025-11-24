pierwszy_obraz = test_data[0][0].reshape(-1,1)  # (784,1)
length = len(pierwszy_obraz)

nody_wejście = length # 784
nody_warstwa_1 = 20
nody_warstwa_2 = 20
nody_wyjście = 10 # 10 możliwych wyników cyfr 0-9

oczekiwana_liczba = test_data[1][0] # wartość
oczekiwany_wynik = np.zeros((nody_wyjście,1)) # (10, 1)
oczekiwany_wynik[oczekiwana_liczba] = 1

def ReLU(x):
  return np.maximum(0,x)

def ReLU_pochodna(x):
    return np.greater(x, 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_pochodna(s):
    si_sj = - s @ s.T
    diagonala = np.diag((s * (1 - s)).flatten()) 
    return diagonala

def oblicznie_nodów(wartości_wejścia, bias, wagi, nody_wchodzące, nody_wychodzące):

    wynik = np.zeros((nody_wychodzące, 1))

    for i in range(nody_wychodzące):
      for j in range(nody_wchodzące):
        wynik[i,0] += wartości_wejścia[j,0] * wagi[i,j]
      wynik[i,0] += bias[i, 0]

    return wynik

# wagi i bias

bias_1 = np.zeros((nody_warstwa_1, 1)) # (20,1)
wagi_1 = np.random.randn(nody_warstwa_1, nody_wejście) * 0.1 # (20, 784)

bias_2 = np.zeros((nody_warstwa_2, 1)) # (20,1)
wagi_2 = np.random.randn(nody_warstwa_2, nody_warstwa_1) * 0.1 # (20, 20)

bias_3 = np.zeros((nody_wyjście, 1)) # (10,1)
wagi_3 = np.random.randn(nody_wyjście, nody_warstwa_2) * 0.1 # (10, 20)

# obliczenia na warstwach

wyniki_1 = oblicznie_nodów(pierwszy_obraz, bias_1, wagi_1, nody_wejście, nody_warstwa_1)
wyniki_1 = ReLU(np.array(wyniki_1)) # (20, 1)

wyniki_2 = oblicznie_nodów(wyniki_1, bias_2, wagi_2, nody_warstwa_1, nody_warstwa_2)
wyniki_2 = ReLU(np.array(wyniki_2)) # (20, 1)

wyniki_3 = oblicznie_nodów(wyniki_2, bias_3, wagi_3, nody_warstwa_2, nody_wyjście)
wyniki_3 = softmax(np.array(wyniki_3)) # (10, 1)

######

wyznaczona_liczba = np.argmax(wyniki_3) # wartość
prawd_wyznaczonej_liczby = max(wyniki_3) # wartość

# błąd średniokwadratowy = 1/10 sum(1:10)(wynik_i-oczekiwany_i)^2
koszt = np.mean(np.square(wyniki_3 - oczekiwany_wynik)) # wartość

learning_rate = 0.01 # wartość

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

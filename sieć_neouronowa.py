import numpy as np
import matplotlib.pyplot as plt

# wagi: 1_x są z inputu 1, a 2_x z inputu 2
weight_1_1 = 0.0
weight_2_1 = 0.0
weight_1_2 = 0.5
weight_2_2 = 0.5
bias_1 = 0.5
bias_2 = 1.0


# klasyfikacja, czy owoc jest bezpieczny:
# 0 - niebezpieczny
# 1 - bezpieczny
def classify(input_1, input_2):
    output_1 = input_1 * weight_1_1 + input_2 * weight_2_1 + bias_1
    output_2 = input_1 * weight_1_2 + input_2 * weight_2_2 + bias_2

    # zwraca 0, gdy output_1 większe, a 1 gdy output_2 większe
    if output_1 > output_2: 
      return(0)
    else: 
      return(1)


#################################################################################################################################
# TUTAJ JEST JAKAŚ WIZUALIZACJA TEGO

# przyporządkowanie koloru do punktu. Sprawdza punkty na grafie i decyduje, czy są trujące czy nie
def visualize(graphX, graphY, graph, safeColour, poisonousColour):
    predictedClass = classify(graphX, graphY)

    if predictedClass == 0:
        graph.setColour(graphX, graphY, safeColour)
    elif predictedClass == 1:
        graph.setColour(graphX, graphY, poisonousColour)

# przykładowe punkty 
points = [
    (-4, -1),
    (-3,  2),
    (-2, -3),
    (-1,  4),
    ( 0,  0),
    ( 1, -2),
    ( 2,  3),
    ( 3, -4),
    ( 4,  1)
]

# --- 1. Tworzymy heatmapę (grid) ---
x_vals = np.linspace(-5, 5, 300)
y_vals = np.linspace(-5, 5, 300)
xx, yy = np.meshgrid(x_vals, y_vals)

Z = np.zeros_like(xx)

# klasyfikujemy każdy punkt siatki
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z[i, j] = classify(xx[i, j], yy[i, j])


# --- 2. Rysowanie ---
plt.figure(figsize=(8, 8))

# HEATMAPA
plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.5)

# PUNKTY
point_x = [p[0] for p in points]
point_y = [p[1] for p in points]
point_classes = [classify(x, y) for x, y in points]

plt.scatter(point_x, point_y, c=point_classes, cmap="coolwarm", edgecolor="black", s=80)

plt.title("Heatmapa klasyfikacji + Twoje punkty")
plt.xlabel("input_1")
plt.ylabel("input_2")
plt.grid(True)

plt.show()

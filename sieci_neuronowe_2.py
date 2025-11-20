import numpy as np
import matplotlib.pyplot as plt

# wagi: 1_x są z inputu 1, a 2_x z inputu 2
weight_1_1 = 0.0
weight_2_1 = 0.0
weight_1_2 = 0.5
weight_2_2 = 0.5
bias_1 = 0.5
bias_2 = 1.0



public class Layer{
    int numNodesIn, numNodesOut;
    double[,] weights; # wagi
    double[] biases; # biasy

    # tworzenie warstwy
    public Layer(int numNodesIn, int numNodesOut){
        this.numNodesIn = numNodesIn; # ilość inputów do warstwy
        this.numNodesOut = numNodesOut; # ilość outputów z warstwy

        weights = new double[numNodesIn, numNodesOut];
        biases = new double[numNodesOut];
    }   

    # obliczanie wyniku warstwy
    public double[] CalcuteOutputs(double[] inputs){ # pobiera inputy
        double[] weightedInputs = new double[numNodesOut]; # tworzy weighted inputs

        for(int nodeOut = 0; nodeOut < numNodesOut; nodeOut++){
            double weightedinput = biases[nodeOut]; # dodajemy bias
            for(int nodeIn = 0; nodeIn<numNodesIn; nodeIn++){
                weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut]; # dodajemy inputy * ich wagi
            }
            weightedInputs[nodeOut] = weightedInput;
        }
        return weightedInputs; # warstwa zwraca weightedInputs
    }
}

public class NeuralNetwork{
    Layer[] layers; # 'szereg' warst

    # stworzenie sieci neuronowej
    public NeuralNetwork(params int[] layerSizes){ # podanie ile nodes ma być w każdej z warst i je tworzy
        layers = new Layer[layerSizes.length - 1];
        for(int i = 0; i < layers.Length; i++){
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }
    }

    # oblicznie outputs całej sieci
    double[] CalculateOutputs(double[] inputs){
        foreach(Layer layer in layers){ # przechodzi przez wszystkei warstwy w naszej sieci
            inputs = layer.CalculateOutputs(inputs); # oblicza output warstwy i używa jako input do kolejnej warstwy
        }
        return inputs; # więc po przejściu całej sieci, nasze 'inputs' stają się tak na prawdę outputs, bo nie ma gdzie ich dalej podać
    }

    int Classify(double[] inputs){
        double[] outputs = CalculateOutputs(inputs); # oblicza outputs
        return IndexOfMaxValue(outputs); # sprawdza, który output jest największy
    }
}

NeuralNetwork network = new NeuralNetwork(2,3,2);

###################################################

import numpy as np

class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in      # ilość inputów do warstwy
        self.num_nodes_out = num_nodes_out    # ilość outputów z warstwy
        self.weights = np.random.randn(num_nodes_in, num_nodes_out)  # wagi (losowo)
        self.biases = np.zeros(num_nodes_out)                        # biasy

    def calculate_outputs(self, inputs):
        """
        Oblicza output warstwy: weighted sum = inputs*weights + bias
        """
        weighted_inputs = np.zeros(self.num_nodes_out)
        for node_out in range(self.num_nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in, node_out]
            weighted_inputs[node_out] = weighted_input
        return weighted_inputs


class NeuralNetwork:
    def __init__(self, *layer_sizes):
        """
        Tworzy sieć neuronową z podanych rozmiarów warstw
        np. NeuralNetwork(2,3,2)
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs):
        """
        Przepuszcza wejście przez wszystkie warstwy sieci
        """
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        outputs = self.calculate_outputs(inputs)
        return int(np.argmax(outputs))  # indeks największej wartości


# przykład użycia
network = NeuralNetwork(2, 3, 2)
inputs = np.array([0.5, 0.8])
outputs = network.calculate_outputs(inputs)
prediction = network.classify(inputs)

print("Outputs:", outputs)
print("Predicted class:", prediction)

#######################################################

import numpy as np

class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        
        # Inicjalizacja wag i biasów
        # W C# 'new double[,]' tworzy same zera. W sieciach neuronowych zazwyczaj 
        # chcemy losowe wagi na start, dlatego używam random.randn.
        # Jeśli chcesz same zera jak w C#, użyj: np.zeros((num_nodes_in, num_nodes_out))
        self.weights = np.random.randn(num_nodes_in, num_nodes_out)
        self.biases = np.zeros(num_nodes_out)

    def calculate_outputs(self, inputs):
        """
        Oblicza wynik warstwy.
        Zamiast dwóch pętli for (jak w C#), używamy mnożenia macierzy (dot product).
        Wzór: inputs * weights + biases
        """
        # To jedno polecenie zastępuje zagnieżdżone pętle z C#:
        return np.dot(inputs, self.weights) + self.biases


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Tworzy sieć na podstawie listy rozmiarów warstw, np. [2, 3, 2]
        odpowiednik C#: params int[] layerSizes
        """
        self.layers = []
        # Pętla tworząca warstwy (od i do i+1)
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

    def calculate_outputs(self, inputs):
        """
        Przepuszcza dane przez całą sieć (Feed Forward)
        """
        # Upewniamy się, że input to tablica numpy
        current_inputs = np.array(inputs)
        
        for layer in self.layers:
            current_inputs = layer.calculate_outputs(current_inputs)
            
        return current_inputs

    def classify(self, inputs):
        """
        Zwraca indeks neuronu z największą wartością na wyjściu
        """
        outputs = self.calculate_outputs(inputs)
        # np.argmax to odpowiednik Twojej funkcji IndexOfMaxValue
        return np.argmax(outputs)

# --- PRZYKŁAD UŻYCIA (odpowiednik main) ---

# 1. Tworzymy sieć: 2 wejścia -> 3 neurony ukryte -> 2 wyjścia
network = NeuralNetwork([2, 3, 2])

# 2. Przykładowe dane wejściowe
inputs = [0.5, 0.8]

# 3. Obliczenie wyniku (surowe wartości)
outputs = network.calculate_outputs(inputs)
print(f"Surowe wyjścia sieci: {outputs}")

# 4. Klasyfikacja (który indeks wygrał)
predicted_class = network.classify(inputs)
print(f"Przewidziana klasa (indeks): {predicted_class}")

# Opcjonalnie: Ręczne ustawienie wag (jak w zmiennych na górze Twojego kodu)
# Dostęp do pierwszej warstwy: network.layers[0]
# network.layers[0].weights[0, 1] = 0.5

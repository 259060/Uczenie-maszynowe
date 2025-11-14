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


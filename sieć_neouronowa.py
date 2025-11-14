# wagi: 1_x są z inputu 1, a 2_x z inputu 2
weight_1_1 = 0.0
weight_2_1 = 0.0
weight_1_2 = 0.0
weight_2_2 = 0.0


# klasyfikacja, czy owoc jest bezpieczny:
# 0 - niebezpieczny
# 1 - bezpieczny
def classify(input_1, input_2):
    output_1 = input_1 * weight_1_1 + input_2 * weight_2_1
    output_2 = input_1 * weight_1_2 + input_2 * weight_2_2

    # zwraca 0, gdy output_1 większe, a 1 gdy output_2 większe
    if output_1 > output_2: 
      return(0)
    else: 
      return(1)


def visualize(graphX, graphY, graph, safeColour, poisonousColour):
    predictedClass = classify(graphX, graphY)

    if predictedClass == 0:
        graph.setColour(graphX, graphY, safeColour)
    elif predictedClass == 1:
        graph.setColour(graphX, graphY, poisonousColour)

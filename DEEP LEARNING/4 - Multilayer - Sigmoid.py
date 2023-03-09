import numpy as np
from sklearn.metrics import accuracy_score


def sigmoid(soma):

    """

        FUNÇÃO PARA CALCULA A FUNÇÃO SIGMOID

    """

    return 1/(1 + np.exp(-soma))


def get_error(predict, y_true):

    """

        CALCULA O ERRO ABSOLUTO

    """

    # OBTENDO A QUANTIDADE ABSOLUTA DE ERROS
    absolute_error = y_true - predict

    return absolute_error, np.mean(absolute_error)


def execute_model(epochs=100):

    # OBTENDO A ENTRADA
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    # CLASSES TRUE
    y_true = np.array([[0], [1], [1], [1]])

    # PESOS - ENTRADA CAMADA OCULTA
    weight_input_hidden_layer = np.array([[-0.424, -0.740, -0.961],
                                          [0.358, -0.577, -0.469]])

    # PESOS - SAIDA CAMADA OCULTA
    weight_output_hidden_layer = np.array([[-0.017], [-0.893], [0.148]])

    for epoch in range(epochs):
        # FUNÇÃO SOMA DO INPUT DA CAMADA OCULTA
        sum_sinapse_in_hidden_layer = np.dot(inputs,
                                             weight_input_hidden_layer)

        # RESULTADO DA FUNÇÃO DE ATIVAÇÃO
        result_activation_function_in_hidden_layer = sigmoid(sum_sinapse_in_hidden_layer)

        print("RESULTADOS DA CAMADA OCULTA")
        print("FUNÇÃO SOMA")
        print(sum_sinapse_in_hidden_layer)
        print("FUNÇÃO DE ATIVAÇÃO")
        print(result_activation_function_in_hidden_layer)
        print("-"*50)

        # FUNÇÃO SOMA DO OUTPUT DA CAMADA OCULTA
        sum_sinapse_out_hidden_layer = np.dot(result_activation_function_in_hidden_layer,
                                              weight_output_hidden_layer)

        # RESULTADO DA FUNÇÃO DE ATIVAÇÃO
        result_activation_function_out_hidden_layer = sigmoid(sum_sinapse_out_hidden_layer)

        print("RESULTADOS DA CAMADA DE SAIDA")
        print("FUNÇÃO SOMA")
        print(sum_sinapse_out_hidden_layer)
        print("FUNÇÃO SOMA")
        print(result_activation_function_out_hidden_layer)
        print("-" * 50)

        # CALCULANDO O ERRO ABSOLUTO
        absolute_error, mean_absolute_error = get_error(predict=result_activation_function_out_hidden_layer,
                                                        y_true=y_true)

        print("ERRO ABSOLUTO")
        print(mean_absolute_error)
        print("-" * 50)


# NÚMERO DE ÉPOCAS DESEJADAS
epochs = 1

# EXECUTANDO O MODELO
execute_model(epochs)
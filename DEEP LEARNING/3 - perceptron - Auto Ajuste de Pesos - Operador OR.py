import numpy as np
from sklearn.metrics import accuracy_score


def get_result_step_function(result_sum_function):

    if result_sum_function >= 1:

        return 1

    return 0

def step_function(result_sum_function):

    """

        FUNÇÃO PARA CALCULA A FUNÇÃO DEGRAU

    """

    if not isinstance(result_sum_function, (tuple, list, np.ndarray)):
        result_sum_function = [result_sum_function]

    # PERCORRENDO PARA CADA VALOR DOS RESULTADOS OBTIDOS
    result_step_function = [get_result_step_function(value) for value in result_sum_function]

    return result_step_function


def sum_function(e, p):

    """

        FUNÇÃO PARA CALCULAR O PRODUTO ESCALAR ENTRE
        INPUTS E PESOS DO MODELO

    """

    # REALIZANDO O PRODUTO ESCALAR
    return e.dot(p)


def get_error(inputs, predict, y_true):

    # OBTENDO OS ELEMENTOS DIFERENTES DA CLASSIFICAÇÃO
    elements_error = {idx: inputs[idx] for idx in range(len(predict)) if predict[idx] != y_true[idx]}

    # OBTENDO O ERRO (INVERSO DA ACURÁCIA)
    error_rate = (1 - accuracy_score(y_pred=predict, y_true=y_true))

    # OBTENDO A QUANTIDADE ABSOLUTA DE ERROS
    absolute_error = len(y_true)*error_rate

    return elements_error, absolute_error, error_rate

def calculate_new_weights(weights, learning_rate, inputs, absolute_error):

    """

        new_weight = weight + learning_rate*value*absolute_error

    """

    for key, value_input in inputs.items():

        new_weights = []

        for idx, value in enumerate(value_input):

            # CALCULANDO OS NOVOS PESOS
            new_weights.append(weights[idx] + learning_rate * value * absolute_error)

    return new_weights


# OBTENDO A ENTRADA E OS PESOS
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
weights = [0, 0]

# CLASSES TRUE
outputs = [0, 1, 1, 1]

# OBTENDO PARÂMETRO DE TAXA DE APRENDIZADO
learning_rate = 0.1

# INICIALIZADO AS VARIÁVEIS PARA MENSURAR ERROS
# ERROR TARGET (%)
error_target = 0

# INICIALIZANDO O ERRO ATUAL
error_initial = np.inf

# INICIANDO CONTADOR DE RODADAS
contador_rodadas = 1

while error_initial > error_target:

    print("RODADA ATUAL: {} - PESOS COM VALOR: {}".format(contador_rodadas,
                                                          weights))
    print("ERRO ATUAL: {}".format(error_initial))

    # CALCULANDO A FUNÇÃO SOMA
    result_sum_function = sum_function(inputs, weights)

    # OBTENDO O OUTPUT
    result_step_function = step_function(result_sum_function)

    # OBTENDO A QUANTIDADE DE ERROR
    elements_error, absolute_error, error_initial = get_error(inputs=inputs,
                                                              predict=result_step_function,
                                                              y_true=outputs)

    if error_initial > error_target:

        # ATUALIZANDO OS PESOS
        weights = calculate_new_weights(weights, learning_rate, elements_error, absolute_error)

        contador_rodadas +=1

print("-"*50)
print("REDE NEURAL - PERCEPTRON - FINALIZADA COM SUCESSO")
print("RODADA FINAL: {} - PESOS COM VALOR: {}".format(contador_rodadas,
                                                      weights))
print("ERRO FINAL: {}".format(error_initial))
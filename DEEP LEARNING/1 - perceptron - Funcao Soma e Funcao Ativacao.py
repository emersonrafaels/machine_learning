import numpy as np


def soma(e, p):

    # REALIZANDO O PRODUTO ESCALAR

    return e.dot(p)


def stepFunction(soma):

    if soma >= 1:
        return 1

    return 0


# OBTENDO A ENTRADA E OS PESOS
inputs = np.array([0, 1])
weights = np.array([0.5, 0.5])

# FUNÇÃO SOMA
result_soma = soma(inputs, weights)

# FUNÇÃO DE ATIVAÇÃO - STEP FUNCTION
result_activation = stepFunction(result_soma)

print("RESULTADO STEP FUNCTION: {}".format(result_activation))

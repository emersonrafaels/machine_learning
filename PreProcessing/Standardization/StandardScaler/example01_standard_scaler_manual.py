import numpy as np

# Dados simples (ex: idade de clientes)
X = np.array([20, 30, 40, 50, 60])

# Média e desvio padrão
media = X.mean()
desvio = X.std()

X_standardizado = (X - media) / desvio

print("-"*50)
print("Dados originais:", X)
print("Média:", media)
print("Desvio padrão:", desvio)
print("-"*50)
print("Dados padronizados:", X_standardizado)
print("Nova média (deve ser ~0):", X_standardizado.mean())
print("Novo desvio padrão (deve ser ~1):", X_standardizado.std())
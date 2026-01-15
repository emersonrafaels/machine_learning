import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[20], [30], [40], [50], [60]])

# Usando StandardScaler do sklearn
scaler = StandardScaler()

"""
📌 Pontos IMPORTANTES

- O fit() aprende média e desvio
- O transform() aplica
- fit_transform() = atalho

"""

# Ajusta e transforma os dados
X_scaled = scaler.fit_transform(X)

print("-" * 50)
print("Dados padronizados:", X_scaled)
print("Média após padronização (deve ser ~0):", X_scaled.mean())
print("Desvio padrão após padronização (deve ser ~1):", X_scaled.std())

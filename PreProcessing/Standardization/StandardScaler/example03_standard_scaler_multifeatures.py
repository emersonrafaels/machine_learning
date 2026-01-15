import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[25, 3000, 600], [40, 8000, 720], [35, 5000, 680], [50, 12000, 750]])

# Usando StandardScaler do sklearn
scaler = StandardScaler()

# Ajusta e transforma os dados
X_scaled = scaler.fit_transform(X)

print("-" * 50)
print("Dados padronizados:", X_scaled)
print("Média após padronização (deve ser ~0):", X_scaled.mean())
print("Desvio padrão após padronização (deve ser ~1):", X_scaled.std())

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# Renda mensal (R$)
X_outliers = np.array([3000, 3500, 4000, 4500, 5000, 200_000])

# Scikit-learn espera matriz 2D
X_reshaped = X_outliers.reshape(-1, 1)

# Aplicando StandardScaler (sensível a outliers)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X_reshaped)

# Aplicando RobustScaler (robusto a outliers)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_reshaped)

df = pd.DataFrame({
    "original": X_outliers,
    "standard": X_std.flatten(),
    "robust": X_robust.flatten()
})

print(df.head())


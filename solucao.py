# ========================================
# IMPORTS
# ========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# ========================================
# 1) CARREGAR OS DADOS
# ========================================
df = pd.read_parquet("part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet")

# 🔹 Coluna de data
if "transaction_date" in df.columns:
    df["date"] = pd.to_datetime(df["transaction_date"])
elif "reference_date" in df.columns:
    df["date"] = pd.to_datetime(df["reference_date"])
else:
    raise ValueError("Nenhuma coluna de data encontrada")

# ========================================
# 2) FEATURES BÁSICAS DE TEMPO
# ========================================
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# ========================================
# 3) ENCODING DOS IDs
# ========================================
print("Convertendo IDs para números...")
le_store = LabelEncoder()
le_product = LabelEncoder()

df['internal_store_id_encoded'] = le_store.fit_transform(df['internal_store_id'].astype(str))
df['internal_product_id_encoded'] = le_product.fit_transform(df['internal_product_id'].astype(str))

# ========================================
# 4) FEATURES DE SÉRIE TEMPORAL
# ========================================
print("Criando features de séries temporais...")
group_cols = ['internal_store_id_encoded', 'internal_product_id_encoded']

# Médias móveis e desvios
df['rolling_mean_4'] = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(4, min_periods=1).mean())
df['rolling_std_4']  = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(4, min_periods=1).std())
df['rolling_mean_8'] = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(8, min_periods=1).mean())
df['rolling_mean_12'] = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(12, min_periods=1).mean())

df['rolling_min_4'] = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(4, min_periods=1).min())
df['rolling_max_4'] = df.groupby(group_cols)['quantity'].transform(lambda x: x.rolling(4, min_periods=1).max())

# Lags
df['lag_1'] = df.groupby(group_cols)['quantity'].shift(1)
df['lag_2'] = df.groupby(group_cols)['quantity'].shift(2)
df['lag_4'] = df.groupby(group_cols)['quantity'].shift(4)
df['lag_8'] = df.groupby(group_cols)['quantity'].shift(8)
df['lag_12'] = df.groupby(group_cols)['quantity'].shift(12)

# Relação aceleração
df['lag1_div_mean4'] = df['lag_1'] / (df['rolling_mean_4'] + 1e-5)

# Preencher NaNs
df.fillna(0, inplace=True)

# ========================================
# 5) FEATURES DE SAZONALIDADE (cíclicas)
# ========================================
print("Criando features sazonais...")
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ========================================
# 6) TREINAMENTO LIGHTGBM
# ========================================
print("Treinando modelo LightGBM...")
train_data = df[df['month'].between(6, 11)]  # Junho-Novembro p/ treino

features = [
    'internal_store_id_encoded', 'internal_product_id_encoded',
    'week', 'month', 'day_of_week', 'is_weekend',
    'rolling_mean_4', 'rolling_std_4', 'rolling_mean_8', 'rolling_mean_12',
    'rolling_min_4', 'rolling_max_4',
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_12',
    'lag1_div_mean4',
    'week_sin', 'week_cos', 'month_sin', 'month_cos'
]
X = train_data[features]
y = train_data['quantity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# ========================================
# 7) VALIDAÇÃO INTERNA
# ========================================
print("Fazendo validação interna...")
val_data = df[df['month'] == 12].copy()
X_val_dec = val_data[features]
val_data['forecast_lgb'] = model.predict(X_val_dec)

def wmape(y_true, y_pred):
    return (abs(y_true - y_pred).sum() / y_true.sum()) * 100

score_lgb = wmape(val_data['quantity'], val_data['forecast_lgb'])
print(f"WMAPE LightGBM (dezembro): {round(score_lgb, 2)}%")

# ========================================
# 8) OTIMIZAÇÃO GLOBAL DE PESO (LGBM x Baseline)
# ========================================
print("Otimização global de peso (dezembro)...")

# baseline de dezembro (média recente por par)
weekly_sales_full = (
    df.groupby(["internal_store_id", "internal_product_id", "week"])["quantity"]
    .sum()
    .reset_index()
)
ultima_semana_2022 = weekly_sales_full['week'].max()
vendas_recentes_full = weekly_sales_full[weekly_sales_full['week'] > (ultima_semana_2022 - 8)]
media_recente_full = vendas_recentes_full.groupby(["internal_store_id", "internal_product_id"])["quantity"].mean()
media_global_full = weekly_sales_full.groupby(["internal_store_id", "internal_product_id"])["quantity"].mean()
baseline_map = media_recente_full.fillna(media_global_full).reset_index().rename(columns={"quantity": "forecast_baseline"})

val_data = val_data.merge(baseline_map, on=["internal_store_id", "internal_product_id"], how="left")
val_data["forecast_baseline"] = val_data["forecast_baseline"].fillna(0)

best_w, best_score = 0, np.inf
for w in np.linspace(0, 1, 11):  # testa pesos de 0.0 a 1.0
    preds = w * val_data["forecast_lgb"] + (1 - w) * val_data["forecast_baseline"]
    score_tmp = wmape(val_data["quantity"], preds)
    if score_tmp < best_score:
        best_score = score_tmp
        best_w = w

print(f"Melhor peso global encontrado: {best_w:.2f} (WMAPE otimizado: {best_score:.2f}%)")

# ========================================
# 9) PREVISÃO FINAL PARA JANEIRO/2023
# ========================================
print("Preparando previsão final...")

# Combinações que tiveram venda
pairs_com_venda = df.groupby(['internal_store_id', 'internal_product_id'])['quantity'].sum().reset_index()
pairs_com_venda = pairs_com_venda[pairs_com_venda['quantity'] > 0][['internal_store_id', 'internal_product_id']]

baseline_final = baseline_map.merge(pairs_com_venda, on=['internal_store_id', 'internal_product_id'], how='inner')

# Codificar IDs
baseline_final['internal_store_id_encoded'] = le_store.transform(baseline_final['internal_store_id'].astype(str))
baseline_final['internal_product_id_encoded'] = le_product.transform(baseline_final['internal_product_id'].astype(str))

# Features default para previsão
baseline_final['week'] = 1
baseline_final['month'] = 1
baseline_final['day_of_week'] = 0
baseline_final['is_weekend'] = 0
baseline_final['rolling_mean_4'] = baseline_final['forecast_baseline']
baseline_final['rolling_std_4'] = 0
baseline_final['rolling_mean_8'] = baseline_final['forecast_baseline']
baseline_final['rolling_mean_12'] = baseline_final['forecast_baseline']
baseline_final['rolling_min_4'] = baseline_final['forecast_baseline']
baseline_final['rolling_max_4'] = baseline_final['forecast_baseline']
baseline_final['lag_1'] = baseline_final['forecast_baseline']
baseline_final['lag_2'] = baseline_final['forecast_baseline']
baseline_final['lag_4'] = baseline_final['forecast_baseline']
baseline_final['lag_8'] = baseline_final['forecast_baseline']
baseline_final['lag_12'] = baseline_final['forecast_baseline']
baseline_final['lag1_div_mean4'] = 1.0
baseline_final['week_sin'] = np.sin(2 * np.pi * baseline_final['week'] / 52)
baseline_final['week_cos'] = np.cos(2 * np.pi * baseline_final['week'] / 52)
baseline_final['month_sin'] = np.sin(2 * np.pi * baseline_final['month'] / 12)
baseline_final['month_cos'] = np.cos(2 * np.pi * baseline_final['month'] / 12)

# Previsão LGBM
X_pred = baseline_final[features]
baseline_final['forecast_lgb'] = model.predict(X_pred)

# Ensemble com peso global ótimo
baseline_final['forecast'] = (
    baseline_final['forecast_lgb'] * best_w +
    baseline_final['forecast_baseline'] * (1 - best_w)
)

# Criar previsão 5 semanas
final_forecast = []
for week in range(1, 6):
    temp = baseline_final.copy()
    temp["semana"] = week
    final_forecast.append(temp)
final_forecast = pd.concat(final_forecast)

# Formatando saída
final_forecast = final_forecast.rename(columns={
    "internal_store_id": "pdv",
    "internal_product_id": "produto",
    "forecast": "quantidade"
})
final_forecast["quantidade"] = final_forecast["quantidade"].round().clip(lower=0).astype(int)
final_forecast = final_forecast[["semana", "pdv", "produto", "quantidade"]]

# Limite de linhas
if len(final_forecast) > 1500000:
    print("Aplicando filtro para limitar a 1.500.000 linhas...")
    final_forecast = final_forecast[final_forecast['quantidade'] > 0]
    if len(final_forecast) > 1500000:
        final_forecast = final_forecast.head(1500000)

# Salvar
out_path = "forecast_lgb_global_weight.parquet"
final_forecast.to_parquet(out_path, index=False)

file_size = os.path.getsize(out_path) / (1024 * 1024)
print("\n" + "="*50)
print("✅ PREVISÃO FINAL GERADA COM SUCESSO!")
print("="*50)
print(f"Arquivo: {out_path}")
print(f"Linhas: {len(final_forecast):,}")
print(f"Tamanho: {file_size:.2f} MB")
print(f"WMAPE LGBM (dez): {round(score_lgb, 2)}%")
print(f"WMAPE Ensemble Global (dez): {round(best_score, 2)}%")
print("\nPrimeiras linhas:")
print(final_forecast.head())

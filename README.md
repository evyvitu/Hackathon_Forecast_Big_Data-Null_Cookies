# 📊 Hackathon 2025 Big Data – Previsão de Vendas no Varejo  

Este repositório contém a solução desenvolvida para o **Hackathon 2025 Big Data**, com foco em **previsão de vendas (forecast)** no varejo.  
O objetivo é apoiar a **gestão de estoque e reposição de produtos**, prevendo a quantidade semanal de vendas por **PDV (Ponto de Venda)** e **SKU (Stock Keeping Unit / Unidade de Manutenção de Estoque)** para as **5 semanas de janeiro/2023**, utilizando como base o **histórico de vendas de 2022**.  

---

## 🚀 Objetivo do Projeto
- Prever a quantidade de vendas semanais por **PDV/SKU**.
- Reduzir perdas com falta ou excesso de estoque.
- Apoiar decisões estratégicas de **supply chain**.
- Aplicar técnicas de **Machine Learning (LightGBM)** para melhorar a acurácia.

---

## 🛠️ Estrutura do Código
O código foi dividido em **etapas principais** para garantir clareza e organização:

### 1️⃣ Importação das bibliotecas
```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
```
Utilizamos pandas e numpy para manipulação de dados, LightGBM para modelagem, além de ferramentas do scikit-learn para pré-processamento.

2️⃣ Carregamento dos dados
```python
df = pd.read_parquet("dados.parquet")
Lemos os dados em formato Parquet.
```
Ajustamos a coluna de data (transaction_date ou reference_date).

3️⃣ Engenharia de Features
Criamos novas variáveis temporais e estatísticas:

Semana, mês, ano, dia da semana, indicador de fim de semana.

Rolling mean / std (médias móveis).

Lags (deslocamentos históricos).

4️⃣ Codificação de IDs
```python
le_store = LabelEncoder()
le_product = LabelEncoder()
Os IDs de lojas e produtos são convertidos para valores numéricos, compatíveis com o LightGBM.
```

5️⃣ Treinamento do Modelo
```python
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
```
LightGBM escolhido pela eficiência e boa performance em grandes volumes de dados.

Divisão em treino (junho–novembro) e validação (dezembro).

6️⃣ Validação Interna
Métrica utilizada: WMAPE (Weighted Mean Absolute Percentage Error).

```python
def wmape(y_true, y_pred):
    return (abs(y_true - y_pred).sum() / y_true.sum()) * 100
```
Mede o erro percentual ponderado em relação às vendas reais.

Permite avaliar a qualidade do modelo.


7️⃣ Previsão Final (Janeiro/2023)
Construção de baseline pelas médias históricas.

Previsão com LightGBM.

Ensemble final: 70% LightGBM + 30% Baseline.

Resultado final salvo em Parquet (forecast_lgb_optimized.parquet).

---

📂 Saída do Modelo
O arquivo final contém as previsões no seguinte formato:

| semana | pdv | produto | quantidade |
| ------ | --- | ------- | ---------- |
| 1      | 123 | A001    | 42         |
| 1      | 456 | B002    | 87         |
| ...    | ... | ...     | ...        |

---

⚙️ Instruções de Execução
🔹 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

🔹 2. Criar ambiente virtual (opcional, recomendado)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

🔹 3. Instalar dependências
```bash
pip install -r requirements.txt
```
O arquivo requirements.txt deve conter:

pandas
numpy
lightgbm
scikit-learn
pyarrow
fastparquet

🔹 4. Executar o script
```bash
python main.py
```

🔹 5. Resultado
O arquivo final será salvo como:
forecast_lgb_optimized.parquet

Contendo até 1.500.000 linhas de previsões.

---

📈 Métricas
Métrica principal: WMAPE.

Modelo ajustado para prever vendas com baixo erro percentual.

---

🏆 Conclusão
Este projeto demonstra a aplicação de Big Data e Machine Learning no varejo, trazendo previsões precisas para auxiliar na gestão de estoque.
A solução pode ser expandida para:

Previsões em períodos mais longos.

Inclusão de variáveis externas (promoções, feriados, clima).

Ajuste fino de hiperparâmetros com técnicas de otimização.

--- 

👨‍💻 Equipe
Projeto desenvolvido para o Hackathon 2025 Big Data.

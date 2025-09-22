🏆 Hackathon 2025 Big Data – Previsão de Vendas no Varejo

Este projeto foi desenvolvido para o Hackathon Big Data 2025, com o desafio de criar um modelo preditivo de forecast de vendas para apoiar o varejo na reposição de produtos.

A tarefa consiste em prever a quantidade semanal de vendas por Ponto de Venda (PDV) e SKU (Stock Keeping Unit) nas cinco semanas de janeiro/2023, utilizando como base o histórico de vendas de 2022.

O modelo foi implementado em Python, utilizando LightGBM e técnicas de engenharia de features temporais, além de um ensemble com baseline estatístico.

🚀 Tecnologias utilizadas

Python 3.9+

Pandas / NumPy → manipulação e análise de dados

LightGBM → modelo de Machine Learning

Scikit-learn → pré-processamento e validação

Parquet → formato de saída do forecast

📂 Estrutura do Código

O script está dividido em etapas bem definidas:

1. Carregamento dos Dados
df = pd.read_parquet("arquivo.parquet")


Leitura do dataset em formato Parquet.

Verificação automática da coluna de data (transaction_date ou reference_date).

2. Criação de Features Temporais

Semana, mês, ano.

Dia da semana e flag de final de semana (is_weekend).

3. Codificação de IDs
LabelEncoder() → internal_store_id / internal_product_id


Necessário porque o LightGBM não aceita strings diretamente como identificadores.

4. Engenharia de Atributos Avançada

Médias móveis (rolling_mean_4, rolling_std_4).

Lags de vendas (lag_1, lag_2).

Preenchimento de valores nulos.

5. Treinamento do Modelo LightGBM

Dados de junho a novembro de 2022 usados no treino.

Divisão treino/validação interna.

Treinamento com LGBMRegressor.

6. Validação Interna (Dezembro/2022)

Previsão para dezembro.

Cálculo do WMAPE (Weighted Mean Absolute Percentage Error).

7. Previsão Final para Janeiro/2023

Construção de baseline estatístico (média recente vs média global).

Previsão com LightGBM.

Ensemble (70% LightGBM + 30% Baseline) para robustez.

Geração de previsões para 5 semanas de janeiro/2023.

8. Exportação do Resultado

Garantia de no máximo 1.500.000 linhas.

Exportação em formato Parquet (forecast_lgb_optimized.parquet).

Impressão de estatísticas finais.

📊 Métrica de Avaliação

Foi utilizada a métrica WMAPE (Weighted Mean Absolute Percentage Error):

⚙️ Como Executar
1. Clone este repositório
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo

2. Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Instale as dependências
pip install -r requirements.txt

4. Coloque o dataset de entrada

Adicione o arquivo .parquet com os dados de 2022 na raiz do projeto.

O script automaticamente detecta a coluna de data correta.

5. Execute o script
python forecast.py

6. Saída esperada

Arquivo final: forecast_lgb_optimized.parquet.

Contendo as colunas:

semana → semana de previsão (1 a 5 de janeiro/2023).

pdv → identificador do ponto de venda.

produto → identificador do produto (SKU).

quantidade → previsão de vendas.

✅ Exemplo de Saída
semana	pdv	produto	quantidade
1	102	5555	34
1	103	7210	12
2	102	5555	29
📌 Observações Importantes

O modelo combina aprendizado estatístico + machine learning.

O ensemble foi escolhido para maior estabilidade nas previsões.

O resultado final foi otimizado para o formato exigido pelo hackathon.

👨‍💻 Equipe

Projeto desenvolvido durante o Hackathon Big Data 2025.
Contribuições são bem-vindas via Pull Requests.

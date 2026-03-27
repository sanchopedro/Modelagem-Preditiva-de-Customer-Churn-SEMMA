# Análise de Churn em Telecomunicações usando Framework SEMMA

## Introdução

Este projeto tem como objetivo identificar padrões e desenvolver modelos preditivos para o churn de clientes em uma empresa de telecomunicações. Utilizamos o framework SEMMA (Sample, Explore, Modify, Model, Assess) para estruturar a análise de dados e modelagem preditiva.

O churn de clientes é um problema crítico para empresas de telecomunicações, pois a retenção de clientes é mais econômica do que a aquisição de novos. Este projeto analisa dados de clientes para prever quais têm maior probabilidade de cancelar seus serviços.

## Requerimentos para Rodar o Código

Para executar o código deste projeto, você precisa de:

- **Python 3.8 ou superior**
- **Bibliotecas Python** listadas no arquivo `requirements.txt`:
  - pandas==2.3.3
  - plotly==6.5.0
  - nbformat==5.10.4
  - scikit-learn==1.8.0
  - tqdm==4.67.1
  - xgboost==3.1.3

### Instalação

1. Clone ou baixe o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script principal:
   ```bash
   python tcc.py
   ```
   Ou abra o notebook `tcc.ipynb` no Jupyter Notebook.

## Etapas do Projeto

O projeto segue o framework SEMMA:

1. **Sample (Amostragem)**: Carregamento e coleta dos dados do dataset.
2. **Explore (Exploração)**: Análise exploratória dos dados, incluindo estatísticas descritivas e visualizações.
3. **Modify (Modificação)**: Preparação dos dados, tratamento de valores ausentes e transformação de variáveis.
4. **Model (Modelagem)**: Desenvolvimento e treinamento de modelos preditivos de churn.
5. **Assess (Avaliação)**: Avaliação dos modelos e interpretação dos resultados.

## Ferramentas

- **Linguagem**: Python
- **Bibliotecas de Análise de Dados**: pandas, numpy
- **Visualização**: plotly, matplotlib
- **Machine Learning**: scikit-learn, xgboost

## Conjunto de Dados

O dataset utilizado é o **Telco Customer Churn** disponível no Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Ele contém informações sobre clientes de uma empresa de telecomunicações fictícia.

### Descrição das Variáveis

- **customerID**: Identificador único do cliente
- **gender**: Gênero (Male/Female)
- **SeniorCitizen**: Se é idoso (0/1)
- **Partner**: Tem parceiro (Yes/No)
- **Dependents**: Tem dependentes (Yes/No)
- **tenure**: Meses de permanência
- **PhoneService**: Serviço de telefone (Yes/No)
- **MultipleLines**: Múltiplas linhas (Yes/No/No phone service)
- **InternetService**: Tipo de internet (DSL/Fiber optic/No)
- **OnlineSecurity**: Segurança online (Yes/No/No internet service)
- **OnlineBackup**: Backup online (Yes/No/No internet service)
- **DeviceProtection**: Proteção de dispositivo (Yes/No/No internet service)
- **TechSupport**: Suporte técnico (Yes/No/No internet service)
- **StreamingTV**: Streaming de TV (Yes/No/No internet service)
- **StreamingMovies**: Streaming de filmes (Yes/No/No internet service)
- **Contract**: Tipo de contrato (Month-to-month/One year/Two year)
- **PaperlessBilling**: Fatura digital (Yes/No)
- **PaymentMethod**: Método de pagamento
- **MonthlyCharges**: Cobrança mensal
- **TotalCharges**: Cobrança total
- **Churn**: Variável target (Yes/No)

O dataset possui aproximadamente 7.000 registros.

## Fluxo do Projeto

1. **Carregamento dos Dados**: Leitura do arquivo CSV e verificação inicial.
2. **Tratamento de Dados**: Conversão de tipos, tratamento de valores ausentes.
3. **Análise Exploratória**: Visualizações e estatísticas para entender os dados.
4. **Pré-processamento**: Codificação de variáveis categóricas, normalização.
5. **Modelagem**: Treinamento de modelos como Regressão Logística, Árvore de Decisão, Random Forest, SVM e XGBoost.
6. **Avaliação**: Comparação de métricas como acurácia, precisão, recall e AUC-ROC.
7. **Salvamento de Modelos**: Modelos treinados são salvos na pasta `models/`.


## Resultados Detalhados (Etapa: Assess)

Após a aplicação do framework **SEMMA** e a otimização de hiperparâmetros via **Grid Search**, o algoritmo **XGBoost** consolidou-se como a melhor solução para a predição de *churn*, superando os demais classificadores testados em todas as métricas de negócio.

### Performance do Modelo
O modelo demonstrou alta capacidade de discriminação entre clientes ativos e potenciais evasões. Abaixo, as principais métricas obtidas:

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **ROC-AUC** | **0,8395** | Excelente capacidade de distinção entre as classes. |
| **Acurácia Global** | **79,10%** | Percentual total de acertos do modelo. |
| **Precisão (Churn)** | **0,6325** | De cada 100 alertas de churn, 63 realmente cancelariam. |
| **Recall (Churn)** | **0,5107** | Identifica precocemente 51% de todas as evasões reais. |

### Análise da Matriz de Confusão
A avaliação diagnóstica revelou um comportamento estratégico e conservador do classificador:

* **Foco em Retenção Passiva:** O modelo apresenta um **Recall de 0,8925 para clientes não-evadidos**. Isso indica uma alta assertividade ao identificar clientes que pretendem permanecer, minimizando o "desperdício" de recursos com descontos ou benefícios desnecessários para quem já está fidelizado.
* **Tratamento de Desequilíbrio:** No setor de telecomunicações, a acurácia isolada é insuficiente. O foco em Precision e Recall garante que o modelo seja aplicável mesmo em cenários de classes desbalanceadas.

### Impacto no Negócio
Os resultados permitem uma atuação direta na redução do **CAC (Custo de Aquisição de Clientes)** e na proteção da receita:

1.  **Intervenção Preventiva:** Ao prever metade do prejuízo potencial com antecedência, a empresa pode aplicar campanhas de retenção direcionadas e eficazes.
2.  **Eficiência Operacional:** A alta precisão na identificação de clientes ativos evita custos operacionais desnecessários e protege a margem de lucro da operação.

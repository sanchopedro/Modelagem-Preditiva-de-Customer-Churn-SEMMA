#!/usr/bin/env python
# coding: utf-8

# ### ============================================================================
# # Análise de Churn em Telecomunicações usando Framework SEMMA
# ## TCC - MBA Data Science and Business Analytics - USP Esalq 
# ### Pedro Sancho Rodrigues
# ### Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
# ### ============================================================================
# 
# ## Objetivo
# Identificar padrões e desenvolver modelos preditivos para churn de clientes em uma empresa de telecomunicações.
# 
# ## Framework SEMMA
#  - **S**ample: Amostragem e coleta de dados
#  - **E**xplore: Exploração e análise descritiva
#  - **M**odify: Modificação e preparação dos dados
#  - **M**odel: Modelagem preditiva
#  - **A**ssess: Avaliação dos modelos
# 
#  ---
# 
#  ## Descrição das Variáveis
# 
# ### **Variável Target (Dependente)**
# - **Churn**: Indica se o cliente deixou a empresa no último mês
#   - Valores: "Yes" (cliente cancelou) ou "No" (cliente permaneceu)
# 
# ### **Identificação do Cliente**
# - **customerID**: Identificador único de cada cliente
# 
# ### **Informações Demográficas** (3 variáveis)
# - **gender**: Gênero do cliente (Male/Female)
# - **SeniorCitizen**: Indica se o cliente é idoso (1) ou não (0)
# - **Partner**: Indica se o cliente tem parceiro/cônjuge (Yes/No)
# - **Dependents**: Indica se o cliente tem dependentes (Yes/No)
# 
# ### **Informações da Conta** (4 variáveis)
# - **tenure**: Número de meses que o cliente permaneceu na empresa
# - **Contract**: Tipo de contrato do cliente
#   - "Month-to-month" (mensal)
#   - "One year" (anual)
#   - "Two year" (bienal)
# - **PaperlessBilling**: Indica se o cliente tem fatura eletrônica (Yes/No)
# - **PaymentMethod**: Método de pagamento do cliente
#   - "Electronic check" (cheque eletrônico)
#   - "Mailed check" (cheque postal)
#   - "Bank transfer (automatic)" (transferência bancária automática)
#   - "Credit card (automatic)" (cartão de crédito automático)
# 
# ### **Informações Financeiras** (2 variáveis)
# - **MonthlyCharges**: Valor cobrado mensalmente do cliente (em dólares)
# - **TotalCharges**: Valor total cobrado do cliente (em dólares)
# 
# ### **Serviços Contratados** (9 variáveis)
# 
# **Serviços de Telefonia:**
# - **PhoneService**: Indica se o cliente tem serviço de telefone (Yes/No)
# - **MultipleLines**: Indica se o cliente tem múltiplas linhas telefônicas
#   - "Yes", "No", "No phone service"
# 
# **Serviços de Internet:**
# - **InternetService**: Tipo de serviço de internet do cliente
#   - "DSL", "Fiber optic", "No" (sem internet)
# - **OnlineSecurity**: Indica se o cliente tem serviço de segurança online
#   - "Yes", "No", "No internet service"
# - **OnlineBackup**: Indica se o cliente tem serviço de backup online
#   - "Yes", "No", "No internet service"
# - **DeviceProtection**: Indica se o cliente tem proteção de dispositivo
#   - "Yes", "No", "No internet service"
# - **TechSupport**: Indica se o cliente tem suporte técnico
#   - "Yes", "No", "No internet service"
# 
# **Serviços de Streaming:**
# - **StreamingTV**: Indica se o cliente tem serviço de streaming de TV
#   - "Yes", "No", "No internet service"
# - **StreamingMovies**: Indica se o cliente tem serviço de streaming de filmes
#   - "Yes", "No", "No internet service"
# 

# ## 1. Importação de Bibliotecas

# In[1]:


# ============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
# ============================================================================

# ------------------------------
# Bibliotecas padrão do Python
# ------------------------------
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ------------------------------
# Manipulação e análise de dados
# ------------------------------
import pandas as pd

# ------------------------------
# Visualização de dados
# ------------------------------
import plotly.graph_objects as go
# import plotly.express as px
from plotly.subplots import make_subplots

# ------------------------------
# Machine Learning — utilidades
# ------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer

# ------------------------------
# Machine Learning — pré-processamento
# ------------------------------
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)

# ------------------------------
# Machine Learning — modelos
# ------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ------------------------------
# Machine Learning — métricas
# ------------------------------
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------
# Utilidades
# ------------------------------
from tqdm import tqdm

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# Configurações de visualização do pandas
pd.set_option('display.max_columns', None)

# ------------------------------
# Configuração de cores dos gráficos
# ------------------------------
COLOR_CHURN = '#e74c3c'      # vermelho
COLOR_NO_CHURN = '#2ecc71'   # verde

PIE_COLORS = [
    '#1f77b4', '#8c564b', '#ff7f0e',
    '#9467bd', '#e377c2', '#7f7f7f'
]


# ## 2. SAMPLE - Amostragem e Coleta de Dados

# In[2]:


print("\n" + "="*80)
print("FASE 1: SAMPLE - AMOSTRAGEM E CARREGAMENTO DOS DADOS")
print("="*80)

# Carregar dataset
arquivo = 'Telco-Customer-Churn.csv'
df = pd.read_csv(arquivo)

print(f"\nDimensões do dataset: {df.shape}")
print(f"Total de registros: {df.shape[0]:,}")
print(f"Total de features: {df.shape[1]}")

df.head()


# In[3]:


# %% Informações gerais do dataset
print("\n" + "=" * 80)
print("INFORMAÇÕES DO DATASET")
print("=" * 80)
df.info()


# apenas a coluna 'TotalCharges' deve ter o DType alterado para número. 
# 
# Podemos também transformar a coluna 'SeniorCitizen' para os casos 0 = 'No' e 1 = 'Yes'. Mais para frente vamos transformar as variáveis categóricas para a modelagem

# In[4]:


# Modificando o Dtype da coluna TotalCharges
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

# Alterando a coluna 'SeniorCitizen'
df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})


# In[5]:


# Verificando novamente as colunas
df.info()


# In[6]:


# %% Verificação de valores missing
print("\n" + "=" * 80)
print("VALORES MISSING")
print("=" * 80)

missing_count = df.isnull().sum()

missing_summary = (
    missing_count[missing_count > 0]
    .to_frame(name="Qtd Missing")
    .assign(
        Percentual=lambda x: (
            (x["Qtd Missing"] / len(df) * 100)
            .round(2)
            .astype(str) + "%"
        )
    )
)

if not missing_summary.empty:
    print(missing_summary)
else:
    print("Nenhum valor ausente detectado.")


# In[7]:


# Filtrando os dados que tem 'TotalCharges' nulos
df[df['TotalCharges'].isna()]


# In[8]:


# Verificando se há outros casos onde 'tenure' = 0
df[(df['tenure'] == 0) & (df['TotalCharges'].notna())]


# Ao analisar os dados para os clientes com o 'TotalCharges' nulos, podemos perceber que são os mesmos com a 'Tenure' igual a 0, ou seja, são clientes que não ficaram pelo menos 1 mês de contrato. Podemos perceber também, que não existe outros casos onde 'tenure' seja igual a 0. Dessa forma, sabendo que a parcela de nulos representa menos de 1% dos dados, podemos remover do dataset.

# In[9]:


# Removendo os dados nulos
df = df.dropna()


# In[10]:


# Estatística Descritivas
print("\n" + "=" * 80)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 80)
df.describe()


# ## 3. EXPLORE - Análise Exploratória dos Dados

# In[11]:


# Criando cópia do df original
df_explore = df.copy()


# In[12]:


print("\n" + "="*80)
print("FASE 2: EXPLORE - ANÁLISE EXPLORATÓRIA DOS DADOS")
print("="*80)

# %% Análise da variável target (Churn)
print("\n" + "=" * 80)
print("DISTRIBUIÇÃO DO CHURN")
print("=" * 80)
churn_dist = df_explore['Churn'].value_counts()
churn_pct = df_explore['Churn'].value_counts(normalize=True) * 100

print(f"\nContagem:")
print(churn_dist)
print("\nPercentual:")
churn_pct_formatado = churn_pct.round(2).astype(str) + "%"
print(churn_pct_formatado)


# In[13]:


# Subplots: 1 linha, 2 colunas
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "bar"}, {"type": "pie"}]],
    subplot_titles=["Distribuição de Churn", "Proporção de Churn"]
)

# --- Gráfico de barras ---
fig.add_trace(
    go.Bar(
        x=churn_dist.index,
        y=churn_dist.values,
        text=churn_dist.values,
        textposition="outside",
        marker_color=['#2ecc71', '#e74c3c'],
        name="Quantidade"
    ),
    row=1,
    col=1
)

fig.update_xaxes(title_text="Churn", row=1, col=1)
fig.update_yaxes(title_text="Quantidade", row=1, col=1)

# --- Gráfico de pizza ---
fig.add_trace(
    go.Pie(
        labels=churn_pct.index,
        values=churn_pct.values,
        hole=0.4,  
        textinfo="label+percent",
        textfont=dict(size=12),
        marker_colors=['#2ecc71', '#e74c3c'],
        sort=False
    ),
    row=1,
    col=2
)

# Layout geral
fig.update_layout(
    width=900,
    height=500,
    showlegend=False,
    title_text="Análise de Churn",
    title_x=0.5
)

fig.show()


# In[14]:


def print_churn_stats(df_explore, feature, title):
    print("=" * 80)
    print(title.upper())
    print("=" * 80)

    for value in df_explore[feature].dropna().unique():
        subset = df_explore[df_explore[feature] == value]
        total = len(subset)
        churned = (subset['Churn'] == 'Yes').sum()
        churn_rate = (churned / total) * 100 if total > 0 else 0

        print(f"\n{value}:")
        print(f"  • Total de clientes: {total:,}")
        print(f"  • Clientes com churn: {churned:,}")
        print(f"  • Taxa de churn: {churn_rate:.2f}%")

    print("\n" + "=" * 80)

def plot_feature_churn(
    df_explore,
    feature,
    title,
    pie_title,
    bar_title,
    height=500
):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        subplot_titles=(pie_title, bar_title)
    )

    # -------------------------
    # Pizza — distribuição
    # -------------------------
    counts = df_explore[feature].value_counts()

    fig.add_trace(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            textinfo='percent',
            hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<extra></extra>',
            marker=dict(colors=PIE_COLORS),
            showlegend=True
        ),
        row=1, col=1
    )

    # -------------------------
    # Barras — churn %
    # -------------------------
    churn_pct = (
        df_explore
        .groupby([feature, 'Churn'])
        .size()
        .reset_index(name='count')
    )

    churn_pct['percentual'] = (
        churn_pct['count']
        / churn_pct.groupby(feature)['count'].transform('sum')
        * 100
    ).round(2)

    for churn_value, color, label in [
        ('No', COLOR_NO_CHURN, 'Não Churn'),
        ('Yes', COLOR_CHURN, 'Churn')
    ]:
        data = churn_pct[churn_pct['Churn'] == churn_value]

        fig.add_trace(
            go.Bar(
                x=data[feature],
                y=data['percentual'],
                text=data['percentual'].astype(str) + '%',
                textposition='auto',
                name=label,
                marker_color=color
            ),
            row=1, col=2
        )

    fig.update_layout(
        barmode='group',
        yaxis_title='Percentual (%)',
        title_text=title,
        title_x=0.5,
        height=700,
        width=1200,  # largura em pixels
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    fig.show()

features = [
    ('gender', 'Análise Demográfica', 'Distribuição por Gênero', 'Percentual de Churn por Gênero'),
    ('Contract', 'Análise de Contratos', 'Distribuição por Contrato', 'Percentual de Churn por Contrato'),
    ('PaymentMethod', 'Análise de Métodos de Pagamento', 'Distribuição por Método de Pagamento', 'Percentual de Churn por Método de Pagamento'),
    ('SeniorCitizen', 'Análise de Cidadão Sênior', 'Distribuição por Cidadão Sênior', 'Percentual de Churn por Cidadão Sênior'),
    ('Dependents', 'Análise de Dependentes', 'Distribuição de Dependentes', 'Percentual de Churn por Dependentes'),
    ('Partner', 'Análise de Cônjunges', 'Distribuição de Cônjuges', 'Percentual de Churn por Cônjuges'),
    ('PhoneService', 'Análise de Serviços de Telefone', 'Distribuição de Serviços Telefone', 'Percentual de Churn por Serviço'),
    ('MultipleLines', 'Análise de Múltiplas Linhas', 'Distribuição de Múltiplas Linhas', 'Percentual de Churn por Múltiplas Linhas'),
    ('InternetService', 'Análise de Serviço de Internet', 'Distribuição de Serviço de Internet', 'Percentual de Churn por Serviço de Internet'),
    ('OnlineSecurity', 'Análise de Segurança Online', 'Distribuição de Segurança Online', 'Percentual de Churn por Segurança Online'),
    ('OnlineBackup', 'Análise de Backup Online', 'Distribuição de Backup Online', 'Percentual de Churn por Backup Online'),
    ('DeviceProtection', 'Análise de Proteção de Dispositivo', 'Distribuição de Proteção de Dispositivo', 'Percentual de Churn por Proteção de Dispositivo'),
    ('TechSupport', 'Análise de Suporte Técnico', 'Distribuição de Suporte Técnico', 'Percentual de Churn por Suporte Técnico'),
    ('StreamingTV', 'Análise de Streaming de TV', 'Distribuição de Streaming de TV', 'Percentual de Churn por Streaming de TV'),
    ('StreamingMovies', 'Análise de Streaming de Filmes', 'Distribuição de Streaming de Filmes', 'Percentual de Churn por Streaming de Filmes'),
    ('PaperlessBilling', 'Análise de Fatura Digital', 'Distribuição de Fatura Digital', 'Percentual de Churn por Fatura Digital')
]

for feature, title, pie_title, bar_title in features:
    print_churn_stats(df_explore, feature, f"Estatísticas de {feature} e Churn")
    plot_feature_churn(
        df_explore,
        feature,
        title,
        pie_title,
        bar_title
    )


# In[15]:


print("=" * 80)
print("ESTATÍSTICAS DE TENURE E CHURN")
print("=" * 80)

# Garantir que o tenure_group exista
df_explore['tenure_group'] = pd.cut(
    df_explore['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12 meses', '13-24 meses', '25-48 meses', '49-72 meses']
)

for group in df_explore['tenure_group'].dropna().unique():
    total = len(df_explore[df_explore['tenure_group'] == group])
    churned = (df_explore[df_explore['tenure_group'] == group]['Churn'] == 'Yes').sum()
    churn_rate = (churned / total) * 100 if total > 0 else 0

    print(f"\n{group}:")
    print(f"  • Total de clientes: {total:,}")
    print(f"  • Clientes com churn: {churned:,}")
    print(f"  • Taxa de churn: {churn_rate:.2f}%")

print("\n" + "=" * 80)

tenure_churn = (
    df_explore
    .groupby(['tenure_group', 'Churn'])
    .size()
    .unstack(fill_value=0)
)

tenure_churn_pct = (
    tenure_churn
    .div(tenure_churn.sum(axis=1), axis=0)
    * 100
).round(2)

# ============================================================================
# ANÁLISE DE TENURE E CHURN
# ============================================================================

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{'type': 'histogram'}, {'type': 'bar'}]],
    subplot_titles=(
        'Distribuição do Tempo de Permanência (Tenure)',
        'Taxa de Churn por Grupo de Tenure (%)'
    )
)

# ============================================================================
# SUBPLOT 1 — Histograma de Tenure
# ============================================================================

fig.add_trace(
    go.Histogram(
        x=df_explore['tenure'],
        nbinsx=72,
        marker_color='#3498db',
        showlegend=False
    ),
    row=1, col=1
)

# ============================================================================
# SUBPLOT 2 — Churn por grupo de tenure
# ============================================================================

fig.add_trace(
    go.Bar(
        name='Não Churn',
        x=tenure_churn_pct.index.astype(str),
        y=tenure_churn_pct['No'],
        text=tenure_churn_pct['No'].astype(str) + '%',
        textposition='inside',
        marker_color=COLOR_NO_CHURN
    ),
    row=1, col=2
)

fig.add_trace(
    go.Bar(
        name='Churn',
        x=tenure_churn_pct.index.astype(str),
        y=tenure_churn_pct['Yes'],
        text=tenure_churn_pct['Yes'].astype(str) + '%',
        textposition='inside',
        marker_color=COLOR_CHURN
    ),
    row=1, col=2
)


# ============================================================================
# LAYOUT E CONFIGURAÇÕES
# ============================================================================

fig.update_layout(
    barmode='group',
    height=500,
    title_text='Análise de Tenure e Churn',
    title_x=0.5,
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.25,
        xanchor='center',
        x=0.5
    )
)

fig.update_xaxes(title_text='Tenure (meses)', row=1, col=1)
fig.update_yaxes(title_text='Quantidade de Clientes', row=1, col=1)

fig.update_xaxes(title_text='Grupo de Tenure', row=1, col=2)
fig.update_yaxes(title_text='Percentual (%)', row=1, col=2)

fig.show()



# In[16]:


fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        'Monthly Charges vs Churn',
        'Total Charges vs Churn',
        'Monthly Charges vs Total Charges',
        'Taxa de Churn por Faixa de Monthly Charges'
    ),
    specs=[
        [{'type': 'box'}, {'type': 'box'}],
        [{'type': 'scatter'}, {'type': 'bar'}]
    ]
)

# -------------------------
# 1. Boxplot — MonthlyCharges × Churn
# -------------------------

for churn_value, color in [('No', COLOR_NO_CHURN), ('Yes', COLOR_CHURN)]:
    fig.add_trace(
        go.Box(
            x=[churn_value] * len(df_explore[df_explore['Churn'] == churn_value]),
            y=df_explore[df_explore['Churn'] == churn_value]['MonthlyCharges'],
            name=churn_value,
            boxmean=True,
            marker_color=color,
            showlegend=False
        ),
        row=1, col=1
    )



# -------------------------
# 2. Boxplot — TotalCharges × Churn
# -------------------------

for churn_value, color in [('No', COLOR_NO_CHURN), ('Yes', COLOR_CHURN)]:
    fig.add_trace(
        go.Box(
            x=[churn_value] * len(df_explore[df_explore['Churn'] == churn_value]),
            y=df_explore[df_explore['Churn'] == churn_value]['TotalCharges'],
            name=churn_value,
            boxmean=True,
            marker_color=color,
            showlegend=False  # evita legenda duplicada
        ),
        row=1, col=2
    )

# -------------------------
# 3. Scatter — MonthlyCharges × TotalCharges
# -------------------------

for churn_value, color in [('No', COLOR_NO_CHURN), ('Yes', COLOR_CHURN)]:
    subset = df_explore[df_explore['Churn'] == churn_value]

    fig.add_trace(
        go.Scatter(
            x=subset['MonthlyCharges'],
            y=subset['TotalCharges'],
            mode='markers',
            name=f'Churn: {churn_value}',
            marker=dict(color=color, size=6),
            opacity=0.6,
            showlegend=True
        ),
        row=2, col=1
    )


# -------------------------
# 4. Bar — taxa de churn por faixa de MonthlyCharges
# -------------------------

df_explore['monthly_bin'] = pd.qcut(df_explore['MonthlyCharges'], q=5)

churn_rate = (
    df_explore
    .groupby('monthly_bin')['Churn']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .reset_index(name='churn_rate')
)

fig.add_trace(
    go.Bar(
        x=churn_rate['monthly_bin'].astype(str),
        y=churn_rate['churn_rate'],
        text=churn_rate['churn_rate'].round(2).astype(str) + '%',
        textposition='auto',
        marker_color=COLOR_CHURN,
        name='Churn Rate',
        showlegend=False
    ),
    row=2, col=2
)


# -------------------------
# Layout e configurações
# -------------------------

fig.update_layout(
    height=900,
    width=1200,
    # title_text='Análise Quantitativa: Charges × Churn',
    title_x=0.5,
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.15,
        xanchor='center',
        x=0.5
    )
)

fig.update_yaxes(title_text='Monthly Charges', row=1, col=1)
fig.update_yaxes(title_text='Total Charges', row=1, col=2)
fig.update_xaxes(title_text='Monthly Charges', row=2, col=1)
fig.update_yaxes(title_text='Total Charges', row=2, col=1)
fig.update_xaxes(title_text='Faixa de Monthly Charges', row=2, col=2)
fig.update_yaxes(title_text='Churn (%)', row=2, col=2)

fig.show()


# ### Análise Quantitativa: Charges, Tenure e Churn
# 
# A análise das variáveis financeiras evidencia uma relação consistente entre Monthly Charges e o churn. Observa-se que clientes que cancelaram o serviço apresentam, em média, mensalidades mais elevadas, enquanto clientes que permaneceram concentram-se em faixas de preço mais baixas. Além disso, a taxa de churn cresce progressivamente conforme o aumento do valor mensal, indicando que preços mais altos estão associados a maior risco de cancelamento.
# 
# Em contrapartida, clientes que não churnaram acumulam valores significativamente maiores de Total Charges, o que sugere maior tempo de permanência na base. Clientes churnados tendem a apresentar valores totais mais baixos, indicando que o cancelamento ocorre, em muitos casos, de forma precoce no ciclo de vida do cliente. O gráfico de dispersão reforça esse comportamento ao mostrar que clientes com mensalidades elevadas nem sempre geram maior valor ao longo do tempo, pois parte significativa cancela antes de consolidar esse faturamento.
# 
# A análise do tenure demonstra que o churn está fortemente concentrado nos primeiros meses de relacionamento. Clientes com até 12 meses apresentam as maiores taxas de cancelamento, enquanto a probabilidade de churn diminui progressivamente à medida que o tempo de permanência aumenta. Clientes com maior tenure, especialmente acima de quatro anos, apresentam churn residual, indicando elevada estabilidade.
# 
# De forma geral, os resultados indicam que preço e tempo de permanência são fatores fortemente associados ao churn, destacando a importância de estratégias de retenção focadas nos primeiros meses de contrato e em clientes com mensalidades mais elevadas.

# ## 4. MODIFY - Modificação e Preparação dos Dados

# In[17]:


# Criação de uma cópia para manipulação
df_processed = df.copy()


# In[18]:


df_processed.head()


# In[19]:


print("=" * 80)
print("FASE 3: MODIFY - PREPARAÇÃO DOS DADOS")
print("=" * 80)

# %% Codificação da variável target
print("\n   - Transformando a variável target (Churn) No = 0 e Yes = 1")
df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})

# Criando novas Features:

# Média de cobrança mensal
print('\n   - Criando a variável avg_charges')
df_processed['avg_charges'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)

# Total de serviços contratados
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

print('\n   - Criando a variável total_services')
df_processed['total_services'] = 0
for col in service_cols:
    if col in df_processed.columns:
        df_processed['total_services'] += (df_processed[col].isin(['Yes', 'DSL', 'Fiber optic'])).astype(int)

# Remover customerID
print('\n   - Removendo a variável customerID')
df_processed.drop('customerID', axis=1, inplace=True)

# Separação de features e target (ANTES de separar colunas)
target_column = "Churn"
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

# Agora separar os tipos de colunas (só nas features X)
unique_counts = X.select_dtypes("O").nunique()
binary_columns = unique_counts[unique_counts == 2].index.tolist()
categorical_columns = unique_counts[unique_counts > 2].index.tolist()

print(f"\n   - Total de Colunas Binárias: {len(binary_columns)} --> {binary_columns}")
print(f"\n   - Total de Colunas Categóricas: {len(categorical_columns)} --> {categorical_columns}")
print(f"\n   - Coluna Target: {target_column}")


print(f"\nSeparação de Features e Target")

print(f"    - Features (X): {X.shape}")
print(f"    - Target (y): {y.shape}")
print(f"    - Balanceamento do target: {y.value_counts().to_dict()}")


# In[20]:


# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Split treino/teste (80/20)\n")

print(f"    - Treino: {X_train.shape[0]} registros")
print(f"    - Teste: {X_test.shape[0]} registros")


# In[21]:


print(f"Transformando Features\n")

transformer = ColumnTransformer(
    [
        ("scaler", StandardScaler(), ["MonthlyCharges", "TotalCharges", "tenure", "avg_charges", "total_services"]),
        ("binary_encoder", OrdinalEncoder(), binary_columns),
        ("ohe", OneHotEncoder(drop="first"), categorical_columns),
    ],
    remainder="passthrough",
)

transformer.fit(X_train)
columns = transformer.get_feature_names_out()
columns = list(map(lambda x: str(x).split("__")[-1], columns))

X_train = pd.DataFrame(transformer.transform(X_train), columns=columns)
X_test = pd.DataFrame(transformer.transform(X_test), columns=columns)

print(f"    - X_train: {X_train.shape}")
print(f"    - X_test: {X_test.shape}")


# ## 5. MODEL - Modelagem Preditiva

# In[22]:


# %% Configuração de diretórios
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)


# In[23]:


# %% Treinamento de múltiplos modelos

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    'SVM': SVC(random_state=42, probability=True)
}


# In[24]:


# Funções auxiliares

# Salvar modelo machine learning formato .pickle
def save_model(model, model_name, directory='models'):
    """
    Salva o modelo em formato pickle
    """
    filepath = Path(directory) / f"{model_name.replace(' ', '_')}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  - Modelo salvo em: {filepath}")

# Ler o modelo machine learning já criado
def load_model(model_name, directory='models'):
    """
    Carrega o modelo salvo em pickle
    """
    filepath = Path(directory) / f"{model_name.replace(' ', '_')}.pkl"
    if filepath.exists():
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"  - Modelo carregado de: {filepath}")
        return model
    return None

# Avaliar modelos
def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Função para avaliar modelo com múltiplas métricas
    """
    all_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    return all_metrics


# In[25]:


print("\n" + "=" * 80)
print("FASE 5: MODEL - TREINAMENTO DE MODELOS")
print("=" * 80)

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Modelo: {name}")
    print(f"{'='*60}")

    model_filename = name.replace(' ', '_')
    model_path = MODELS_DIR / f"{model_filename}.pkl"

    # Verificar se o modelo já existe
    if model_path.exists():
        # Carregar modelo existente
        print(f"  - Carregando modelo salvo...")
        model = load_model(name, MODELS_DIR)
    else:
        # Treinar e salvar novo modelo
        print(f"  - Modelo não encontrado. Treinando...")
        model.fit(X_train, y_train)
        save_model(model, name, MODELS_DIR)

    # Fazer predições
    print(f"  Gerando predições...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Avaliar modelo
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    # Armazenar resultados
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,

        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
    }

    # Exibir métricas
    print(f"\n Métricas de Avaliação:")
    print(f"     • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     • Precision: {metrics['precision']:.4f}")
    print(f"     • Recall:    {metrics['recall']:.4f}")
    print(f"     • F1-Score:  {metrics['f1']:.4f}")
    print(f"     • ROC-AUC:   {metrics['roc_auc']:.4f}")


# In[26]:


print("\n" + "=" * 80)
print("TABELA DE RESULTADOS - COMPARAÇÃO DE MODELOS")
print("=" * 80)

# Criar DataFrame com os resultados
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],

})

# Ordenar por ROC-AUC (decrescente)
results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

# Formatar valores com 4 casas decimais
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")

# print("\n", results_df.to_string(index=False))
display(results_df)

# Identificar melhor modelo
print("\n" + "=" * 80)
print(f"🏆 MELHOR MODELO PADRÃO (ROC-AUC): {results_df.iloc[0]['Modelo']}")
print("=" * 80)


# In[27]:


# Definir os grids de hiperparâmetros para cada modelo
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

# Armazenar os melhores modelos
best_models = {}
best_params = {}
grid_results = {}


# In[28]:


for name in models.keys():
    print(f"\n{'='*60}")
    print(f"Grid Search - Otimização de Hiperparâmetros: {name}")
    print(f"{'='*60}")

    # Verificar se já existe modelo otimizado salvo
    optimized_model_path = MODELS_DIR / f"{name.replace(' ', '_')}_optimized.pkl"

    if optimized_model_path.exists():
        print(f"  - Carregando modelo otimizado salvo...")
        best_model = load_model(f"{name}_optimized", MODELS_DIR)

        # Carregar parâmetros salvos
        params_path = MODELS_DIR / f"{name.replace(' ', '_')}_best_params.pkl"
        with open(params_path, 'rb') as f:
            best_params[name] = pickle.load(f)

    else:
        print(f"  - Executando Grid Search...")

        # Criar o modelo base
        base_model = models[name]

        # Configurar Grid Search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            cv=5,  
            scoring='roc_auc',  
            n_jobs=-1, 
            verbose=0
        )

        # Executar Grid Search
        grid_search.fit(X_train, y_train)

        # Melhor modelo
        best_model = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_

        # Salvar modelo otimizado
        save_model(best_model, f"{name}_optimized", MODELS_DIR)

        # Salvar melhores parâmetros
        params_path = MODELS_DIR / f"{name.replace(' ', '_')}_best_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(best_params[name], f)
        print(f"  - Parâmetros salvos em: {params_path}")

    # Fazer predições com o melhor modelo
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Avaliar
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    # Armazenar resultados
    best_models[name] = best_model
    grid_results[name] = {
        'model': best_model,
        'best_params': best_params[name],
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc']
    }

    # Exibir resultados
    print(f"\n  - Melhores Parâmetros:")
    for param, value in best_params[name].items():
        print(f"     • {param}: {value}")

    print(f"\n  - Métricas com Parâmetros Otimizados:")
    print(f"     • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     • Precision: {metrics['precision']:.4f}")
    print(f"     • Recall:    {metrics['recall']:.4f}")
    print(f"     • F1-Score:  {metrics['f1']:.4f}")
    print(f"     • ROC-AUC:   {metrics['roc_auc']:.4f}")


# In[29]:


print("\n" + "=" * 80)
print("TABELA DE RESULTADOS OTIMIZADOS - COMPARAÇÃO DE MODELOS")
print("=" * 80)

# Criar DataFrame com os resultados
results_grid_df = pd.DataFrame({
    'Modelo': list(grid_results.keys()),
    'Accuracy': [grid_results[m]['accuracy'] for m in grid_results.keys()],
    'Precision': [grid_results[m]['precision'] for m in grid_results.keys()],
    'Recall': [grid_results[m]['recall'] for m in grid_results.keys()],
    'F1-Score': [grid_results[m]['f1'] for m in grid_results.keys()],
    'ROC-AUC': [grid_results[m]['roc_auc'] for m in grid_results.keys()],

})

# Ordenar por ROC-AUC (decrescente)
results_grid_df = results_grid_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

# Formatar valores com 4 casas decimais
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    results_grid_df[col] = results_grid_df[col].apply(lambda x: f"{x:.4f}")

# print("\n", results_grid_df.to_string(index=False))
display(results_grid_df)

# Identificar melhor modelo
print("\n" + "=" * 80)
print(f"🏆 MELHOR MODELO OTIMIZADO (ROC-AUC): {results_grid_df.iloc[0]['Modelo']}")
print("=" * 80)


# ## 6. ASSESS - Avaliação Detalhada dos Modelos

# In[30]:


print("\n" + "=" * 80)
print("FASE 6: ASSESS - AVALIAÇÃO DETALHADA DOS MODELOS")
print("=" * 80)

print("COMPARAÇÃO: MODELOS DEFAULT vs OTIMIZADOS")

comparison_data = []
for name in models.keys():
    comparison_data.append({
        'Modelo': name,
        'ROC-AUC (Default)': results[name]['roc_auc'],
        'ROC-AUC (Otimizado)': grid_results[name]['roc_auc'],
        'Melhoria': grid_results[name]['roc_auc'] - results[name]['roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('ROC-AUC (Otimizado)', ascending=False).reset_index(drop=True)

# Formatar valores
comparison_df['ROC-AUC (Default)'] = comparison_df['ROC-AUC (Default)'].apply(lambda x: f"{x:.4f}")
comparison_df['ROC-AUC (Otimizado)'] = comparison_df['ROC-AUC (Otimizado)'].apply(lambda x: f"{x:.4f}")
comparison_df['Melhoria'] = comparison_df['Melhoria'].apply(lambda x: f"{x:+.4f}")

display(comparison_df)

print(" - SELEÇÃO DO MELHOR MODELO")

# Encontrar o melhor modelo baseado em ROC-AUC
best_model_name = max(grid_results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model_info = grid_results[best_model_name]

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"\nMétricas:")
print(f"   • Accuracy:  {best_model_info['accuracy']:.4f}")
print(f"   • Precision: {best_model_info['precision']:.4f}")
print(f"   • Recall:    {best_model_info['recall']:.4f}")
print(f"   • F1-Score:  {best_model_info['f1']:.4f}")
print(f"   • ROC-AUC:   {best_model_info['roc_auc']:.4f}")

print(f"\nMelhores Hiperparâmetros:")
for param, value in best_model_info['best_params'].items():
    print(f"   • {param}: {value}")

# Salvar o melhor modelo separadamente
best_model_final = best_model_info['model']
save_model(best_model_final, "BEST_MODEL_FINAL", MODELS_DIR)


# In[31]:


# Confusion Matrix Best Model

cm = confusion_matrix(y_test, best_model_info['y_pred'])

fig = go.Figure(go.Heatmap(
    z=cm,
    x=['Não Churn', 'Churn'],
    y=['Não Churn', 'Churn'],
    colorscale='Blues',
    showscale=True,
    hovertemplate='Real: %{y}<br>Predito: %{x}<br>Quantidade: %{z}<extra></extra>'
))

for i in range(2):
    for j in range(2):
        fig.add_annotation(
            x=j,
            y=i,
            text=str(cm[i][j]),
            showarrow=False,
            font=dict(
                size=18,
                color='white' if cm[i][j] > cm.max()/2 else 'black'
            )
        )

fig.update_layout(
    title=f'Matriz de Confusão — {best_model_name}',
    height=500
)

fig.show()


# In[32]:


# ROC Curve Best Model

fpr, tpr, _ = roc_curve(y_test, best_model_info['y_pred_proba'])
auc_score = auc(fpr, tpr)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name=f'{best_model_name} (AUC = {auc_score:.4f})',
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random (AUC = 0.500)',
    line=dict(dash='dash', color='gray')
))

fig.update_layout(
    title='Curva ROC — Melhor Modelo',
    xaxis_title='FPR',
    yaxis_title='TPR',
    height=500
)

fig.show()


# In[33]:


# Feature Importance Best Model

model = best_model_info['model']

if hasattr(model, 'feature_importances_'):
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=feature_importance_df['Importance'],
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        title=f'Top 15 Features — {best_model_name}',
        yaxis={'categoryorder': 'total ascending'},
        height=600
    )

    fig.show()
else:
    print("Modelo não possui feature_importances_.")


# In[34]:


# Report Best Model

print(f"\nMODELO: {best_model_name}")
print("=" * 60)

print(classification_report(
    y_test,
    best_model_info['y_pred'],
    target_names=['Não Churn', 'Churn'],
    digits=4
))


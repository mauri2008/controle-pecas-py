import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem",
    page_icon="⚙️",
    layout="wide",
)

# Mapeamento de classes
classes = {0: "Aprovada", 1: "Retrabalho", 2: "Refugo"}


# 1. Gerador e Treinador de Modelo (com cache para não re-treinar a cada interação)
@st.cache_data
def treinar_modelo_svm():
    # Gerar dados sintéticos de classificação multiclasse (3 classes)
    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Converter para DataFrame e ajustar escala realista de peças usinadas
    df = pd.DataFrame(X, columns=["Peso_gramas", "Espessura_mm"])
    df["Peso_gramas"] = (df["Peso_gramas"] * 15) + 200
    df["Espessura_mm"] = (df["Espessura_mm"] * 2.5) + 15
    df["Classe_ID"] = y
    df["Status"] = df["Classe_ID"].map(classes)

    # Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        df[["Peso_gramas", "Espessura_mm"]],
        df["Classe_ID"],
        test_size=0.25,
        random_state=42,
    )

    # Treinamento do Modelo SVM (Kernel RBF)
    modelo = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    modelo.fit(X_train, y_train)

    # Predições de teste para calcular matriz de confusão
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return modelo, cm, df


# Executa o carregamento dos dados e treino do modelo
modelo, cm, df_dados = treinar_modelo_svm()

# Interface Streamlit
st.title("⚙️ Controle de Qualidade de Peças - Usinagem")
st.markdown(
    "Sistema inteligente para classificação de peças usinadas baseado em tolerâncias de **Peso** e **Espessura**."
)

st.divider()

col_esq, col_dir = st.columns([1, 1], gap="large")

# --- COLUNA ESQUERDA: Formulário do Inspetor ---
with col_esq:
    st.subheader("🔍 Inspeção de Peça")
    st.write("Ajuste os valores medidos na peça para realizar a avaliação:")

    # Sliders para entrada de dados
    peso = st.slider(
        "Peso da Peça (g):",
        min_value=150.0,
        max_value=250.0,
        value=200.0,
        step=0.1,
    )

    espessura = st.slider(
        "Espessura da Peça (mm):",
        min_value=8.0,
        max_value=22.0,
        value=15.0,
        step=0.1,
    )

    # Botão de avaliação
    if st.button("Avaliar Peça", type="primary", use_container_width=True):
        # Predição da peça inserida
        entrada = np.array([[peso, espessura]])
        predicao = modelo.predict(entrada)[0]
        resultado = classes[predicao]

        st.subheader("Resultado da Avaliação:")

        # Exibição de mensagens customizadas conforme a classe
        if resultado == "Aprovada":
            st.success("✅ **PEÇA APROVADA**: Dentro das especificações!")
        elif resultado == "Retrabalho":
            st.warning(
                "⚠️ **RETRABALHO**: Peça necessita de ajustes ou reusinagem."
            )
        else:
            st.error(
                "🚨 **REFUGO**: Peça fora dos padrões toleráveis (Descarte)."
            )

# --- COLUNA DIREITA: Desempenho do Modelo (Matriz de Confusão) ---
with col_dir:
    st.subheader("📊 Desempenho do Modelo SVM")
    st.write("Matriz de Confusão do modelo treinado com os dados sintéticos:")

    labels_classes = ["Aprovada", "Retrabalho", "Refugo"]

    # Plotagem da Matriz de Confusão
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_classes,
        yticklabels=labels_classes,
        ax=ax,
    )
    ax.set_xlabel("Classe Preditada")
    ax.set_ylabel("Classe Real")
    ax.set_title("Matriz de Confusão")

    st.pyplot(fig)

st.divider()

# --- NOVA SEÇÃO: Exibição do DataFrame Sintético ---
st.subheader("📋 Base de Dados Sintética do Treinamento")

tab1, tab2 = st.tabs(["📄 Tabela de Dados", "📈 Resumo Estatístico"])

with tab1:
    st.dataframe(
        df_dados[["Peso_gramas", "Espessura_mm", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.dataframe(
        df_dados[["Peso_gramas", "Espessura_mm"]].describe(),
        use_container_width=True,
    )

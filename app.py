import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# CAMINHOS
# -------------------------------------------------------------
MODEL_PATH = Path("models/obesity_pipeline.pkl")
DATA_PATH = Path("Obesity.csv")

# -------------------------------------------------------------
# TRADUÇÕES (conforme dicionário oficial)
# -------------------------------------------------------------
translate_obesity = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}

translate_gender = {"Male": "Masculino", "Female": "Feminino"}
translate_yes_no = {"yes": "Sim", "no": "Não"}

translate_freq = {
    "no": "Nunca",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
}

translate_mtrans = {
    "Walking": "A Pé",
    "Bike": "Bicicleta",
    "Automobile": "Carro",
    "Motorbike": "Moto",
    "Public_Transportation": "Transporte Público",
}

ch2o_labels = {1: "< 1L/dia", 2: "1–2L/dia", 3: "> 2L/dia"}


# -------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------------
@st.cache_data
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode train_model.py.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Obesity.csv não encontrado")
    
    df = pd.read_csv(DATA_PATH)

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Age"] = df["Age"].round().astype("Int64")

    scale_cols = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in scale_cols:
        df[col] = df[col].round().astype("Int64")

    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Obesity_PT"] = df["Obesity"].map(translate_obesity)
    df["CH2O_Label"] = df["CH2O"].map(ch2o_labels)

    return df


def translate_prediction(value):
    return translate_obesity.get(value, value)


# -------------------------------------------------------------
# CONFIGURAÇÃO DO LAYOUT
# -------------------------------------------------------------
st.set_page_config(page_title="Preditor de Obesidade", page_icon="🧬", layout="wide")

st.title("Sistema Preditivo de Obesidade – FIAP Tech Challenge")
st.caption("Modelo, predição e dashboard analítico completos.")

model = load_model()
df = load_data()

st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Predição individual", "Dashboard analítico"])


# =================================================================
# PÁGINA 1 — PREDIÇÃO INDIVIDUAL
# ==================================================================
if page == "Predição individual":
    st.header("Predição individual do nível de obesidade")

    with st.form("form_predicao"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gênero", ["Male", "Female"], format_func=lambda x: translate_gender[x])
            age = st.number_input("Idade", min_value=14, max_value=80, value=25)
            height = st.number_input("Altura (m)", min_value=1.0, max_value=2.2, value=1.75)
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=80.0)
            family_history = st.selectbox("Histórico familiar de obesidade?", ["yes", "no"], format_func=lambda x: translate_yes_no[x])
            favc = st.selectbox("Consumo de comida muito calórica?", ["yes", "no"], format_func=lambda x: translate_yes_no[x])
            fcvc = st.selectbox("Frequência de consumo de vegetais", [1, 2, 3])
            ncp = st.selectbox("Refeições principais por dia", [1, 2, 3, 4])

        with col2:
            caec = st.selectbox("Lanches entre refeições (CAEC)", list(translate_freq.keys()), format_func=lambda x: translate_freq[x])
            smoke = st.selectbox("Fuma?", ["yes", "no"], format_func=lambda x: translate_yes_no[x])
            ch2o = st.selectbox("Consumo de água diário (CH2O)", [1, 2, 3], format_func=lambda x: ch2o_labels[x])
            scc = st.selectbox("Monitora calorias (SCC)?", ["yes", "no"], format_func=lambda x: translate_yes_no[x])
            faf = st.selectbox("Atividade física semanal (FAF)", [0, 1, 2, 3])
            tue = st.selectbox("Horas em eletrônicos por dia (TUE)", [0, 1, 2])
            calc = st.selectbox("Consumo de álcool (CALC)", list(translate_freq.keys()), format_func=lambda x: translate_freq[x])
            mtrans = st.selectbox("Transporte (MTRANS)", list(translate_mtrans.keys()), format_func=lambda x: translate_mtrans[x])

        submit = st.form_submit_button("Prever Obesidade")

    if submit:
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }])

        result = model.predict(input_df)[0]
        result_pt = translate_prediction(result)
        imc = weight / (height ** 2)

        st.success(f"**Nível previsto:** {result_pt}")
        st.info(f"IMC aproximado: **{imc:.2f}**")
        st.caption("Esta ferramenta é apoio à decisão médica, não diagnóstico clínico.")


# ==================================================================
# DASHBOARD ANALÍTICO (CORRIGIDO + KEYS ÚNICOS)
# ==================================================================

else:
    st.header("Dashboard Analítico Interativo")

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # GRÁFICO 1 — Distribuição dos níveis de obesidade
    # ---------------------------------------------------------
    with col1:
        st.subheader("Distribuição dos níveis de obesidade")

        dist_df = df["Obesity_PT"].value_counts().reset_index()
        dist_df.columns = ["Nível", "Quantidade"]

        fig_dist = px.bar(
            dist_df,
            x="Nível", y="Quantidade",
            text="Quantidade",
            color="Nível",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues,
            height=450
        )
        fig_dist.update_traces(textposition="outside")
        fig_dist.update_layout(showlegend=False)

        st.plotly_chart(fig_dist, width='stretch', key="grafico_distribuicao")

    # ---------------------------------------------------------
    # GRÁFICO 2 — IMC médio por nível de obesidade
    # ---------------------------------------------------------
    with col2:
        st.subheader("IMC médio por nível de obesidade")

        bmi_df = df.groupby("Obesity_PT")["BMI"].mean().reset_index()

        fig_bmi = px.bar(
            bmi_df,
            x="Obesity_PT",
            y="BMI",
            text="BMI",
            template="plotly_dark",
            color="Obesity_PT",
            color_discrete_sequence=px.colors.sequential.Plasma,
            height=450
        )

        fig_bmi.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bmi.update_layout(showlegend=False)

        st.plotly_chart(fig_bmi, width='stretch', key="grafico_imc_medio")

    # ---------------------------------------------------------
    # GRÁFICO 3 — Altura vs Peso (MANTIDO NO ESTILO BUBBLE)
    # ---------------------------------------------------------
    st.subheader("Altura vs Peso (Bubble Chart por nível de obesidade)")

    fig_height_weight = px.scatter(
        df,
        x="Height",
        y="Weight",
        color="Obesity_PT",
        size="BMI",
        template="plotly_dark",
        hover_data=["Age", "BMI"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=550
    )

    st.plotly_chart(fig_height_weight, width='stretch', key="grafico_altura_peso")

    # ---------------------------------------------------------
    # GRÁFICO 4 — Consumo diário de água (CH2O) x IMC (BARRA)
    # ---------------------------------------------------------
    st.subheader("Consumo diário de água (CH2O) x IMC (comparação por categoria)")

    ch2o_df = df.groupby("CH2O_Label")["BMI"].mean().reset_index()

    fig_ch2o_bar = px.bar(
        ch2o_df,
        x="CH2O_Label",
        y="BMI",
        color="BMI",
        text=ch2o_df["BMI"].round(1),
        template="plotly_dark",
        color_continuous_scale="Blues",
        height=450
    )

    fig_ch2o_bar.update_traces(textposition="outside")
    fig_ch2o_bar.update_layout(
        xaxis_title="Consumo de água (CH2O)",
        yaxis_title="IMC Médio",
    )

    st.plotly_chart(fig_ch2o_bar, width='stretch', key="bar_ch2o")


    # ---------------------------------------------------------
    # GRÁFICO 5 — Atividade física (FAF) x IMC (BARRA)
    # ---------------------------------------------------------
    st.subheader("Atividade física semanal (FAF) x IMC (comparação por categoria)")

    faf_df = df.groupby("FAF")["BMI"].mean().reset_index()
    faf_df = faf_df.sort_values("FAF")

    fig_faf_bar = px.bar(
        faf_df,
        x="FAF",
        y="BMI",
        color="BMI",
        text=faf_df["BMI"].round(1),
        template="plotly_dark",
        color_continuous_scale="Viridis",
        height=450
    )

    fig_faf_bar.update_traces(textposition="outside")
    fig_faf_bar.update_layout(
        xaxis_title="Frequência de atividade física semanal (FAF)",
        yaxis_title="IMC Médio",
    )

    st.plotly_chart(fig_faf_bar, width='stretch', key="bar_faf")


    # ---------------------------------------------------------
    # GRÁFICO 6 — Consumo calórico (FAVC) x IMC (BARRA)
    # ---------------------------------------------------------
    st.subheader("Consumo de comida muito calórica (FAVC) x IMC (comparação por categoria)")

    favc_df = df.groupby("FAVC")["BMI"].mean().reset_index()
    favc_df["FAVC_Label"] = favc_df["FAVC"].map({"yes": "Sim", "no": "Não"})

    fig_favc_bar = px.bar(
        favc_df,
        x="FAVC_Label",
        y="BMI",
        color="BMI",
        text=favc_df["BMI"].round(1),
        template="plotly_dark",
        color_continuous_scale="Plasma",
        height=450
    )

    fig_favc_bar.update_traces(textposition="outside")
    fig_favc_bar.update_layout(
        xaxis_title="Consumo calórico (FAVC)",
        yaxis_title="IMC Médio",
    )

    st.plotly_chart(fig_favc_bar, width='stretch', key="bar_favc")


    # ---------------------------------------------------------
    # GRÁFICO 7 — Heatmap de correlação NUMÉRICO (NÃO bubble)
    # ---------------------------------------------------------
    st.subheader("Mapa de correlação das variáveis numéricas (Heatmap)")

    numeric_cols = ["Age", "Height", "Weight", "BMI", "FCVC", "CH2O", "FAF", "TUE"]
    corr = df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Teal",
        template="plotly_dark",
        height=1000
    )

    fig_corr.update_layout(
        xaxis_title="Variáveis",
        yaxis_title="Variáveis",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_corr, width='stretch', key="heatmap_corr")


    # ---------------------------------------------------------
    # RODAPÉ
    # ---------------------------------------------------------
    st.markdown("---")
    st.caption("© Dashboard desenvolvido para FIAP – Tech Challenge por Diego Maradini.")


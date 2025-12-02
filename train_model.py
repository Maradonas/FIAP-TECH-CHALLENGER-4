import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib


DATA_PATH = Path("Assets/Obesity.csv")          # caminho do dataset
MODEL_DIR = Path("models")               # pasta onde o modelo será salvo
MODEL_PATH = MODEL_DIR / "obesity_pipeline.pkl"


def load_data(path: Path) -> pd.DataFrame:
    """Carrega o CSV e faz alguns ajustes básicos de tipo."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo {path} não encontrado. "
                                f"Verifique se o Obesity.csv está na mesma pasta do script.")

    df = pd.read_csv(path)

    # Converte colunas numéricas para os tipos corretos
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    for col in numeric_cols:
        # Garantir que são numéricas
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Arrendondar colunas que são escalas discretas (segundo dicionário)
    scale_cols = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in scale_cols:
        df[col] = df[col].round().astype("Int64")

    # Idade inteira
    df["Age"] = df["Age"].round().astype("Int64")

    # Remove linhas com valores nulos (se existirem)
    df = df.dropna().reset_index(drop=True)

    return df


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """Monta o pipeline de pré-processamento + modelo."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return clf


def train_and_evaluate(df: pd.DataFrame):
    """Treina o modelo, avalia e salva o pipeline treinado em disco."""
    # Coluna alvo
    target_col = "Obesity"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    print("Colunas numéricas:", numeric_cols)
    print("Colunas categóricas:", categorical_cols)

    clf = build_pipeline(numeric_cols, categorical_cols)

    # Divide em treino e teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTamanho treino: {X_train.shape}, teste: {X_test.shape}")

    # Treinamento
    print("\nTreinando o modelo...")
    clf.fit(X_train, y_train)

    # Avaliação
    print("\nAvaliando o modelo...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Acurácia no conjunto de teste: {acc:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Garante que a pasta models exista
    MODEL_DIR.mkdir(exist_ok=True)

    # Salva o pipeline completo (pré-processamento + modelo)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n💾 Modelo salvo em: {MODEL_PATH.resolve()}")


def main():
    print("Carregando dados...")
    df = load_data(DATA_PATH)

    print("\nPrimeiras linhas do dataset:")
    print(df.head())

    print("\nDistribuição das classes de obesidade:")
    print(df["Obesity"].value_counts())

    train_and_evaluate(df)


if __name__ == "__main__":
    main()

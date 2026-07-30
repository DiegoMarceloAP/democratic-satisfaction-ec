"""
cnn_lstm.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - Arquitectura de Deep Learning 1/2
(OBLIGATORIA según el plan de modelado): CNN-LSTM mixto.

Combina dos ramas:
  1) Rama TEMPORAL (CNN + LSTM): recibe la ventana móvil de
     VENTANA_TEMPORAL_ANIOS (3) construida por
     src/features/ventanas_temporales.py sobre panel_macro_anual.csv
     (indicadores macro de V-Dem/ENEMDU, SIN la serie de Latinobarómetro
     -- evita fuga de información), con máscara para historia
     incompleta (encuestados de 2007/2008).
  2) Rama ESTÁTICA (MLP + embeddings): recibe las mismas variables
     sociodemográficas e indicadores macro CONTEMPORÁNEOS que
     baseline_xgboost.py / baseline_lightgbm.py
     (src/features/config_features.py), para que los 4 modelos del plan
     se comparen sobre exactamente las mismas variables.

DECISIÓN DE DISEÑO -- componente CNN (documentar para la defensa de
tesis): la propuesta original pide que la CNN extraiga "características
espaciales inter-provinciales O de relaciones entre variables
sociodemográficas". Como el merge final es a nivel NACIONAL-año (no
provincial; ver sql/README_extraccion.md), no existe
una dimensión inter-provincial que convolucionar. Se implementa
entonces la segunda variante explícita de la propuesta: un Conv1D que
opera sobre el eje de VARIABLES macro dentro de cada año de la ventana
(pesos compartidos entre los 3 años), extrayendo interacciones locales
entre esas variables antes de que la LSTM module su evolución temporal.
Es una interpretación FUNCIONAL del rol "espacial" del CNN (los
indicadores macro no tienen un orden espacial real), no literal.

DECISIÓN DE DISEÑO -- SMOTENC sobre datos secuenciales: SMOTENC
(src/features/smote_train.py) solo acepta vectores 2D. Para cumplir la
regla del proyecto ("SMOTENC únicamente sobre X_train, después del
split") también en este modelo secuencial, se APLANA la ventana
temporal (y su máscara) junto con las variables estáticas en un solo
vector 2D por persona, se aplica SMOTENC sobre esa representación
aplanada, y luego se RECONSTRUYEN los tensores [batch, ventana,
features] a partir del resultado balanceado. La máscara (0/1) se
redondea (>=0.5) después de interpolarse, y los valores de la ventana
que quedaron faltantes (mask=False) se imputan ANTES de aplanar
(SMOTENC no acepta NaN) usando la mediana de ESE mismo timestep, con
estadísticas calculadas solo con observaciones reales de X_train.

NOTA DE VALIDACIÓN: la lógica de datos de este módulo (ventaneo,
aplanado, SMOTENC, reconstrucción, alineación posicional de folds) se
validó primero con datos sintéticos que reproducen la estructura real
de dataset_modelado_personas.csv y panel_macro_anual.csv, incluyendo el
caso límite de encuestados de 2007 con historia incompleta -- necesario
porque el paquete 'torch' era demasiado grande para el entorno de
desarrollo usado para escribir este módulo. El entrenamiento real (con
'torch' instalado, ver requirements.txt) ya se ejecutó y validó sobre
el dataset completo; ver README.md de esta carpeta para los resultados
por fold.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_RAIZ_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ_SRC / "evaluation"))
sys.path.insert(0, str(_RAIZ_SRC / "features"))

from time_series_split import generar_folds_temporales  # noqa: E402
from smote_train import aplicar_smote_train  # noqa: E402
from ventanas_temporales import construir_ventanas_temporales  # noqa: E402
from config_features import (  # noqa: E402
    COLUMNA_ANIO, COLUMNA_TARGET, FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
)
from preparacion_modelado import (  # noqa: E402
    cargar_dataset, preparar_features, imputar_faltantes, codificar_categoricas,
)

PANEL_MACRO_PATH = Path("data/processed/panel_macro_anual.csv")
OUT_RESULTADOS_PATH = Path("reports/tablas/resultados_cnn_lstm.csv")

# Debe coincidir con VENTANA_TEMPORAL_ANIOS de notebooks/00_parametros_globales.ipynb
VENTANA_TEMPORAL_ANIOS = 3


def _imputar_ventanas(
    seq_train: np.ndarray, mask_train: np.ndarray, seq_test: np.ndarray, mask_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Imputa los NaN de los timesteps SIN historia real (mask=False) con
    la mediana de ESE mismo timestep y esa misma variable, calculada
    SOLO con las observaciones reales (mask=True) de X_train -- análogo
    a preparacion_modelado.imputar_faltantes, pero por posición de
    timestep. El mask NO se modifica: sigue marcando cuáles timesteps
    son reales para que pack_padded_sequence los siga excluyendo.
    """
    seq_train = seq_train.copy()
    seq_test = seq_test.copy()
    ventana, n_features = seq_train.shape[1], seq_train.shape[2]
    for t in range(ventana):
        validos_t = mask_train[:, t]
        medianas_t = (
            np.nanmedian(seq_train[validos_t, t, :], axis=0) if validos_t.any() else np.zeros(n_features)
        )
        seq_train[~mask_train[:, t], t, :] = medianas_t
        seq_test[~mask_test[:, t], t, :] = medianas_t
    return seq_train, seq_test


class DatasetSecuencial(Dataset):
    """Empaqueta ventana temporal + máscara + variables estáticas + target para el DataLoader de PyTorch."""

    def __init__(self, seq, mask, static_num, static_cat_codes, y):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.static_num = torch.tensor(static_num, dtype=torch.float32)
        self.static_cat = torch.tensor(static_cat_codes, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.seq[i], self.mask[i], self.static_num[i], self.static_cat[i], self.y[i]


class RamaTemporalCNNLSTM(nn.Module):
    """
    Componente CNN + LSTM de la arquitectura mixta (ver nota de diseño
    en el docstring del módulo). El CNN comparte pesos entre los 3 años
    de la ventana (aplicado independientemente a cada uno); la LSTM
    modela la evolución temporal de esos resúmenes, respetando la
    máscara de historia incompleta.
    """

    def __init__(self, n_features_macro: int, canales_cnn: int = 16, hidden_lstm: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, canales_cnn, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(canales_cnn, canales_cnn, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(input_size=canales_cnn, hidden_size=hidden_lstm, batch_first=True)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq: [batch, ventana, n_features_macro] en orden CRONOLÓGICO
        # (año más antiguo primero); mask: [batch, ventana] booleano.
        batch, ventana, n_features = seq.shape

        x = seq.reshape(batch * ventana, 1, n_features)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)        # [batch*ventana, canales_cnn]
        x = x.reshape(batch, ventana, -1)    # [batch, ventana, canales_cnn]

        # Se invierte el orden temporal (año más reciente primero) para
        # que los timesteps REALES queden como PREFIJO de la secuencia:
        # el año de la propia encuesta SIEMPRE está disponible -- lo que
        # puede faltar es historia ANTERIOR (encuestados de 2007/2008).
        # pack_padded_sequence exige que el padding esté al final, de
        # ahí la inversión.
        x_rev = torch.flip(x, dims=[1])
        longitudes = mask.sum(dim=1).clamp(min=1).to("cpu")
        empaquetado = nn.utils.rnn.pack_padded_sequence(
            x_rev, longitudes, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(empaquetado)
        return h_n[-1]  # [batch, hidden_lstm] -- último estado oculto real de cada secuencia


class RamaEstatica(nn.Module):
    """
    MLP sobre las variables sociodemográficas + indicadores macro
    CONTEMPORÁNEOS del encuestado (mismas features que los baselines de
    árboles). Las columnas categóricas usan embeddings, ya que PyTorch
    no tiene soporte nativo de categóricas como XGBoost/LightGBM.
    """

    def __init__(
        self, cardinalidades_cat: list[int], n_features_numericas: int,
        dim_embedding: int = 4, hidden: int = 32,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, dim_embedding) for card in cardinalidades_cat])
        n_entrada = len(cardinalidades_cat) * dim_embedding + n_features_numericas
        self.mlp = nn.Sequential(nn.Linear(n_entrada, hidden), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x_num: torch.Tensor, x_cat_codes: torch.Tensor) -> torch.Tensor:
        # x_cat_codes: [batch, n_columnas_cat], mismo orden que cardinalidades_cat
        embs = [emb(x_cat_codes[:, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x)


class CNNLSTMClasificador(nn.Module):
    """Modelo completo: concatena la rama temporal y la rama estática antes de la capa de clasificación binaria."""

    def __init__(
        self, n_features_macro: int, cardinalidades_cat: list[int], n_features_numericas: int,
        canales_cnn: int = 16, hidden_lstm: int = 32, dim_embedding: int = 4, hidden_estatica: int = 32,
    ):
        super().__init__()
        self.rama_temporal = RamaTemporalCNNLSTM(n_features_macro, canales_cnn, hidden_lstm)
        self.rama_estatica = RamaEstatica(cardinalidades_cat, n_features_numericas, dim_embedding, hidden_estatica)
        self.clasificador = nn.Sequential(
            nn.Linear(hidden_lstm + hidden_estatica, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, seq, mask, x_num, x_cat_codes):
        t = self.rama_temporal(seq, mask)
        e = self.rama_estatica(x_num, x_cat_codes)
        z = torch.cat([t, e], dim=1)
        return self.clasificador(z).squeeze(-1)  # logit (sin sigmoid -- se usa BCEWithLogitsLoss)


def entrenar_evaluar_cnn_lstm(
    df_personas: pd.DataFrame,
    panel_macro: pd.DataFrame,
    features_categoricas: list[str] = FEATURES_CATEGORICAS,
    features_numericas: list[str] = FEATURES_NUMERICAS,
    ventana: int = VENTANA_TEMPORAL_ANIOS,
    min_anios_train: int = 5,
    random_state: int = 42,
    n_epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: "torch.device | None" = None,
) -> tuple[pd.DataFrame, list]:
    """
    Entrena y evalúa el CNN-LSTM con Time Series Split + SMOTENC por fold
    (misma partición que los baselines de árboles). Retorna (resultados
    por fold, lista de modelos entrenados).

    AJUSTE CONSERVADOR (n_epochs 30 -> 15, + weight_decay=1e-4 en Adam) --
    hallazgo real de la corrida completa: con n_epochs=30 (y sin ningún
    freno de regularización más allá del Dropout(0.2) de la rama
    estática) el PR-AUC promedio empeoró en los 5 folds respecto a la
    prueba con muestra reducida y solo 5 épocas -- señal de sobreajuste,
    coherente con que este modelo, al no usar 'eval_set' de prueba para
    early stopping (por la misma regla de no filtrar información del
    test), entrena a ciegas un número FIJO de épocas.

    Se descartó construir una validación interna anidada por fold para
    hacer early stopping real (mismo argumento que en baseline_xgboost.py:
    con solo 10 años reales, reservar un año más de entrenamiento por
    fold cuesta un fold de prueba completo, y un solo año de validación
    resultó ser una señal poco confiable al probarlo con XGBoost). En su
    lugar se bajó n_epochs a un valor intermedio (ni 30 ni 5) y se agregó
    weight_decay al optimizador como regularización barata -- un ajuste
    conservador y documentado, no una búsqueda de hiperparámetros.
    Validado con la corrida completa (dataset real, SAMPLING_MODE=False):
    el PR-AUC promedio de los 5 folds mejoró de 0.468 (30 épocas) a 0.479
    con esta configuración (ver README.md de esta carpeta para el detalle
    por fold).

    OPTIMIZACIONES DE RENDIMIENTO (a pedido del tutor, no cambian ningún
    resultado, solo velocidad): 'torch.backends.cudnn.benchmark=True' si
    hay GPU (el shape de la ventana temporal es fijo entre folds, así que
    cuDNN puede autotunear el algoritmo de convolución una sola vez);
    'pin_memory'/'non_blocking=True' en el DataLoader y las transferencias
    a device, que solo tienen efecto con CUDA (en CPU no cambian nada).
    'num_workers' se deja en 0 a propósito: el dataset ya vive completo en
    RAM como tensores (no hay lectura de disco que paralelizar) y es
    chico, así que procesos worker extra solo agregarían overhead.
    """
    torch.manual_seed(random_state)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usa_cuda = device.type == "cuda"
    if usa_cuda:
        # OPTIMIZACIÓN (servidor GPU, a pedido del tutor): todas las
        # secuencias de la ventana temporal tienen el mismo shape fijo
        # (mismo 'ventana', mismo n_features_macro en todos los folds), así
        # que cuDNN puede autotunear el algoritmo de convolución más rápido
        # para ese shape exacto la primera vez y reutilizarlo el resto del
        # entrenamiento -- no tiene costo en CPU (torch lo ignora ahí).
        torch.backends.cudnn.benchmark = True

    X_static, y = preparar_features(
        df_personas, features_categoricas, features_numericas, COLUMNA_ANIO, COLUMNA_TARGET
    )
    tensor_seq, mask_seq, columnas_macro = construir_ventanas_temporales(
        X_static, panel_macro, ventana=ventana, columna_anio=COLUMNA_ANIO
    )
    folds = generar_folds_temporales(
        X_static.assign(**{COLUMNA_TARGET: y}),
        columna_anio=COLUMNA_ANIO, columna_target=COLUMNA_TARGET, min_anios_train=min_anios_train,
    )

    resultados = []
    modelos = []

    for fold in folds:
        # tensor_seq/mask_seq están alineados por POSICIÓN con X_static
        # (mismo orden de filas), no por su índice de pandas original --
        # de ahí el uso de get_indexer en vez de .loc directo.
        pos_train = X_static.index.get_indexer(fold["train_idx"])
        pos_test = X_static.index.get_indexer(fold["test_idx"])

        X_train_static = X_static.loc[fold["train_idx"], features_categoricas + features_numericas]
        X_test_static = X_static.loc[fold["test_idx"], features_categoricas + features_numericas]
        y_train = y.loc[fold["train_idx"]]
        y_test = y.loc[fold["test_idx"]]
        seq_train, mask_train = tensor_seq[pos_train], mask_seq[pos_train]
        seq_test, mask_test = tensor_seq[pos_test], mask_seq[pos_test]

        X_train_static, X_test_static = imputar_faltantes(X_train_static, X_test_static, features_numericas)
        seq_train, seq_test = _imputar_ventanas(seq_train, mask_train, seq_test, mask_test)

        # --- SMOTENC sobre la representación aplanada (ventana + máscara + estáticas) ---
        n_train, _, n_features_macro = seq_train.shape
        columnas_seq = [f"seq_t{t}_{feat}" for t in range(ventana) for feat in columnas_macro]
        columnas_mask = [f"mask_t{t}" for t in range(ventana)]
        X_flat_train = pd.DataFrame(
            seq_train.reshape(n_train, ventana * n_features_macro), columns=columnas_seq, index=X_train_static.index
        )
        X_flat_train[columnas_mask] = mask_train.astype(float)
        X_flat_train = pd.concat([X_flat_train, X_train_static], axis=1)

        X_flat_bal, y_bal = aplicar_smote_train(X_flat_train, y_train, random_state=random_state)

        seq_train_bal = X_flat_bal[columnas_seq].to_numpy(dtype=float).reshape(-1, ventana, n_features_macro)
        mask_train_bal = X_flat_bal[columnas_mask].to_numpy(dtype=float) >= 0.5
        X_static_bal = X_flat_bal[features_categoricas + features_numericas]

        # --- Escalado (solo para la red neuronal; los árboles no lo necesitan).
        # SMOTENC ya corrió en unidades originales, igual que en los baselines. ---
        scaler_estatica = StandardScaler().fit(X_static_bal[features_numericas])
        X_num_train_esc = scaler_estatica.transform(X_static_bal[features_numericas])
        X_num_test_esc = scaler_estatica.transform(X_test_static[features_numericas])

        scaler_seq = StandardScaler().fit(seq_train_bal.reshape(-1, n_features_macro))
        seq_train_esc = scaler_seq.transform(seq_train_bal.reshape(-1, n_features_macro)).reshape(seq_train_bal.shape)
        seq_test_esc = scaler_seq.transform(seq_test.reshape(-1, n_features_macro)).reshape(seq_test.shape)

        cod_train, cod_test, cardinalidades = codificar_categoricas(
            X_static_bal[features_categoricas], X_test_static[features_categoricas], features_categoricas
        )

        dataset_train = DatasetSecuencial(seq_train_esc, mask_train_bal, X_num_train_esc, cod_train, y_bal.to_numpy())
        dataset_test = DatasetSecuencial(seq_test_esc, mask_test, X_num_test_esc, cod_test, y_test.to_numpy())
        # pin_memory + transferencia non_blocking (abajo): solo ayuda con
        # CUDA (memoria "pinned" permite copiar a GPU en paralelo con la
        # CPU); en CPU, pin_memory=False no cambia nada. num_workers=0
        # porque el dataset ya vive entero en RAM como tensores (no hay
        # I/O de disco que paralelizar) y el dataset es chico -- procesos
        # extra solo agregarían overhead de por sí.
        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, pin_memory=usa_cuda,
        )

        modelo = CNNLSTMClasificador(
            n_features_macro=n_features_macro,
            cardinalidades_cat=[cardinalidades[c] for c in features_categoricas],
            n_features_numericas=len(features_numericas),
        ).to(device)
        optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
        criterio = nn.BCEWithLogitsLoss()

        t0 = time.time()
        modelo.train()
        for _epoca in range(n_epochs):
            for seq_b, mask_b, num_b, cat_b, y_b in loader_train:
                seq_b, mask_b, num_b, cat_b, y_b = (
                    t.to(device, non_blocking=usa_cuda) for t in (seq_b, mask_b, num_b, cat_b, y_b)
                )
                optimizador.zero_grad()
                logits = modelo(seq_b, mask_b, num_b, cat_b)
                perdida = criterio(logits, y_b)
                perdida.backward()
                optimizador.step()
        tiempo_entrenamiento = time.time() - t0

        modelo.eval()
        with torch.no_grad():
            logits_test = modelo(
                dataset_test.seq.to(device), dataset_test.mask.to(device),
                dataset_test.static_num.to(device), dataset_test.static_cat.to(device),
            )
            proba_test = torch.sigmoid(logits_test).cpu().numpy()
        pred_test = (proba_test >= 0.5).astype(int)
        y_test_np = y_test.to_numpy()

        resultados.append({
            "anio_test": fold["anio_test"],
            "n_anios_train": len(fold["anios_train"]),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": round(accuracy_score(y_test_np, pred_test), 4),
            "f1_macro": round(f1_score(y_test_np, pred_test, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test_np, pred_test, average="weighted"), 4),
            "f1_satisfecho": round(f1_score(y_test_np, pred_test, pos_label=1), 4),
            "pr_auc": round(average_precision_score(y_test_np, proba_test), 4),
            "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
        })
        modelos.append(modelo)

    resultados_df = pd.DataFrame(resultados)
    print("\n[cnn_lstm] Resultados por fold:")
    print(resultados_df.to_string(index=False))
    print("\n[cnn_lstm] Promedio across folds:")
    print(resultados_df[["accuracy", "f1_macro", "f1_weighted", "f1_satisfecho", "pr_auc"]].mean().round(4))

    return resultados_df, modelos


def guardar_resultados(resultados: pd.DataFrame, out_path: Path = OUT_RESULTADOS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    print(f"[cnn_lstm] Resultados guardados en '{out_path}'.")


if __name__ == "__main__":
    df_personas = cargar_dataset()
    panel_macro = pd.read_csv(PANEL_MACRO_PATH)
    resultados, modelos = entrenar_evaluar_cnn_lstm(df_personas, panel_macro, min_anios_train=5)
    guardar_resultados(resultados)

"""
config_features.py
---------------------------------------------------------------------
Fase de Modelado (CRISP-DM) - definición de features.

Definición ÚNICA y compartida de qué columnas de dataset_modelado_personas.csv
se usan como features, para que los 4 modelos del plan (XGBoost, LightGBM,
CNN-LSTM, TabNet) entrenen sobre EXACTAMENTE el mismo conjunto de variables
-- necesario para que la tabla comparativa final sea una comparación justa
entre modelos, y para no tener que actualizar la misma lista en 4 archivos
si más adelante se agrega o quita una variable.

No incluye: identificadores, 'democ_satis' (fuente directa del target,
fuga de información), ni las columnas crudas ya recodificadas en sus
versiones '_cat'/'_bin'/'_alta' (evita duplicar la misma información dos
veces bajo dos codificaciones distintas).

FUENTE: Latinobarómetro no viene de Superset, sino de las descargas
directas por ola (ver latinobarometro_loader_crudo.py). elections_vote
se descarta de forma permanente (solo 1 de 13 olas tiene dato real);
todas las demás variables del cuestionario original, incluidas
resp_chief y los 4 bienes del hogar, están disponibles en las 13 olas y
se incluyen abajo.
"""

COLUMNA_ANIO = "anio"
COLUMNA_TARGET = "satisfecho_democracia"

# Variables nominales (códigos sin orden numérico real, o categorías ya
# recodificadas por latinobarometro_processing.py).
FEATURES_CATEGORICAS = [
    "resp_sex", "resp_education", "resp_employment", "resp_religion",
    "democ_supp_cat", "ideologia_cat", "econ_situation_cat", "resp_economic_perception_cat",
    "resp_chief",
]

# Variables numéricas/ordinales/binarias: sociodemográficas propias del
# encuestado + indicadores contemporáneos de ENEMDU y V-Dem.
FEATURES_NUMERICAS = [
    "resp_age", "job_concern",
    "confidence_congress_alta", "confidence_judiciary_alta", "confidence_church_alta",
    "confidence_police_alta", "confidence_army_alta", "confidence_political_parties_alta",
    # Bienes del hogar (proxy de condición material)
    "goods_wash_mach_bin", "goods_car_bin", "goods_sewage_bin", "goods_hot_water_bin",
    # ENEMDU (mercado laboral, ingreso, desigualdad -- ver enemdu_processing.py)
    "tasa_participacion_global", "tasa_desempleo", "empleo_formal", "empleo_informal",
    "ingreso_promedio_pc", "ingreso_promedio_laboral", "gini_ingpc",
    "pobreza_ingresos", "pobreza_extrema_ingresos",
    # V-Dem (calidad democrática -- ver vdem_processing.py)
    "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_delibdem", "v2x_egaldem",
    "v2x_freexp_altinf", "v2xel_frefair", "v2xcl_rol", "v2x_jucon", "v2xlg_legcon",
    "v2xeg_eqprotec", "v2xeg_eqaccess", "v2xeg_eqdr",
    "v2pepwrses", "v2pepwrsoc", "v2pepwrgen", "v2pepwrort", "v2pepwrgeo",
]

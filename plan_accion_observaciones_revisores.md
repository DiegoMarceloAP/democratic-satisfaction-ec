# Plan de acción: observaciones de los revisores

Fuentes: María Gabriela Cajamarca, Ph.D. (Directora/Asesora) y Rolando Armas (revisor externo).

Nota sobre los archivos recibidos: el .zip "Degree_project_Diego_Altamirano" descargado de Overleaf corresponde a una plantilla vacía (secciones sin contenido, nombres de archivo distintos a los capítulos actuales), no a la versión final de la tesis. Este plan se construyó cruzando las observaciones contra el PDF completo más reciente ("Degree_project_Diego_Altamirano (2).pdf"), que sí refleja el contenido actual. Conviene volver a exportar el .zip de Overleaf antes de la siguiente entrega.

Las observaciones se agrupan en tres niveles según el esfuerzo requerido, no por revisor, porque varias se repiten o se complementan entre ambos.

---

## Nivel A — Ediciones de texto directas (sin rehacer análisis)

Estas se pueden resolver editando la prosa existente; no requieren volver a entrenar nada.

| # | Observación | Acción concreta | Ubicación |
|---|---|---|---|
| A1 | (Gabriela, mayor) Evitar lenguaje causal: "influyen", "determinantes", "evidencia a favor" | Reemplazar por "predictores", "asociaciones", "contribuciones a la predicción"; acotar las implicaciones de política pública a hipótesis que requieren validación causal | Resumen, Abstract, Cap. 5 (SHAP, hallazgo 2), Cap. 6 (Hallazgos, Reflexión final) |
| A2 | (Gabriela, menor) Orden de objetivos invertido entre capítulos | En Cap. 6.1, intercambiar las etiquetas: el objetivo de XAI es el "quinto" (no el cuarto) y la evaluación de métricas es el "cuarto" (no el quinto), para que coincida con el orden de la Sección 1.3.2 | Cap. 1.3.2 vs Cap. 6.1 |
| A3 | (Gabriela, menor) Referencia a "octubre 2025" en ENEMDU, fuera del periodo de estudio (2007-2024) | Verificar contra los datos si existe un segundo hueco real dentro de 2007-2024 aparte de noviembre 2021; si no lo hay, eliminar "octubre 2025" y dejar solo el hueco real dentro del periodo | Cap. 4.3.2 (Fase 2 – ENEMDU) |
| A4 | (Gabriela, menor) Matizar "por primera vez" | Cambiar a "hasta donde se identificó en la revisión de literatura realizada" (o equivalente), salvo que se amplíe la búsqueda del Estado del Arte para respaldar la afirmación tal cual | Resumen, Abstract |
| A5 | (Gabriela, menor) Descripción incorrecta de la línea base PR-AUC "sin habilidad" | Reformular: el valor de PR-AUC "sin habilidad" corresponde a la prevalencia de la clase positiva en el fold, no a que el clasificador "prediga siempre la clase mayoritaria" (PR-AUC depende de scores/probabilidades, no de una etiqueta dura). Ajustar la descripción de cómo se genera esa línea base | Cap. 5.1 (Diseño de la evaluación) y Cap. 6.2 (repite la misma frase) |
| A6 | (Gabriela, menor) Precisar cobertura temporal en resumen/conclusiones | Añadir explícitamente "Latinobarómetro aporta 13 olas irregulares entre 2007 y 2024" en vez de solo "entre los años 2007 y 2024", para no sugerir observaciones continuas todos los años | Resumen, Abstract, apertura de Cap. 6.1 |
| A7 | (Yo, hallazgo de la revisión general anterior, aún pendiente) Afirmación sobre el INEC y "solo la ronda de diciembre" | Ya se entregó el texto de reemplazo verificado en el turno anterior; falta confirmar que se copió en el documento | Cap. 5.5 (limitación 3) |
| A8 | (Yo, hallazgo de la revisión general anterior) Formato de decimales con coma en vez de punto | Corregir χ²=20,20→20.20, p=0,0005, χ²=5,31→5.31, p=0,2571, y los valores de la Tabla 5.4 + coeficientes de regresión logística (−0,632→−0.632, etc.) | Cap. 5.3.1, Cap. 5.4.2–5.4.3, Cap. 6 |
| A9 | (Yo, hallazgo de la revisión general anterior) Categorización inconsistente de Regresión Logística | Unificar si se le trata como uno de los "tres baselines de machine learning" (como en Cap. 5) o como categoría aparte (como en Cap. 1, 2 y 4); recomendado alinear todo con el uso del Cap. 5 | Cap. 1.3.2, Cap. 2.2.3, Cap. 4.1 |
| A10 | (Yo, hallazgo de la revisión general anterior) Frase con palabra suelta "...del modelo data leakage." | Agregar puntuación/paréntesis: "...del modelo (fuga de información, *data leakage*)." | Cap. 2.4 |

## Nivel B — Aclaraciones y contenido adicional (sin nuevos experimentos, pero sí redacción sustancial)

| # | Observación | Acción concreta | Ubicación sugerida |
|---|---|---|---|
| B1 | (Rolando) Justificar CNN-LSTM frente a Temporal Fusion Transformer, TCN o Transformers puros | Redactar una subsección "Selección de la arquitectura secuencial" que argumente: (a) la ventana temporal real es corta (3 años por encuestado, ~13-18 años reales de historia en total), un régimen de datos donde los mecanismos de atención de TFT/Transformer no tienen suficiente señal para superar a una LSTM y sı́ tienen más riesgo de sobreajuste; (b) TCN no ofrece ventaja clara sobre Conv1D+LSTM en secuencias tan cortas y añade más hiperparámetros a ajustar sin datos para validarlos; (c) la arquitectura CNN-LSTM fue además un requisito fijado en la propuesta de tesis, pero eso no exime de justificar técnicamente por qué es adecuada para este volumen de datos | Cap. 2.3.2 o inicio del Cap. 4 |
| B2 | (Rolando) Reconciliar el título ("Temporal Deep Learning...") con que el deep learning no demostró superioridad | Añadir un párrafo explícito (Introducción o Conclusiones) que aclare que el aporte no es que el deep learning "gane", sino someter empíricamente la hipótesis de que la complejidad temporal ayuda — y que el resultado nulo/negativo es en sí mismo un hallazgo válido y ya se discute en el Cap. 6, pero conviene anticiparlo desde el título/introducción para que no se lea como una promesa incumplida | Cap. 1.2 (Planteamiento) o Cap. 6.5 (Reflexión final) |
| B3 | (Gabriela, menor) Tabla de configuración experimental reproducible | Construir una tabla (posible Apéndice B) con: hiperparámetros finales de los 5 modelos, semilla aleatoria (random_state=42), k_neighbors de SMOTENC, versiones de librerías (xgboost, lightgbm, scikit-learn, imbalanced-learn, pytorch/pytorch-tabnet), y el criterio exacto de preprocesamiento ya descrito en el Cap. 4.3.5 | Nuevo Apéndice o Cap. 4.3.5 ampliado |
| B4 | (Gabriela, menor) Tamaño efectivo del dataset tras excluir respuestas inválidas del target | Añadir una nota/tabla con: 15,600 encuestados totales → N después de excluir target inválido → N de entrenamiento/prueba por fold (estos valores ya existen en los CSV de resultados, `n_train`/`n_test`; falta exponerlos en el documento) | Cap. 4.2.1 o Tabla 5.3 (agregar columnas n_train/n_test) |
| B5 | (Rolando) Discusión explícita de costo computacional (no solo tiempo) | Ampliar el párrafo ya existente sobre tiempos de entrenamiento (Cap. 5.3) con una frase sobre qué modelos requirieron GPU (CNN-LSTM, TabNet) frente a los que corrieron solo en CPU (XGBoost, LightGBM, Regresión Logística), y la implicación práctica para entornos con recursos limitados | Cap. 5.3 (ya tiene la nota de hardware; falta la discusión de RAM/GPU) |
| B6 | (Gabriela, menor) Legibilidad de variables en figuras SHAP | Definir un diccionario de etiquetas descriptivas (`econ_situation_cat` → "Percepción económica personal", `v2xeg_eqprotec` → "Protección igualitaria (V-Dem)", etc.) y regenerar las figuras SHAP con esas etiquetas en los ejes, manteniendo el nombre técnico en la Tabla 4.2/4.3/4.4 o en un anexo | `src/evaluation/shap_explicabilidad.py` (generación de figuras) + regenerar 34 figuras afectadas |

## Nivel C — Requieren nuevos experimentos o cambios de código con reentrenamiento

Estas son las de mayor esfuerzo; conviene decidir con la tutora el alcance mínimo aceptable antes de implementarlas.

| # | Observación | Acción concreta | Esfuerzo |
|---|---|---|---|
| C1 | (Gabriela, mayor) Comparación injusta CNN-LSTM vs. modelos tabulares (usa rezagos de Latinobarómetro que los demás no tienen) | Entrenar y reportar una variante "CNN-LSTM sin rezagos agregados de Latinobarómetro" (mismo panel de V-Dem/ENEMDU que los demás modelos), y presentarla junto a la versión actual, dejando explícito cuál es la comparación principal y cuál es el experimento adicional. El script `rezagos_macro.py` ya mencionado en la guía de ejecución hace una comparación de naturaleza similar (con/sin rezagos) y puede adaptarse | Medio-alto: nueva corrida de CNN-LSTM (8 folds) + tabla comparativa + prosa |
| C2 | (Gabriela, mayor) Cuantificar el aporte de integrar las 3 fuentes | Análisis de ablación con el mejor modelo tabular (XGBoost): entrenar y comparar 4 variantes de features — solo Latinobarómetro; Latinobarómetro+V-Dem; Latinobarómetro+ENEMDU; las tres fuentes — sobre los mismos 8 folds, reportando PR-AUC/F1 | Alto: 4 corridas completas de XGBoost (32 entrenamientos fold×variante) + tabla + prosa nueva, probablemente una subsección nueva en el Cap. 5 |
| C3 | (Gabriela, mayor) SMOTENC en escenario temporal — riesgo de combinaciones sintéticas de contexto irreales | Dos acciones posibles, no excluyentes: (a) correr una variante de cada modelo con `class_weight`/`scale_pos_weight` en vez de SMOTENC y comparar resultados; (b) si se mantiene SMOTENC, restringir la búsqueda de vecinos al mismo año (`anio`) dentro de `X_train`, o documentar explícitamente por qué la interpolación entre años cercanos del mismo fold no distorsiona el contexto (los valores de V-Dem/ENEMDU cambian poco de un año al siguiente) | Alto: cambio en `src/features/smote_train.py` y/o nuevas corridas comparativas para los 5 modelos |
| C4 | (Gabriela, mayor) Reformular la interpretación de las pruebas de Wilcoxon/Holm | Reemplazar "indistinguibles"/"equivalentes" por una formulación más cauta ("no se detectó una diferencia significativa con la potencia estadística disponible"), y calcular tamaños de efecto (p. ej. correlación rank-biserial para Wilcoxon) o diferencias pareadas con intervalo de confianza bootstrap, añadidos a la Tabla 5.2 | Medio: extender `src/evaluation/pruebas_estadisticas.py` para calcular tamaños de efecto + reescribir la Sección 5.3.1 |
| C5 | (Gabriela, mayor) Replantear la neutralización automática de SHAP > 10 | Decidir un enfoque menos invasivo: en vez de forzar a cero, considerar (a) aumentar el número de muestras de perturbación de KernelSHAP en los folds afectados, (b) excluir esas columnas del resumen global de ese fold con una nota explícita en vez de alterar su valor, o (c) reportar cuántos folds/columnas fueron afectados como una limitación cuantificada en vez de una corrección silenciosa | Medio-alto: cambio en `src/evaluation/shap_explicabilidad.py`, posible regeneración de tablas/figuras SHAP para CNN-LSTM/TabNet |

---

## Sobre las observaciones de Rolando ya resueltas en la versión actual

Dos de sus observaciones parecen estar contestadas en la versión más reciente del documento, posiblemente porque revisó una versión anterior:

- "No se incluye la configuración del hardware" → la Tabla 5.1 ya incluye una nota con el hardware usado (AMD Ryzen 9 9900X3D, 123 GiB RAM, GPU RTX 4090 24 GiB).
- "No se expresa si existió optimización de hiperparámetros" → la Sección 5.2 ya aclara explícitamente que no hubo búsqueda automática por fold, sino un único ajuste conservador decidido a priori.

Conviene confirmar con él que está evaluando la versión final antes de invertir tiempo en estos dos puntos.

---

## Orden sugerido de trabajo

1. Nivel A completo (todo es edición de texto, se puede hacer en una sola sesión).
2. B3, B4, B5 (no requieren reentrenar nada, solo exponer información que ya existe o es fácil de calcular).
3. B1, B2 (redacción argumentativa, sin código).
4. B6 (cambio de código pero mecánico, sin nueva lógica de modelado).
5. C4 (extiende un script ya existente, no reentrena modelos).
6. C1, C2, C3, C5 en el orden que decidas con la tutora — son los que más tiempo de cómputo y de redacción nueva van a tomar, y varios de ellos podrían cambiar cifras ya reportadas en la Tabla 5.1/5.3, así que conviene agruparlos en una sola ronda de reentrenamiento en vez de ir actualizando la tabla comparativa varias veces.

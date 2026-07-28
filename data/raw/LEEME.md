# Carpeta raw

Contiene los resultados de las consultas SQL (Superset) para V-Dem, ENEMDU y el diccionario de provincias, exportados como CSV, más las descargas directas de Latinobarómetro. Nombres exactos esperados por los scripts de `src/data_prep/`:

```
data/raw/
├── vdem_ecuador.csv                     <- resultado de sql/01_vdem_ecuador.sql
├── diccionario_provincias.csv           <- resultado de sql/03_diccionario_provincias.sql
├── enemdu_persona/
│   ├── enemdu_persona_200706.csv        <- un archivo por periodo (sql/04_...)
│   ├── enemdu_persona_200709.csv
│   ├── ...
│   ├── enemdu_persona_201712_pagina1.csv  <- periodos grandes: varias páginas (sql/04b_...)
│   └── enemdu_persona_201712_pagina2.csv
├── latinobarometro_codebook_oficial.xlsx  <- crosswalk oficial 
└── latinobarometro/                     <- descargas DIRECTAS de Latinobarómetro (NO Superset)
    ├── Latinobarometro_2007_Ecuador_Csv_esp_v1.csv   (... 2008, 2009, 2010: CSV crudo)
    └── Latinobarometro_2011_Ecuador_Spss_esp_v1.sav  (... 2013...2024: SPSS .sav)
```

El script enemdu_processing.py es el encargado de juntar automáticamente los archivos de `enemdu_persona/`, siempre que empiece con `enemdu_persona_` y termine en `.csv`

**Latinobarómetro:** la extracción vía Superset (`sql/02_latinobarometro_ecuador.sql`) presentó inconsistencias y no se pudo usar; se conserva solo como registro histórico. Los datos de Latinobarómetro se descargan directamente del sitio oficial y se procesan con `src/data_prep/latinobarometro_loader_crudo.py` (ver docstring de ese módulo para el detalle de por qué se mezclan dos formatos, CSV y SPSS).

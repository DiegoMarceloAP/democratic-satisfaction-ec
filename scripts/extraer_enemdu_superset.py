"""
extraer_enemdu_superset.py
---------------------------------------------------------------------
Automatiza la extracción de indicadores.enemdu_persona por periodo desde
la API REST de Superset, para no tener que reemplazar '${PERIODO}' a
mano y correr ~123 consultas una por una en SQL Lab (117 periodos, 6 de
ellos con 2 páginas cada uno por superar 100,000 filas).

TU ACCESO: solo usuario/contraseña por el navegador (confirmado). Esto
SÍ es suficiente para usar la API de Superset -- el login programático
usa el mismo mecanismo de autenticación que el navegador, no requiere
que un administrador te habilite nada especial. Lo que no puedo
garantizar sin probarlo contra tu instalación real es:
  - Que tu Superset no use SSO/2FA (si el login es "usuario+contraseña
    normal" en la pantalla de Superset, debería funcionar; si te
    redirige a Google/Azure/etc., este método NO va a funcionar).
  - La versión exacta de la API (este script usa los endpoints
    estándar de Superset >= 2.0: /api/v1/security/login y
    /api/v1/sqllab/execute/; si el Superset de destino es más viejo, el
    endpoint legado es /superset/sql_json/ -- si el endpoint estándar
    devuelve 404, adaptar la constante de endpoint más abajo).

SOBRE EL DATABASE_ID (perfil de solo consulta, sin panel de administración):
  El DATABASE_ID es un dato interno de SUPERSET (qué conexión usar), no
  de ClickHouse -- por eso ninguna sentencia SQL contra 'indicadores' o
  contra 'system.databases' puede revelarlo: ese tipo de consulta solo
  confirma que el esquema 'indicadores' existe dentro de ClickHouse,
  pero no tiene relación con el ID que usa Superset para identificar
  esa conexión.
  Alternativa sin acceso de administrador: deja SOLO_LISTAR_BASES = True
  (ver más abajo) y corre el script una vez -- inicia sesión con tu
  usuario y le pide a la API de Superset la misma lista de bases de
  datos que ya ves en el dropdown "Database" de SQL Lab (la de tu
  captura, donde eliges "clickhouse"). Como esa lista SÍ te la muestra
  la interfaz normal de SQL Lab, tu usuario tiene permiso para pedirla
  también por API, aunque no tengas acceso al panel de administración.
  Si aun así la API te devuelve vacío o un error 403, la alternativa
  100% segura es: abre las herramientas de desarrollador del navegador
  (F12) > pestaña "Network", vuelve a seleccionar "clickhouse" en el
  dropdown de SQL Lab, y busca en las peticiones alguna a
  ".../api/v1/database/..." -- ahí viene el id en la respuesta.

CÓMO USAR:
  1) Completa SUPERSET_URL abajo. Para el DATABASE_ID, corre primero el
     script con SOLO_LISTAR_BASES = True (ver nota arriba) para
     descubrirlo, luego cámbialo a False y pon el id encontrado.
  2) Prueba primero con PRUEBA_RAPIDA = True (corre solo 1 consulta chica)
     para confirmar que el login funciona en tu instalación ANTES de
     lanzar las 123 consultas completas.
  3) NUNCA escribas tu contraseña en este archivo ni la pegues en el
     chat. El script la pide de forma oculta (getpass) al ejecutarlo, o
     la toma de la variable de entorno SUPERSET_PASSWORD si la
     exportaste tú mismo en tu terminal.
  4) Corre en tu terminal, parado en la raíz del proyecto (la carpeta
     que contiene este archivo dentro de scripts/):
         pip install requests
         python scripts/extraer_enemdu_superset.py

Qué hace:
  - Inicia sesión en Superset (usuario/contraseña) y obtiene el token de
    acceso + el token CSRF que exige la API para peticiones POST.
  - Por cada periodo (y cada página, para los 6 periodos que superan
    100k filas) ejecuta la consulta contra indicadores.enemdu_persona y
    guarda el resultado como CSV en data/raw/enemdu_persona/, con el
    mismo nombre de archivo que ya usa enemdu_processing.py.
  - Si un archivo ya existe, lo salta -- puedes interrumpir el script
    (Ctrl+C) y volver a correrlo después sin perder lo ya descargado.
  - Alcance temporal: por defecto solo extrae periodos hasta 2024
    (rango de la propuesta de tesis). Tu volumetría real ya trae datos
    hasta 202604; si decides extender el alcance de la tesis más allá
    de 2024, cambia INCLUIR_POSTERIORES_A_2024 a True.
"""
import csv
import getpass
import math
import os
import sys
import time
from pathlib import Path

import requests

# ======================================================================
# CONFIGURACIÓN -- ajusta esto a tu instalación de Superset
# ======================================================================

# URL base de Superset tal como la usas en el navegador, SIN slash final.
# Ejemplo: "https://superset.tuinstitucion.edu.ec"
SUPERSET_URL = os.environ.get("SUPERSET_URL", "http://172.20.9.55:8088")

# ID numérico de la conexión a ClickHouse dentro de Superset.
# Cómo encontrarlo: en Superset ve a Configuración > Bases de datos
# (o "Data > Databases"), pasa el mouse/haz clic en "Editar" sobre la
# conexión que apunta al esquema 'indicadores' -- el ID aparece en la
# URL, algo como .../databases/edit/7  ->  el ID es 7.
DATABASE_ID = int(os.environ.get("SUPERSET_DATABASE_ID", "1"))  # confirmado: id=1 -> "clickhouse"

SQL_SCHEMA = os.environ.get("SUPERSET_SCHEMA", "indicadores")
USERNAME = os.environ.get("SUPERSET_USERNAME") or input("Usuario de Superset: ")

# Proveedor de autenticación que espera Superset en el login. "db" es el
# más común (usuario/contraseña guardados en la propia base de Superset).
# Si tu organización usa LDAP/Active Directory detrás del formulario de
# login (aunque visualmente sea igual: usuario+contraseña), puede que
# necesites "ldap" aquí en vez de "db" -- pregúntale a quien administra
# Superset si no lo sabes.
AUTH_PROVIDER = os.environ.get("SUPERSET_AUTH_PROVIDER", "db")

# Ya se identificó el DATABASE_ID (1 -> "clickhouse"), así que esto queda
# en False para pasar a probar la extracción real. Si alguna vez agregas
# otra conexión y necesitas volver a listar bases de datos, ponlo en True.
SOLO_LISTAR_BASES = False

PRUEBA_RAPIDA = False  # deja True hasta confirmar que el login funciona; luego cambia a False
INCLUIR_POSTERIORES_A_2024 = False  # True si decides extender el alcance de la tesis

OUT_DIR = Path("data/raw/enemdu_persona")

# IMPORTANTE -- descubierto empíricamente al correr esto (no estaba
# documentado de antemano): el límite de 100,000 filas que nos habían
# indicado aplica a la exportación/descarga de CSV desde la interfaz de
# SQL Lab, pero el endpoint de la API que este script usa para ejecutar
# consultas de forma síncrona (/api/v1/sqllab/execute/) tiene su PROPIO
# límite, más chico, para lo que devuelve inline en la respuesta JSON
# (el conocido SQLLAB_DEFAULT_DBAPI_ROW_LIMIT de Superset, que en la
# instalación real de este proyecto resultó ser 10,000). Por eso el primer intento con
# páginas de 100,000 devolvió silenciosamente solo 10,000 filas por
# archivo, sin error -- Superset simplemente corta ahí, no avisa. Se usan
# páginas de 10,000 para esta vía de extracción (API, no descarga de CSV).
FILAS_POR_PAGINA_API = 10000

# Volumetría real (periodo, n_filas), obtenida de 00_verificacion_volumetria.sql. Se usa
# solo para decidir cuántas páginas necesita cada periodo -- la consulta
# real vuelve a contar las filas en el servidor, esto no se asume ciego.
VOLUMETRIA = [
    ("200706", 26774), ("200709", 26180), ("200712", 76922), ("200803", 26161), ("200806", 37869), ("200809", 26339),
    ("200812", 78742), ("200903", 26439), ("200906", 26772), ("200909", 25376), ("200912", 78878), ("201003", 24958),
    ("201006", 79232), ("201009", 25584), ("201012", 82774), ("201103", 25580), ("201106", 80504), ("201109", 23924),
    ("201112", 69653), ("201203", 23496), ("201206", 71183), ("201209", 23200), ("201212", 73686), ("201303", 24011),
    ("201306", 77521), ("201309", 23939), ("201312", 81386), ("201403", 58711), ("201406", 115298), ("201409", 59312),
    ("201412", 116505), ("201503", 60265), ("201506", 114989), ("201509", 58444), ("201512", 112821), ("201603", 57951),
    ("201606", 57997), ("201609", 59354), ("201612", 114086), ("201703", 59242), ("201706", 58888), ("201709", 57329),
    ("201712", 110283), ("201803", 59348), ("201806", 59958), ("201809", 59736), ("201812", 59350), ("201903", 60173),
    ("201906", 60417), ("201909", 60065), ("201912", 59208), ("202009", 30317), ("202010", 30997), ("202011", 30790),
    ("202012", 30646), ("202101", 29804), ("202102", 29523), ("202103", 29502), ("202104", 30414), ("202105", 30339),
    ("202106", 30598), ("202107", 29925), ("202108", 30092), ("202109", 30424), ("202110", 30814), ("202112", 30026),
    ("202201", 30585), ("202202", 30087), ("202203", 30127), ("202204", 30031), ("202205", 29641), ("202206", 29734),
    ("202207", 30210), ("202208", 29942), ("202209", 29855), ("202210", 29722), ("202211", 29169), ("202212", 28993),
    ("202301", 29216), ("202302", 29302), ("202303", 28881), ("202304", 28851), ("202305", 29155), ("202306", 28757),
    ("202307", 28245), ("202308", 28680), ("202309", 28575), ("202310", 28718), ("202311", 28488), ("202312", 28306),
    ("202401", 28637), ("202402", 28689), ("202403", 28304), ("202404", 28606), ("202405", 28432), ("202406", 28670),
    ("202407", 28696), ("202408", 28598), ("202409", 28860), ("202410", 28028), ("202411", 28264), ("202412", 27610),
    ("202501", 28320), ("202502", 28303), ("202503", 27932), ("202504", 28092), ("202505", 27903), ("202506", 28072),
    ("202507", 27509), ("202508", 27580), ("202509", 27332), ("202511", 28126), ("202512", 27808), ("202601", 27528),
    ("202602", 27699), ("202603", 27667), ("202604", 26867),
]

COLUMNAS_SQL = (
    "periodo, id_persona, id_hogar, id_vivienda, ciudad, area, zona, sector, "
    "estrato, condact, empleo, desempleo, secemp, rama1, grupo1, ingpc, ingrl, "
    "pobreza, epobreza, fexp, p03, nnivins"
)


def construir_tareas():
    """A partir de la volumetría, genera la lista de (periodo, offset, nombre_archivo, filas_esperadas)."""
    tareas = []
    for periodo, n in VOLUMETRIA:
        if not INCLUIR_POSTERIORES_A_2024 and int(periodo[:4]) > 2024:
            continue
        n_paginas = math.ceil(n / FILAS_POR_PAGINA_API)
        for pagina in range(n_paginas):
            offset = pagina * FILAS_POR_PAGINA_API
            filas_esperadas = min(FILAS_POR_PAGINA_API, n - offset)
            nombre = (
                f"enemdu_persona_{periodo}.csv"
                if n_paginas == 1
                else f"enemdu_persona_{periodo}_pagina{pagina + 1}.csv"
            )
            tareas.append((periodo, offset, nombre, filas_esperadas))
    if PRUEBA_RAPIDA:
        tareas = tareas[:1]
    return tareas


def _fallar_con_detalle(resp: requests.Response, paso: str):
    """
    Imprime el cuerpo completo de la respuesta antes de fallar. Un 403 de
    Flask-AppBuilder/Flask-WTF casi siempre trae un mensaje específico en
    el cuerpo (ej. 'CSRF token missing', 'referrer header invalid',
    'Forbidden'), y ese mensaje es la única forma de saber si el problema
    es de permisos, de CSRF/Referer, o de otra cosa -- no tiene sentido
    seguir probando arreglos a ciegas sin verlo primero.
    """
    print(f"\n--- Detalle del error en '{paso}' ---")
    print(f"Status: {resp.status_code}")
    print(f"Headers de respuesta relevantes: "
          f"{ {k: v for k, v in resp.headers.items() if k.lower() in ('content-type','www-authenticate','server')} }")
    print(f"Cuerpo de la respuesta:\n{resp.text[:2000]}")
    print("--- fin del detalle ---\n")
    resp.raise_for_status()


def iniciar_sesion(password: str) -> requests.Session:
    resp = requests.post(
        f"{SUPERSET_URL}/api/v1/security/login",
        json={"username": USERNAME, "password": password, "provider": AUTH_PROVIDER, "refresh": True},
        headers={"Referer": SUPERSET_URL},
        timeout=30,
    )
    if not resp.ok:
        _fallar_con_detalle(resp, "POST /api/v1/security/login")
    access_token = resp.json()["access_token"]
    print("  -> login OK, access_token obtenido.")

    # El 403 en csrf_token que ya viste es "Forbidden" genérico de
    # Flask-AppBuilder -- es decir, tu rol no tiene el permiso para ESE
    # endpoint puntual, no es un problema de CSRF/Referer mal armado. En
    # vez de fallar aquí, seguimos sin el token CSRF: las peticiones GET
    # (como listar bases de datos) no lo necesitan, y vamos a probar
    # aparte si la petición POST de extracción lo exige de verdad o si
    # Superset la exime al venir autenticada con Bearer token.
    csrf_token = None
    csrf_resp = requests.get(
        f"{SUPERSET_URL}/api/v1/security/csrf_token/",
        headers={"Authorization": f"Bearer {access_token}", "Referer": SUPERSET_URL},
        timeout=30,
    )
    if csrf_resp.ok:
        csrf_token = csrf_resp.json()["result"]
        print("  -> csrf_token OK.")
    else:
        print(
            f"  -> AVISO: no se pudo obtener csrf_token (status {csrf_resp.status_code}, "
            f"cuerpo: {csrf_resp.text[:200]}). Se continúa sin él; puede que la extracción "
            "falle más adelante si Superset SÍ lo exige para las peticiones POST."
        )

    session = requests.Session()
    headers = {"Authorization": f"Bearer {access_token}", "Referer": SUPERSET_URL}
    if csrf_token:
        headers["X-CSRFToken"] = csrf_token
    session.headers.update(headers)
    return session


def listar_bases_de_datos(session: requests.Session):
    """
    Pide a Superset la lista de conexiones a bases de datos que tu
    usuario puede ver (el mismo listado que llena el dropdown "Database"
    de SQL Lab -- por eso, si puedes elegir "clickhouse" ahí, tu usuario
    SÍ tiene permiso para ver este listado por API, aunque tu rol sea de
    solo consulta y no tengas acceso al panel de administración).
    """
    resp = session.get(f"{SUPERSET_URL}/api/v1/database/", params={"q": "(page_size:100)"}, timeout=30)
    if not resp.ok:
        _fallar_con_detalle(resp, "GET /api/v1/database/")
    resultados = resp.json().get("result", [])
    if not resultados:
        print("La API no devolvió ninguna base de datos (revisar permisos).")
        return
    print("\nBases de datos visibles para tu usuario:")
    for r in resultados:
        print(f"  id={r['id']:<4} nombre={r.get('database_name', r.get('name'))}")
    print(
        "\nBusca en esta lista la fila cuyo nombre es 'clickhouse' (la misma que ves en el "
        "dropdown 'Database' de SQL Lab) y usa ese 'id' como DATABASE_ID."
    )


def ejecutar_consulta(session: requests.Session, periodo: str, offset: int):
    sql = (
        f"SELECT {COLUMNAS_SQL} FROM {SQL_SCHEMA}.enemdu_persona "
        f"WHERE periodo = '{periodo}' ORDER BY id_persona "
        f"LIMIT {FILAS_POR_PAGINA_API} OFFSET {offset}"
    )
    resp = session.post(
        f"{SUPERSET_URL}/api/v1/sqllab/execute/",
        json={"database_id": DATABASE_ID, "sql": sql, "schema": SQL_SCHEMA, "runAsync": False},
        timeout=180,
    )
    if not resp.ok:
        _fallar_con_detalle(resp, "POST /api/v1/sqllab/execute/")
    payload = resp.json()
    columnas = [c["name"] for c in payload["columns"]]
    return columnas, payload["data"]


def guardar_csv(path: Path, columnas, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)


def main():
    if "COMPLETAR" in SUPERSET_URL:
        sys.exit("Falta SUPERSET_URL. Complétalo al inicio del archivo antes de correr esto.")
    if DATABASE_ID == 0 and not SOLO_LISTAR_BASES:
        sys.exit(
            "Falta DATABASE_ID. Si no lo conoces, deja SOLO_LISTAR_BASES = True y corre el "
            "script para que te lo muestre (ver nota 'SOBRE EL DATABASE_ID' en el docstring)."
        )

    if SOLO_LISTAR_BASES:
        password = os.environ.get("SUPERSET_PASSWORD") or getpass.getpass("Contraseña de Superset (no se muestra en pantalla): ")
        try:
            session = iniciar_sesion(password)
        except requests.HTTPError as e:
            sys.exit(f"No se pudo iniciar sesión ({e}).")
        listar_bases_de_datos(session)
        return

    tareas = construir_tareas()
    print(f"Total de consultas a ejecutar: {len(tareas)}" + (" (PRUEBA_RAPIDA=True: solo 1)" if PRUEBA_RAPIDA else ""))

    password = os.environ.get("SUPERSET_PASSWORD") or getpass.getpass("Contraseña de Superset (no se muestra en pantalla): ")
    try:
        session = iniciar_sesion(password)
    except requests.HTTPError as e:
        sys.exit(
            f"No se pudo iniciar sesión ({e}). Posibles causas: tu Superset usa SSO/2FA "
            "(este método no funciona en ese caso), o la URL/endpoint no es la esperada. Avísame el error exacto."
        )
    print("Login OK.")

    filas_por_periodo = {}
    for i, (periodo, offset, nombre, filas_esperadas) in enumerate(tareas, start=1):
        destino = OUT_DIR / nombre
        if destino.exists():
            with open(destino, encoding="utf-8") as f:
                n_en_disco = sum(1 for _ in f) - 1  # -1 por el encabezado
            filas_por_periodo[periodo] = filas_por_periodo.get(periodo, 0) + n_en_disco
            print(f"[{i}/{len(tareas)}] {nombre} ya existe ({n_en_disco} filas), se salta.")
            continue
        print(f"[{i}/{len(tareas)}] periodo={periodo} offset={offset} -> {nombre} (esperadas: {filas_esperadas})")
        try:
            columnas, filas = ejecutar_consulta(session, periodo, offset)
        except requests.HTTPError as e:
            print(f"  ERROR en periodo {periodo} offset {offset}: {e}. Se continúa con el siguiente.")
            continue
        guardar_csv(destino, columnas, filas)
        filas_por_periodo[periodo] = filas_por_periodo.get(periodo, 0) + len(filas)
        if len(filas) != filas_esperadas:
            print(f"  AVISO: se esperaban {filas_esperadas} filas y llegaron {len(filas)} -- revisar.")
        else:
            print(f"  guardado: {len(filas)} filas (OK)")
        time.sleep(0.5)  # no saturar el servidor

    # Verificación final: suma de filas por periodo (todas sus páginas)
    # contra la volumetría real que compartiste -- esto es justamente lo
    # que hubiera detectado antes el problema de las páginas de 10,000.
    if PRUEBA_RAPIDA:
        print(
            "\n(PRUEBA_RAPIDA=True: la verificación de abajo va a marcar 'DISCREPANCIA' para el "
            "periodo probado porque solo se pidió 1 de sus páginas a propósito -- es esperado, no es un error.)"
        )
    print("\n--- Verificación de totales por periodo ---")
    volumetria_dict = dict(VOLUMETRIA)
    hay_discrepancias = False
    periodos_procesados = list(dict.fromkeys(periodo for periodo, _, _, _ in tareas))  # únicos, en orden
    for periodo in periodos_procesados:
        esperado = volumetria_dict.get(periodo)
        obtenido = filas_por_periodo.get(periodo, 0)
        if esperado is not None and obtenido != esperado:
            hay_discrepancias = True
            print(f"  periodo {periodo}: esperado={esperado} obtenido={obtenido}  <-- DISCREPANCIA")
    if not hay_discrepancias:
        print("  Todos los periodos procesados coinciden con la volumetría esperada.")
    print("--- fin de la verificación ---\n")

    print("Listo.")
    if PRUEBA_RAPIDA:
        print("Era una PRUEBA_RAPIDA (1 sola consulta). Si salió bien, cambia PRUEBA_RAPIDA a False y vuelve a correr.")


if __name__ == "__main__":
    main()

-- =====================================================================
-- 03_diccionario_provincias.sql
-- -----------------------------------------------------------------------
-- Fuente: indicadores.diccionario_provincias (tabla de dimensión
--         geográfica: provincia / cantón / parroquia, estándar INEC).
-- Uso: catálogo para traducir los códigos de 2/4/6 dígitos que se
--      derivan de 'ciudad' en ENEMDU (ver 04 y 05) a nombres oficiales
--      de provincia/cantón/parroquia, y como tabla de referencia para
--      homogeneizar la división político-administrativa a lo largo de
--      2007-2024 (hubo creación de provincias nuevas en el periodo:
--      Santo Domingo de los Tsáchilas y Santa Elena, 2007).
-- Volumen: tabla de dimensión, pequeña (universo de parroquias del
--          Ecuador, del orden de ~1,000 filas). No requiere partición.
-- =====================================================================

SELECT DISTINCT
    CodigoProvincia,
    NombreProvincia,
    CodigoCanton,
    NombreCanton,
    CodigoParroquia,
    NombreParroquia
FROM indicadores.diccionario_provincias
ORDER BY CodigoProvincia, CodigoCanton, CodigoParroquia
LIMIT 100000;

-- Vista reducida a nivel provincia únicamente (la granularidad que
-- necesitamos para el merge final con Latinobarómetro/V-Dem, según el
-- alcance definido en la propuesta de tesis):
--
-- SELECT DISTINCT CodigoProvincia, NombreProvincia
-- FROM indicadores.diccionario_provincias
-- ORDER BY CodigoProvincia;

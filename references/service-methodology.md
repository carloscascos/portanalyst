# Service Analysis - Metodología de Análisis

## Objetivo

Analizar en profundidad un servicio marítimo (liner service) desde su primera ocurrencia hasta la actualidad, incluyendo estructura, regularidad, operadores, alianzas y costes ETS.

---

## 1. Identificación del Servicio

### Datos Maestros (econdb tables)

```sql
-- Obtener metadatos del servicio
SELECT
    sls.id as service_id,
    sls.name as servicio,
    sls.duration as rotacion_dias,
    sls.active_vessels,
    sls.teu_avg,
    sls.trade_lane,
    sls.is_feeder,
    sls.is_regular,
    sls.start_date,
    sls.end_date
FROM econdb_shared_line_service sls
WHERE sls.name LIKE '%[SERVICIO]%'
```

### Operadores y Alianzas

```sql
-- Quién opera el servicio
SELECT
    sls.name as servicio,
    ls.operator,
    ls.alliance,
    ls.code as operator_code,
    ls.first_voyage,
    ls.last_voyage
FROM econdb_shared_line_service sls
JOIN econdb_line_service ls ON ls.shared_line_service_id = sls.id
WHERE sls.name LIKE '%[SERVICIO]%'
```

**Output esperado:**

| Campo | Descripción |
|-------|-------------|
| Servicio | Nombre del shared line service |
| Operadores | Lista de carriers que participan |
| Alianza | Gemini, Ocean Alliance, THE Alliance, etc. |
| Rotación | Días para completar un loop |
| Buques activos | Número de vessels asignados |
| TEU medio | Capacidad promedio de la flota |
| Trade lane | FEA/EUR, Transatlantic, etc. |
| Vida | first_voyage → last_voyage |
| **Frecuencia** | Un buque cada X días |

---

## 2. Frecuencia del Servicio

### Concepto

La frecuencia mide cada cuántos días llega un buque a un puerto determinado. Es la métrica comercial más importante para un cargador.

```
Frecuencia = días_actividad / escalas
```

### Puerto de Referencia

Calcular la frecuencia en el **puerto de origen principal** o **puerto de destino principal**, NO en puertos hub que se visitan múltiples veces por rotación.

**Criterios para elegir puerto de referencia:**
- Puerto con más escalas que NO sea hub (ratio ≈ 1.0)
- En servicios Asia→Europa: usar último puerto asiático o primer puerto europeo
- En servicios Asia→USA: usar último puerto asiático o primer puerto USA

### Query de Frecuencia

```sql
-- Calcular frecuencia en puertos principales
SELECT
    e.portname,
    p.country,
    COUNT(*) as escalas,
    MIN(e.start) as primera_escala,
    MAX(e.start) as ultima_escala,
    DATEDIFF(MAX(e.start), MIN(e.start)) as dias_actividad,
    ROUND(DATEDIFF(MAX(e.start), MIN(e.start)) / COUNT(*), 1) as dias_entre_escalas
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND e.portname IN ('[PUERTO_ORIGEN]', '[PUERTO_DESTINO]')
GROUP BY e.portname, p.country
ORDER BY escalas DESC
```

### Interpretación de Frecuencias

| Frecuencia | Clasificación | Uso típico |
|------------|---------------|------------|
| ~3-4 días | Muy alta | Feeders, short-sea |
| ~7 días | **Semanal** | Servicios mainline estándar |
| ~10-14 días | Quincenal | Servicios secundarios |
| >14 días | Baja | Servicios de nicho |

### Output en Ficha

```
Frecuencia: 1 buque cada 8 días (semanal) en Long Beach
```

---

## 3. Fiabilidad del Servicio (Schedule Reliability)

### Concepto

La fiabilidad mide la consistencia del servicio: cuántas veces el buque llega dentro de la tolerancia esperada respecto al intervalo teórico.

**Nota:** No disponemos de horarios programados en la base de datos. Usamos la **consistencia de intervalos** como proxy de fiabilidad.

```
Intervalo esperado = frecuencia calculada (días entre escalas)
Desviación = |intervalo_real - intervalo_esperado|
Fiabilidad = % de intervalos dentro de tolerancia
```

### Puerto de Medición

Medir en el **puerto gateway europeo** (primer puerto EUROMED en servicios EXTRA):
- Servicios Asia→Europa: Primer puerto europeo (ej: Tanger Med, Piraeus, Algeciras)
- Servicios USA→Europa: Primer puerto europeo
- Servicios INTRA: Puerto principal del loop

**IMPORTANTE:** En puertos hub que se visitan dos veces por rotación (entrada + retorno), filtrar solo las escalas donde `prev_port.zone` está **fuera de EUROMED**. Esto asegura medir solo la llegada inicial desde Asia/USA, no el retorno desde North Europe.

### Query de Fiabilidad

```sql
-- Calcular fiabilidad en puerto gateway europeo (solo entradas desde fuera EUROMED)
WITH intervalos AS (
    SELECT
        e.portname,
        e.start,
        LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start) as escala_anterior,
        TIMESTAMPDIFF(HOUR,
            LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start),
            e.start) as intervalo_horas
    FROM escalas e
    JOIN econdb_ship_service_ranges s ON e.imo = s.imo
        AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
    JOIN ports p_prev ON e.prev_port = p_prev.portname
    WHERE s.sls_name LIKE '%[SERVICIO]%'
      AND e.portname = '[PUERTO_GATEWAY]'
      -- Solo entradas desde fuera de EUROMED (evita contar retornos desde N.Europe)
      AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
),
stats AS (
    SELECT
        portname,
        COUNT(*) as total_intervalos,
        ROUND(AVG(intervalo_horas), 1) as intervalo_medio_h,
        ROUND(STDDEV(intervalo_horas), 1) as desviacion_std_h
    FROM intervalos
    WHERE intervalo_horas IS NOT NULL
    GROUP BY portname
)
SELECT
    i.portname,
    s.total_intervalos,
    ROUND(s.intervalo_medio_h / 24, 1) as frecuencia_dias,
    ROUND(s.desviacion_std_h / 24, 2) as std_dias,
    -- Fiabilidad ±6 horas
    ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 6 THEN 1 ELSE 0 END)
          / s.total_intervalos, 1) as fiab_6h_pct,
    -- Fiabilidad ±12 horas
    ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 12 THEN 1 ELSE 0 END)
          / s.total_intervalos, 1) as fiab_12h_pct,
    -- Fiabilidad ±24 horas
    ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 24 THEN 1 ELSE 0 END)
          / s.total_intervalos, 1) as fiab_24h_pct
FROM intervalos i
JOIN stats s ON i.portname = s.portname
WHERE i.intervalo_horas IS NOT NULL
GROUP BY i.portname, s.total_intervalos, s.intervalo_medio_h, s.desviacion_std_h
```

### Interpretación de Fiabilidad

| Fiabilidad ±12h | Clasificación | Descripción |
|----------------:|---------------|-------------|
| ≥ 90% | **Excelente** | Servicio muy predecible |
| 70-89% | **Buena** | Servicio fiable con variaciones ocasionales |
| 50-69% | **Moderada** | Variabilidad significativa |
| < 50% | **Baja** ⚠️ | Servicio irregular o datos inconsistentes |

### Factores que Afectan la Fiabilidad

- **Congestión portuaria**: Retrasos en puertos congestionados
- **Meteorología**: Temporada de monzones, huracanes
- **Canal de Suez/Panamá**: Esperas por tráfico o restricciones
- **Blank sailings**: Cancelaciones por demanda baja
- **Cambios de flota**: Sustitución de buques

### Output en Ficha

```
Fiabilidad: 78% ±12h en Tanger Med (Buena)
Frecuencia: 1 buque cada 7.2 días | Std: ±0.8 días
```

---

## 4. Vida del Servicio y Rotaciones

### Regla: Mínimo 1 rotación completa

```
Periodo mínimo = rotacion_dias × 1.5
```

**Ejemplo AE3:** 45 días × 1.5 = 68 días mínimo

### Determinación automática

```sql
-- Vida del servicio desde escalas reales
SELECT
    MIN(e.start) as primera_escala,
    MAX(e.start) as ultima_escala,
    DATEDIFF(MAX(e.start), MIN(e.start)) as dias_actividad,
    COUNT(DISTINCT e.imo) as buques_historicos
FROM escalas e
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
```

---

## 5. Puertos y Frecuencia (Tabla Unificada)

### 5.1 Query Unificada

```sql
-- Tabla única: puertos + frecuencia + clasificación
SELECT
    e.portname,
    p.country,
    p.zone,
    COUNT(*) as escalas,
    DATEDIFF(MAX(e.start), MIN(e.start)) as dias,
    ROUND(DATEDIFF(MAX(e.start), MIN(e.start)) / COUNT(*), 1) as dias_entre_escalas,
    ROUND(COUNT(*) / [ROTACIONES_ESPERADAS], 2) as ratio
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND f.fleet = 'containers'
GROUP BY e.portname, p.country, p.zone
ORDER BY escalas DESC
```

### 5.2 Formato de Salida

| Puerto | País | Zona | Escalas | Frecuencia | Clasificación |
|--------|------|------|--------:|:----------:|---------------|
| [puerto] | [país] | [zona] | N | 1 cada X días | Regular/Ocasional/Esporádico |

### 5.3 Clasificación de Puertos

| Ratio | Clasificación | Descripción |
|------:|---------------|-------------|
| ≥ 0.5 | **Regular** | Puerto fijo del servicio |
| 0.2 - 0.5 | **Ocasional** | Port call adicional o desvío |
| < 0.2 | **Esporádico** ⚠️ | Revisar - posible error de asignación |

**Cálculo:** `ratio = escalas / max_escalas` (puerto con más escalas = referencia)

---

## 6. Análisis EUROMED

### 6.1 Clasificación del Servicio

```sql
-- Determinar si el servicio cruza boundary EUROMED
SELECT
    CASE
        WHEN COUNT(DISTINCT CASE WHEN p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED') THEN e.portname END) > 0
         AND COUNT(DISTINCT CASE WHEN p.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED') THEN e.portname END) > 0
        THEN 'EUROMED-EXTRA'
        WHEN COUNT(DISTINCT CASE WHEN p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED') THEN e.portname END) > 0
        THEN 'EUROMED-INTRA'
        ELSE 'NON-EUROMED'
    END as tipo_servicio,
    GROUP_CONCAT(DISTINCT p.zone ORDER BY p.zone) as zonas_tocadas
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
```

### 6.2 Puertos Gateway (Entrada/Salida EUROMED)

```sql
-- Identificar puertos donde el servicio cruza la frontera EUROMED
SELECT
    e.portname as gateway_port,
    p.zone,
    p.country,
    COUNT(*) as cruces,
    'ENTRADA' as direccion
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN ports p_prev ON e.prev_port = p_prev.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
GROUP BY e.portname, p.zone, p.country

UNION ALL

SELECT
    e.portname as gateway_port,
    p.zone,
    p.country,
    COUNT(*) as cruces,
    'SALIDA' as direccion
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN ports p_next ON e.next_port = p_next.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND p_next.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
GROUP BY e.portname, p.zone, p.country
```

---

## 7. Análisis ETS

### 7.1 Coste ETS por Rotación

```sql
SELECT
    YEAR(e.start) as año,
    QUARTER(e.start) as trimestre,
    COUNT(*) as escalas,
    ROUND(SUM(em.incoming_ets_cost_eur), 0) as ets_entrada,
    ROUND(SUM(em.outgoing_ets_cost_eur), 0) as ets_salida,
    ROUND(SUM(em.total_ets_cost_eur), 0) as ets_total,
    ROUND(AVG(em.total_ets_cost_eur), 0) as ets_medio_escala
FROM escalas e
JOIN v_escalas_metrics em ON e.imo = em.imo AND e.start = em.start
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND e.start >= '2024-01-01'  -- ETS activo desde 2024
GROUP BY YEAR(e.start), QUARTER(e.start)
ORDER BY año, trimestre
```

### 7.2 Evolución Temporal ETS

Comparar:
- Q1 2024 vs Q1 2025 (YoY)
- Tendencia trimestral
- Impacto del phase-in (40% → 70% → 100%)

---

## 8. Métricas Operacionales

### 8.1 Flota por Trimestre

```sql
SELECT
    YEAR(e.start) as año,
    QUARTER(e.start) as trimestre,
    COUNT(DISTINCT e.imo) as buques_activos,
    COUNT(*) as total_escalas,
    COUNT(DISTINCT e.portname) as puertos_tocados,
    ROUND(COUNT(*) / COUNT(DISTINCT e.imo), 1) as escalas_por_buque,
    ROUND(90 / (COUNT(*) / COUNT(DISTINCT e.portname)), 1) as frecuencia_dias
FROM escalas e
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
GROUP BY YEAR(e.start), QUARTER(e.start)
ORDER BY año, trimestre
```

### 8.2 Flota Desplegada

```sql
-- Incluir año de construcción y combustible
SELECT
    f.name as buque,
    f.imo,
    ROUND(f.teus, 0) as teus,
    f.category,
    f.Year as año_construccion,
    f.fuel,
    f.group as operador,
    COUNT(*) as escalas
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
  AND f.fleet = 'containers'
GROUP BY f.name, f.imo, f.teus, f.category, f.Year, f.fuel, f.group
ORDER BY escalas DESC
```

### 8.3 Formato de Salida Flota

| Buque | TEU | Categoría | Año | Fuel | Operador | Escalas |
|-------|----:|-----------|----:|------|----------|--------:|

**Nota:** Campo `fuel` puede ser NULL (convencional HFO/VLSFO) o indicar LNG, methanol, etc.

---

## 9. Detección de Anomalías

### 9.1 Escalas Esporádicas

Puertos con ratio < 0.2 → revisar si:
- Error de asignación de servicio
- Desvío por emergencia
- Port call adicional puntual
- Cambio de ruta temporal

### 9.2 Gaps en el Servicio

```sql
-- Detectar periodos sin actividad
SELECT
    e.portname,
    e.start as escala_actual,
    LEAD(e.start) OVER (PARTITION BY e.portname ORDER BY e.start) as siguiente_escala,
    DATEDIFF(LEAD(e.start) OVER (PARTITION BY e.portname ORDER BY e.start), e.start) as dias_gap
FROM escalas e
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%[SERVICIO]%'
HAVING dias_gap > [ROTACION_DIAS] * 2  -- Gap > 2 rotaciones
```

### 9.3 Rotaciones Incompletas

Las últimas `rotacion_dias` del análisis pueden tener rotaciones en curso.
Marcar con ⚠️ y excluir de métricas de regularidad.

---

## 10. Output del Service Analysis

### Informe Estructurado

1. **Ficha del Servicio**
   - Nombre, operadores, alianza
   - Trade lane, rotación, buques
   - **Frecuencia: 1 buque cada X días**
   - **Fiabilidad: XX% ±12h en [Puerto Gateway]**
   - Vida: primera → última escala

2. **Mapa de Puertos**
   - Tabla ordenada por escalas
   - Clasificación: Regular / Ocasional / Esporádico
   - Zonas EUROMED

3. **Análisis EUROMED**
   - Tipo: EXTRA / INTRA / NON-EUROMED
   - Gateways de entrada/salida
   - Impacto ETS

4. **Flota Desplegada**
   - Tabla con TEU, categoría, año, fuel, operador
   - Escalas por buque

5. **Alertas**
   - Puertos esporádicos (ratio < 0.2)
   - Gaps detectados
   - Rotaciones incompletas

---

## Tablas Utilizadas

| Tabla | Uso |
|-------|-----|
| `econdb_shared_line_service` | Metadatos del servicio compartido |
| `econdb_line_service` | Operadores y alianzas |
| `econdb_ship_service_ranges` | Asignación buque ↔ servicio por fecha |
| `escalas` | Port calls reales |
| `v_fleet` | Datos del buque (TEU, categoría) |
| `v_escalas_metrics` | Costes ETS y emisiones |
| `ports` | Zonas y clasificación EEA |

---

## 11. Análisis Comparativo de Períodos (P1 vs P2)

### Objetivo

Cuando se analiza un servicio que ha cambiado entre dos períodos, NO basta con describir el estado actual. Hay que **medir el delta** y **explicar qué lo causó**.

### 11.1 Métricas Obligatorias P1 vs P2

| Métrica | P1 | P2 | Δ | Interpretación |
|---------|---:|---:|--:|----------------|
| Escalas | | | | Cambio de frecuencia |
| TEU | | | | Cambio de capacidad |
| **KTM** | | | | **Cambio de distancia × capacidad** |
| Dist. media | | | | Cambio de ruta |

### 11.2 Query de Métricas por Período

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  COUNT(*) as escalas,
  SUM(f.teus) as teus,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000, 2) as ktm,
  ROUND(AVG(e.prev_leg)) as dist_media
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%'
  AND f.fleet = 'containers'
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period
```

### 11.3 Puertos Ganados/Perdidos

```sql
-- Puertos en P1 que no están en P2 (PERDIDOS)
SELECT e.portname, p.zone, COUNT(*) as escalas_p1
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%'
  AND e.start >= '{p1_start}' AND e.start < '{p1_end}'
  AND e.portname NOT IN (
    SELECT DISTINCT e2.portname
    FROM escalas e2
    JOIN econdb_ship_service_ranges s2 ON e2.imo = s2.imo
      AND e2.start BETWEEN s2.start_range AND COALESCE(s2.end_range, '2099-12-31')
    WHERE s2.sls_name LIKE '%{SERVICE}%'
      AND e2.start >= '{p2_start}' AND e2.start < '{p2_end}'
  )
GROUP BY e.portname, p.zone
ORDER BY escalas_p1 DESC
```

### 11.4 Interpretación de Cambios de Distancia

| Δ Dist. Media | Posible Causa |
|---------------|---------------|
| +30% a +100% | **Cierre de ruta** (Suez, Panamá) → ruta alternativa larga |
| +10% a +30% | Cambio de hub o añadido de puertos intermedios |
| -10% a -30% | Optimización de ruta o eliminación de escalas |
| -30% a -50% | Eliminación de leg largo (pérdida de conectividad intercontinental) |

**CRÍTICO:** Si desaparecen Port Said, Suez, Djibouti o puertos del Mar Rojo → investigar cierre de Suez.

---

## 12. Análisis ETS Comparativo (Hub Switch)

### Cuándo Activar

Cuando un servicio EUROMED-EXTRA cambia su **puerto gateway europeo** entre P1 y P2:
- Algeciras → Tangier Med
- Valencia → Port Said
- Piraeus → Port Said

### 12.1 Query ETS por Puerto Gateway

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  e.portname as gateway,
  p.EEA,
  COUNT(*) as cruces,
  ROUND(SUM(em.total_ets_cost_eur), 0) as ets_total,
  ROUND(AVG(em.total_ets_cost_eur), 0) as ets_medio
FROM escalas e
JOIN v_escalas_metrics em ON e.imo = em.imo AND e.start = em.start
JOIN ports p ON e.portname = p.portname
JOIN ports p_prev ON e.prev_port = p_prev.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%'
  AND e.portname IN ('{GATEWAY_P1}', '{GATEWAY_P2}')
  AND p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND e.start >= '2024-01-01'
GROUP BY period, e.portname, p.EEA
ORDER BY period, e.portname
```

### 12.2 Tabla de Output

| Gateway | EEA | Cruces P1 | ETS P1 (€) | Cruces P2 | ETS P2 (€) | Δ ETS |
|---------|----:|----------:|-----------:|----------:|-----------:|------:|
| Algeciras | 0.5 | 27 | €XXX | 0 | €0 | -100% |
| Tangier Med | 0.0 | 0 | €0 | 13 | €0 | — |

### 12.3 Interpretación

- **EEA = 0.5** (puertos EU): 50% coste ETS en 2024, 100% en 2025+
- **EEA = 0.0** (puertos no-EU): 0% coste ETS

**Cambio de gateway EEA → no-EEA = carbon leakage operativo**

---

## 13. Checklist Obligatorio Service Analysis

Antes de entregar un Service Analysis, verificar:

| Check | Query/Método | ¿Hecho? |
|-------|--------------|---------|
| Métricas P1 vs P2 (escalas, TEU, **KTM**, dist.) | Sección 11.2 | ☐ |
| Puertos ganados/perdidos | Sección 11.3 | ☐ |
| Si Δ dist. >30%: investigar causa (Suez, hub) | Sección 11.4 | ☐ |
| Si servicio EUROMED-EXTRA: ETS por gateway | Sección 12.1 | ☐ |
| Si cambio de gateway EEA→no-EEA: calcular ahorro | Sección 12.2 | ☐ |
| Flota desplegada con categoría y fuel | Sección 8.2 | ☐ |
| Fiabilidad en puerto gateway europeo | Sección 3 | ☐ |

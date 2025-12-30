# Query Templates

Reusable SQL patterns for port traffic analysis. All queries require `traffic-db` MCP server.

## Period Conversion

| Quarter | Start Date | End Date |
|---------|------------|----------|
| Q1 | `'{year}-01-01'` | `'{year}-04-01'` |
| Q2 | `'{year}-04-01'` | `'{year}-07-01'` |
| Q3 | `'{year}-07-01'` | `'{year}-10-01'` |
| Q4 | `'{year}-10-01'` | `'{next_year}-01-01'` |

---

## Golden Rule

```sql
⚠️ ALWAYS include in WHERE:
   f.fleet = 'containers' AND f.teus > 0

Without this filter, you'll include bulkers, tankers, etc.
```

---

## Data Quality Validation (Paso 0)

**Run BEFORE any analysis to validate data quality.**

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  COUNT(*) as calls,
  COUNT(DISTINCT DATE(e.start)) as days,
  ROUND(100.0 * SUM(CASE WHEN e.prev_port IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_null_prev,
  ROUND(100.0 * SUM(CASE WHEN e.prev_leg IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_null_dist
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period
```

**Thresholds:**
- Days differ >20% → **Normalize before continuing**
- Missing prev_port >15% → **Severe warning in report**
- Missing 5-15% → Confidence "Medium"
- Sample <20 calls in segment → **Warn "low sample size"**

---

## Macro Metrics

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  COUNT(*) as escalas,
  SUM(f.teus) as teus,
  ROUND(AVG(e.prev_leg)) as dist_media,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000) as ktm,
  ROUND(AVG(e.duration), 1) as horas_media
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.prev_leg IS NOT NULL
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period
```

---

## By Vessel Category

**For Checklist Diagnosis: Node Degradation pattern.**

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  f.category,
  COUNT(*) as escalas,
  SUM(f.teus) as teus
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, f.category
ORDER BY f.category, period
```

**Categories:**
| Cat | Name | TEU Range |
|-----|------|-----------|
| 1 | Feeder | < 3,000 |
| 2 | Panamax | 3,000-5,000 |
| 3 | Post Panamax | 5,000-10,000 |
| 4 | New Panamax | 10,000-15,000 |
| 5 | ULCV | > 15,000 |

---

## Mainline Analysis (teus > 10000)

### By Operator

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  f.group as operator,
  COUNT(*) as escalas,
  SUM(f.teus) as teus
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 10000
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, operator
ORDER BY operator, period
```

### By Service

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  s.line_svc_name,
  s.operator_name,
  COUNT(*) as escalas
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN v_econdb_service_escala s ON e.imo = s.imo AND e.start = s.start
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 10000
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, s.line_svc_name, s.operator_name
ORDER BY s.line_svc_name, period
```

### Vessel Tracing (Service Investigation)

**Vessels operating a route in P1:**

```sql
SELECT DISTINCT f.imo, f.shipname, f.teus, f.group, COUNT(*) as calls
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}' AND e.prev_port = '{origin_port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.start >= '{p1_start}' AND e.start < '{p1_end}'
GROUP BY f.imo, f.shipname, f.teus, f.group
ORDER BY calls DESC
```

**Where those IMOs operate in P2:**

```sql
SELECT e.prev_port, e.portname as next_port, p.zone, COUNT(*) as calls
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE f.imo IN ({imo_list})
  AND f.fleet = 'containers'
  AND e.start >= '{p2_start}' AND e.start < '{p2_end}'
GROUP BY e.prev_port, e.portname, p.zone
ORDER BY calls DESC
LIMIT 20
```

**New vessels in P2 (not in P1):**

```sql
SELECT f.imo, f.shipname, f.teus, f.group, COUNT(*) as calls_p2
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.start >= '{p2_start}' AND e.start < '{p2_end}'
  AND f.imo NOT IN (
    SELECT DISTINCT f2.imo FROM v_escalas e2
    JOIN v_fleet f2 ON e2.imo = f2.imo
    WHERE e2.portname = '{port}' AND f2.fleet = 'containers'
      AND e2.start >= '{p1_start}' AND e2.start < '{p1_end}'
  )
GROUP BY f.imo, f.shipname, f.teus, f.group
ORDER BY calls_p2 DESC
```

**Port substitution check:**

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  e.portname, COUNT(*) as calls
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE f.imo IN ({service_imos})
  AND e.portname IN ({candidate_ports})
  AND f.fleet = 'containers'
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, e.portname
ORDER BY e.portname, period
```

### By Zone (Origin)

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  p.zone,
  COUNT(*) as escalas,
  ROUND(AVG(f.teus)) as teu_medio,
  ROUND(AVG(e.prev_leg)) as dist_media
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.prev_port = p.portname
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 10000
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, p.zone
ORDER BY p.zone, period
```

---

## Feeder Analysis (teus <= 10000)

### By Operator

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  f.group as operator,
  COUNT(*) as escalas,
  SUM(f.teus) as teus
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0 AND f.teus <= 10000
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, operator
ORDER BY operator, period
```

---

## Distance by Zone (Route Alteration Check)

**For Checklist Diagnosis: Route Alteration pattern.**

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  p.zone,
  COUNT(*) as escalas,
  ROUND(AVG(e.prev_leg)) as dist_media
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.prev_port = p.portname
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, p.zone
HAVING COUNT(*) >= 5
ORDER BY p.zone, period
```

---

## Group Metrics (Aggregated)

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  COUNT(*) as escalas,
  SUM(f.teus) as teus,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000, 1) as ktm_m,
  COUNT(DISTINCT e.portname) as ports_count
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE {group_filter}  -- e.g., p.EEA = 0.00
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.prev_leg IS NOT NULL
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period
```

**Group filters:**
- `p.EEA = 0.00` — Non-EEA ports
- `p.EEA = 0.50` — EEA ports
- `p.zone = 'WEST MED'` — Zone filter
- `p.country = 'Spain'` — Country filter

---

## Inter-Group Flows (EEA Analysis)

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  CASE
    WHEN p_origin.EEA = 0 AND p_dest.EEA = 0 THEN 'non-EEA → non-EEA'
    WHEN p_origin.EEA = 0 AND p_dest.EEA = 0.5 THEN 'non-EEA → EEA'
    WHEN p_origin.EEA = 0.5 AND p_dest.EEA = 0 THEN 'EEA → non-EEA'
    WHEN p_origin.EEA = 0.5 AND p_dest.EEA = 0.5 THEN 'EEA → EEA'
  END as flow_type,
  COUNT(*) as escalas,
  SUM(f.teus) as teus
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p_origin ON e.prev_port = p_origin.portname
JOIN ports p_dest ON e.portname = p_dest.portname
WHERE f.fleet = 'containers' AND f.teus > 0
  AND e.prev_port IS NOT NULL
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, flow_type
```

---

## Bidirectional Traffic (for Heatmaps)

```sql
SELECT
  'INCOMING' as direction,
  e.portname as port,
  e.prev_port as connected_port,
  f.teus,
  e.prev_leg as distance,
  f.category,
  e.start
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.prev_port IS NOT NULL
  AND e.start >= '{start}' AND e.start < '{end}'

UNION ALL

SELECT
  'OUTGOING' as direction,
  e.prev_port as port,
  e.portname as connected_port,
  f.teus,
  e.prev_leg as distance,
  f.category,
  e.start
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.prev_port = '{port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.start >= '{start}' AND e.start < '{end}'
```

---

## Top Ports by KTM (for Port Matrix)

```sql
SELECT
  e.portname,
  p.zone,
  p.country,
  p.EEA,
  COUNT(*) as escalas,
  SUM(f.teus) as teus,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000, 1) as ktm_m
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE f.fleet = 'containers' AND f.teus > 0
  AND p.zone IN ('NORTH EUROPE', 'WEST MED', 'EAST MED', 'ATLANTIC', 'BLACK SEA')
  AND e.start >= '{start}' AND e.start < '{end}'
GROUP BY e.portname, p.zone, p.country, p.EEA
ORDER BY ktm_m DESC
LIMIT 30
```

---

## Checklist Diagnosis Patterns

| Pattern | Signature | Query to Check |
|---------|-----------|----------------|
| Route Alteration | Distance +25% WITHIN zone, TEU stable | Distance by Zone |
| Node Degradation | Mainline disappears, feeders compensate | By Category |
| Operational Consolidation | Fewer calls, same TEU, larger vessels | Macro + By Category |
| Client Abandonment | One operator drops, others stable | Mainline by Operator |
| Data Artifact | Missing >10%, days differ >15% | Data Quality (Paso 0) |

---

## Service Analysis Queries

### Service Metadata

```sql
SELECT sls.id as service_id, sls.name as servicio, sls.duration as rotacion_dias,
       sls.active_vessels, sls.teu_avg, sls.trade_lane, sls.is_feeder, sls.is_regular,
       sls.start_date, sls.end_date
FROM econdb_shared_line_service sls
WHERE sls.name LIKE '%{SERVICE}%'
```

### Operators and Alliances

```sql
SELECT sls.name as servicio, ls.operator, ls.alliance, ls.code as operator_code,
       ls.first_voyage, ls.last_voyage
FROM econdb_shared_line_service sls
JOIN econdb_line_service ls ON ls.shared_line_service_id = sls.id
WHERE sls.name LIKE '%{SERVICE}%'
```

### Ports with Frequency (Unified Table)

```sql
SELECT e.portname, p.country, p.zone, COUNT(*) as escalas,
       DATEDIFF(MAX(e.start), MIN(e.start)) as dias,
       ROUND(DATEDIFF(MAX(e.start), MIN(e.start)) / COUNT(*), 1) as dias_entre_escalas
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%' AND f.fleet = 'containers'
GROUP BY e.portname, p.country, p.zone
ORDER BY escalas DESC
```

### European Gateway (Entry from outside EUROMED)

```sql
SELECT e.portname, COUNT(*) as entries
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN ports p_prev ON e.prev_port = p_prev.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%'
  AND p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
GROUP BY e.portname
ORDER BY entries DESC LIMIT 1
```

### Reliability at Gateway Port

**CRITICAL:** Filter `prev_port.zone NOT IN EUROMED` to avoid counting return legs.

```sql
WITH intervalos AS (
  SELECT e.portname, e.start,
    LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start) as escala_anterior,
    TIMESTAMPDIFF(HOUR, LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start), e.start) as intervalo_horas
  FROM escalas e
  JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
  JOIN ports p_prev ON e.prev_port = p_prev.portname
  WHERE s.sls_name LIKE '%{SERVICE}%'
    AND e.portname = '{GATEWAY_PORT}'
    AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
),
stats AS (
  SELECT portname, COUNT(*) as total_intervalos,
    ROUND(AVG(intervalo_horas), 1) as intervalo_medio_h,
    ROUND(STDDEV(intervalo_horas), 1) as desviacion_std_h
  FROM intervalos WHERE intervalo_horas IS NOT NULL GROUP BY portname
)
SELECT i.portname, s.total_intervalos,
  ROUND(s.intervalo_medio_h / 24, 1) as frecuencia_dias,
  ROUND(s.desviacion_std_h / 24, 2) as std_dias,
  ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 6 THEN 1 ELSE 0 END) / s.total_intervalos, 1) as fiab_6h_pct,
  ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 12 THEN 1 ELSE 0 END) / s.total_intervalos, 1) as fiab_12h_pct,
  ROUND(100.0 * SUM(CASE WHEN ABS(i.intervalo_horas - s.intervalo_medio_h) <= 24 THEN 1 ELSE 0 END) / s.total_intervalos, 1) as fiab_24h_pct
FROM intervalos i JOIN stats s ON i.portname = s.portname
WHERE i.intervalo_horas IS NOT NULL
GROUP BY i.portname, s.total_intervalos, s.intervalo_medio_h, s.desviacion_std_h
```

### EUROMED Type Classification

```sql
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
WHERE s.sls_name LIKE '%{SERVICE}%'
```

### ETS Costs by Quarter

```sql
SELECT YEAR(e.start) as año, QUARTER(e.start) as trimestre, COUNT(*) as escalas,
  ROUND(SUM(em.incoming_ets_cost_eur), 0) as ets_entrada,
  ROUND(SUM(em.outgoing_ets_cost_eur), 0) as ets_salida,
  ROUND(SUM(em.total_ets_cost_eur), 0) as ets_total,
  ROUND(AVG(em.total_ets_cost_eur), 0) as ets_medio_escala
FROM escalas e
JOIN v_escalas_metrics em ON e.imo = em.imo AND e.start = em.start
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%' AND e.start >= '2024-01-01'
GROUP BY YEAR(e.start), QUARTER(e.start)
ORDER BY año, trimestre
```

### Deployed Fleet

```sql
SELECT f.name as buque, f.imo, ROUND(f.teus, 0) as teus, f.category,
  f.Year as año_construccion, f.fuel, f.group as operador, COUNT(*) as escalas
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{SERVICE}%' AND f.fleet = 'containers'
GROUP BY f.name, f.imo, f.teus, f.category, f.Year, f.fuel, f.group
ORDER BY escalas DESC
```

### Reliability Classification

| Fiabilidad ±12h | Clasificación | Descripción |
|----------------:|---------------|-------------|
| ≥ 90% | **Excelente** | Servicio muy predecible |
| 70-89% | **Buena** | Servicio fiable con variaciones ocasionales |
| 50-69% | **Moderada** | Variabilidad significativa |
| < 50% | **Baja** ⚠️ | Servicio irregular o datos inconsistentes |

### Service KTM Comparison (P1 vs P2)

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  COUNT(*) as escalas,
  ROUND(SUM(f.teus) / 1000, 0) as kteu,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000, 2) as ktm,
  ROUND(AVG(e.prev_leg)) as dist_media
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.service_id = {SERVICE_ID}
  AND f.fleet = 'containers'
  AND e.prev_leg IS NOT NULL
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period
```

### Service Ports Lost (P1 → P2)

```sql
SELECT e.portname, p.zone, COUNT(*) as escalas_p1,
  ROUND(SUM(f.teus * e.prev_leg) / 1000000, 2) as ktm_lost
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.service_id = {SERVICE_ID}
  AND f.fleet = 'containers'
  AND e.start >= '{p1_start}' AND e.start < '{p1_end}'
  AND e.portname NOT IN (
    SELECT DISTINCT e2.portname FROM escalas e2
    JOIN econdb_ship_service_ranges s2 ON e2.imo = s2.imo
      AND e2.start BETWEEN s2.start_range AND COALESCE(s2.end_range, '2099-12-31')
    WHERE s2.service_id = {SERVICE_ID}
      AND e2.start >= '{p2_start}' AND e2.start < '{p2_end}'
  )
GROUP BY e.portname, p.zone
ORDER BY ktm_lost DESC
```

### ETS Gateway Comparison

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  e.portname, p.EEA, COUNT(*) as entries,
  ROUND(SUM(em.total_ets_cost_eur), 0) as ets_total
FROM escalas e
JOIN v_escalas_metrics em ON e.imo = em.imo AND e.start = em.start
JOIN ports p ON e.portname = p.portname
JOIN ports p_prev ON e.prev_port = p_prev.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.service_id = {SERVICE_ID}
  AND e.portname IN ('{GATEWAY_1}', '{GATEWAY_2}')
  AND p.zone IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND p_prev.zone NOT IN ('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')
  AND e.start >= '2024-01-01'
GROUP BY period, e.portname, p.EEA
```

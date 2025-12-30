# Group Dive - Análisis Comparativo de Grupos de Puertos

> **Referencia metodológica**: Para análisis de EU ETS / carbon leakage, consultar el framework EUROMED en TRAFFIC.md — secciones "EUROMED Traffic Analysis Framework" y "escalas_metrics".

## Objetivo

Comparar un **grupo de puertos** entre dos períodos, identificando:

1. **Cambio agregado** del grupo (escalas, TEU, KTM)
2. **Top contributors** — qué puertos explican el cambio
3. **Flujos inter-grupo** — si hay dos grupos, cómo cambiaron los flujos entre ellos

El lector es un analista de mercado que necesita entender patrones regionales o regulatorios (ej: impacto EU ETS).

---

## Tipos de Grupo

| Método | Sintaxis | SQL |
|--------|----------|-----|
| Lista explícita | `"Vigo, Leixoes, Bilbao"` | `portname IN ('Vigo', 'Leixoes', 'Bilbao')` |
| Por zona | `zone:ATLANTIC` | `zone = 'ATLANTIC'` |
| Por país | `country:ES` | `country = 'Spain'` |
| Por EEA | `EEA=0` o `EEA=0.5` | `EEA = 0.00` o `EEA = 0.50` |
| Combinación | `zone:WEST MED AND EEA=0` | `zone = 'WEST MED' AND EEA = 0.00` |

### Comparación de DOS grupos

Sintaxis: `<grupo_A> vs <grupo_B> <P1> <P2>`

Ejemplo: `EEA=0 vs EEA=0.5 2023-Q3 2025-Q3`

Esto activa el análisis de **flujos inter-grupo**.

---

## Secuencia de Análisis

### Paso 1: Métricas Agregadas por Grupo

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

### Paso 2: Desglose por Puerto (Top Contributors)

```sql
SELECT
    e.portname,
    p.zone,
    CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
    COUNT(*) as escalas,
    SUM(f.teus) as teus
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE {group_filter}
  AND f.fleet = 'containers' AND f.teus > 0
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
       OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY e.portname, p.zone, period
```

**Cálculo de contribución:**
```
contrib_i = |Δ_i| / Σ|Δ_i|
```

Ordenar por `|Δ|` descendente, reportar Top-5.

### Paso 3: Flujos Inter-Grupo (solo si hay dos grupos)

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

## Estructura del Informe

### Para UN solo grupo:

```markdown
# {Grupo} — Análisis de Tráfico ({P1} vs {P2})

## 1. Resumen Ejecutivo
[3-5 bullets: qué pasó, quién lo causó, magnitud]

## 2. Métricas Agregadas

| Métrica | {P1} | {P2} | Δ |
|---------|-----:|-----:|--:|
| Escalas | X | Y | Z% |
| TEU (K) | X | Y | Z% |
| KTM (M) | X | Y | Z% |
| Puertos activos | N | M | — |

## 3. Top Contributors

| Puerto | Zona | Δ Escalas | Contrib % |
|--------|------|----------:|----------:|
| ... | ... | ... | ... |

[Interpretación: qué puertos explican el cambio y por qué]

## 4. Hipótesis

**H1: [Título]**
- Evidence: [datos]
- Hypothesis: [mecanismo]
- Implicación comercial: [qué significa para el mercado]
- Confidence: Alto/Medio/Bajo
```

### Para DOS grupos (comparación):

```markdown
# {Grupo A} vs {Grupo B} — Análisis Comparativo ({P1} vs {P2})

## 1. Resumen Ejecutivo
[Qué grupo ganó, cuál perdió, magnitud del shift]

## 2. Métricas por Grupo

| Grupo | Métrica | {P1} | {P2} | Δ |
|-------|---------|-----:|-----:|--:|
| {A} | Escalas | ... | ... | ... |
| {A} | TEU (K) | ... | ... | ... |
| {B} | Escalas | ... | ... | ... |
| {B} | TEU (K) | ... | ... | ... |

## 3. Top Contributors por Grupo

### {Grupo A}
| Puerto | Δ | Contrib % |

### {Grupo B}
| Puerto | Δ | Contrib % |

## 4. Flujos Inter-Grupo

| Flujo | {P1} | {P2} | Δ |
|-------|-----:|-----:|--:|
| {A} → {A} (internal) | ... | ... | ... |
| {A} → {B} | ... | ... | ... |
| {B} → {A} | ... | ... | ... |
| {B} → {B} (internal) | ... | ... | ... |

[Interpretación: ¿hay evidencia de redistribución entre grupos?]

## 5. Hipótesis

**H1: [Título]**
...
```

---

## Caso de Uso: Impacto EU ETS

**Pregunta clave:** ¿Hay evidencia de carbon leakage? ¿El tráfico se está moviendo de puertos EEA a puertos non-EEA?

**Señales a buscar:**
1. Grupo EEA=0 crece más que EEA=0.5
2. Flujo `EEA → non-EEA` crece más que `non-EEA → EEA`
3. Top contributors en EEA=0 son hubs cercanos (Tangier Med, Port Said)

**Precauciones:**
- Correlación ≠ causación: el shift puede deberse a otros factores (capacidad, congestión)
- Verificar si los mismos operadores están moviendo tráfico (seguir IMOs)
- EU ETS solo entró en vigor 2024 → baseline 2023-Q3 es pre-ETS

---

## Tabla EEA

La columna `EEA` en la tabla `ports` indica:
- `0.00` = Puerto fuera del Espacio Económico Europeo (Tangier Med, Port Said, etc.)
- `0.50` = Puerto dentro del EEA (Rotterdam, Algeciras, Piraeus, etc.)

El valor 0.50 corresponde a la fracción de emisiones que se aplica EU ETS en viajes intra-EEA.

---

## Limitaciones

1. **No hay datos de servicio a nivel de grupo** — solo escalas agregadas
2. **No distingue transbordo vs import/export** — todos los movimientos cuentan igual
3. **EEA es binario** — no captura gradaciones (puertos parcialmente cubiertos)
4. **Baseline temporal** — EU ETS entró en vigor enero 2024; comparaciones pre/post requieren cuidado

---

## Regla de Oro

```sql
⚠️ SIEMPRE incluir en WHERE:
   f.fleet = 'containers' AND f.teus > 0

Sin este filtro, incluirás graneleros, tankers, etc.
```

# Deep Dive - Análisis de Transformación Portuaria

> **Referencia metodológica**: Para análisis de EU ETS / carbon leakage, consultar el framework EUROMED en TRAFFIC.md — secciones "EUROMED Traffic Analysis Framework" y "escalas_metrics".

## Objetivo

Escribir un informe para un **planificador estratégico** (autoridad portuaria, regulador, competidor) que necesita entender:

1. **Qué transformación** ha ocurrido en el puerto
2. **Quién la ha provocado** (operadores, alianzas, servicios)
3. **Qué implica** para el mercado (concentración, conectividad, competencia)

El lector tolera complejidad técnica pero no especulación. Cada afirmación debe estar soportada por datos.

Los ejes de transformación (mix, conectividad, hub/spoke, concentración) emergen del análisis — no están prefijados.

---

## Perfil del Analista

Eres un analista de tráfico marítimo. Tu trabajo no es describir datos, es **explicar decisiones comerciales**.

> La mecánica (queries, métricas) es el instrumento; la narrativa comercial es el producto.

### Mandatos verificables

| Principio | Mandato |
|-----------|---------|
| "Piensa en lógica comercial" | Cada hipótesis incluye `Implicación comercial: ...` |
| "Traza la causa" | Cada Δ significativo tiene Top-3 drivers con % contribución |
| "Pregunta a dónde fue" | Pérdida >20% activa módulo redistribución |

### Umbrales de activación

- Pérdida >20% escalas en segmento → investigar redistribución
- Operador pierde >50% share → rastrear destino
- Zona pierde conectividad mainline completa → obligatorio rastrear
- Nueva ruta aporta >20% del KTM total → investigar origen del servicio
- Nuevo operador aparece con >10% share → rastrear procedencia de buques

### Constraints (NO puede hacer)

- ❌ No atribuir a crisis externas sin evidencia en datos
- ❌ No nombrar servicios sin consultar `v_econdb_service_escala`
- ❌ No afirmar redistribución sin verificar salida Y llegada
- ❌ No mezclar conclusiones mainline/feeder
- ❌ No concluir sobre mainline usando solo Cat 5 (ULCV) — usar siempre Cat 4+5 combinada
- ❌ No afirmar "operador se fue" sin verificar TODAS las categorías del operador
- ❌ No concluir "downgrade" si KTM o escalas totales crecen

---

## Disciplina Analítica

### Hard stops (abortar o advertencia grave)

- Períodos difieren >20% en días → **normalizar antes de continuar**
- Missing prev_port >15% → **advertencia grave en informe**
- Categoría sin datos → **no concluir sobre ese segmento**

### Soft (reducir confianza)

- Missing 5-15% → confianza "Medio"
- Muestra <20 escalas en segmento → advertir "low sample size"

### Qué cuenta como evidencia

- Un **corte** = desglose por una dimensión (zona, operador, categoría, servicio)
- **Mínimo para afirmar**: 2 cortes independientes que cuenten la misma historia
- Ejemplo: "TEU cayó 30%" requiere desglose por operador Y por zona

### Convención de contribución

- Usar **share del delta**: `contrib_i = ΔX_i / Σ|ΔX_i|` (suma 100%)
- Siempre reportar Top-5 drivers con % contribución

### Interpretación TEU vs KTM

| Δ TEU | Δ KTM | Patrón |
|-------|-------|--------|
| ≈ 0 | < 0 | Downgrade (rutas largas→cortas) |
| ≈ 0 | > 0 | Upgrade (nuevas rutas largas) |
| < 0 | < 0 | Contracción regional |
| ≈ 0 | ≈ 0 | Redistribución pura |

### Regla de verificación de redistribución

**NUNCA afirmes que "el tráfico se movió de A a B" sin verificar ambos lados:**

1. ✅ Verificar SALIDA: Operador X perdió N escalas en Puerto A
2. ✅ Verificar LLEGADA: Operador X ganó M escalas en Puerto B

| Nivel de evidencia | Frase requerida |
|--------------------|-----------------|
| Mismo operador + mismos puertos | "evidencia fuerte de redistribución" |
| Mismo operador, puertos diferentes | "consistente con redistribución" |
| Diferente operador | "correlación geográfica, no se confirma redistribución" |

### Checklist de Coherencia (OBLIGATORIO antes de concluir)

**Antes de afirmar "downgrade", "declive" o "pérdida":**

| Verificar | Condición para "downgrade" válido |
|-----------|-----------------------------------|
| KTM | Decrece |
| Escalas totales | Decrecen |
| TEU total | Decrece |
| Mainline (Cat 4+5) | Decrece combinada |

⚠️ **Si algún indicador CRECE → la conclusión "downgrade" es INVÁLIDA**

**Ejemplo de error (Felixstowe Q3'23→Q3'25):**
- ULCV: -40% → Conclusión errónea: "downgrade"
- Pero: KTM +157%, escalas +7%, New Panamax +158%
- Conclusión correcta: "cambio de mix operativo, no declive"

### Regla de Operador Completo

**NUNCA afirmes "operador X abandonó el puerto" sin verificar:**

1. ✅ Verificar Cat 5 (ULCV)
2. ✅ Verificar Cat 4 (New Panamax)
3. ✅ Verificar Cat 3 (Post Panamax)

| Situación | Frase correcta |
|-----------|----------------|
| Sale de Cat 5, crece en Cat 4 | "cambió estrategia de flota" |
| Sale de todas las categorías | "abandonó el puerto" |
| Sale de Cat 5, estable en otras | "redujo presencia mainline" |

---

## Investigación de Servicios por IMO

### Principio: Trazar el Cambio Completo

Cuando identificamos un cambio significativo, debemos investigar **ambas direcciones**:

1. **Para PÉRDIDAS**: ¿A dónde fue el tráfico/servicio?
2. **Para GANANCIAS**: ¿De dónde viene el nuevo tráfico/servicio?

### Procedimiento: Trazar Pérdidas

**Activar cuando:** Ruta pierde >50% KTM, operador pierde >50% share.

**Paso 1:** Identificar buques que operaban la ruta en P1

```sql
SELECT DISTINCT f.imo, f.shipname, f.teus, f.group, COUNT(*) as escalas_p1
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE e.portname = '{port}' AND e.prev_port = '{connected_port}'
  AND f.fleet = 'containers' AND f.teus > 0
  AND e.start >= '{p1_start}' AND e.start < '{p1_end}'
GROUP BY f.imo, f.shipname, f.teus, f.group
ORDER BY escalas_p1 DESC
```

**Paso 2:** Verificar dónde operan esos IMOs en P2

```sql
SELECT e.prev_port, e.portname as next_port, p.zone, COUNT(*) as escalas
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE f.imo IN ({imos_from_step1})
  AND f.fleet = 'containers'
  AND e.start >= '{p2_start}' AND e.start < '{p2_end}'
GROUP BY e.prev_port, e.portname, p.zone
ORDER BY escalas DESC
LIMIT 20
```

**Paso 3:** Clasificar resultado

| Situación | Conclusión |
|-----------|------------|
| Mismo operador, mismos puertos conectados | "Reducción de frecuencia" |
| Mismo operador, puertos diferentes | "Reestructuración de red" |
| Buque redeployado a otra zona | "Redeployment estratégico" |
| Buque ya no opera | "Retirada de servicio" |

### Procedimiento: Trazar Ganancias

**Activar cuando:** Nueva ruta aporta >20% del KTM total, nuevo operador >10% share.

**Paso 1:** Identificar buques nuevos en P2 (no operaban el puerto en P1)

```sql
SELECT f.imo, f.shipname, f.teus, f.group, COUNT(*) as escalas_p2
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
ORDER BY escalas_p2 DESC
```

**Paso 2:** Verificar dónde operaban esos IMOs en P1

```sql
SELECT e.portname, p.zone, p.country, COUNT(*) as escalas
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE f.imo IN ({new_imos})
  AND f.fleet = 'containers'
  AND e.start >= '{p1_start}' AND e.start < '{p1_end}'
GROUP BY e.portname, p.zone, p.country
ORDER BY escalas DESC
LIMIT 20
```

**Paso 3:** Clasificar origen

| Situación | Conclusión |
|-----------|------------|
| Servicio existía sin el puerto | "Puerto AÑADIDO a servicio existente" |
| Buques operaban otra zona | "Redeployment hacia el puerto" |
| Buques no operaban (nuevos) | "Nuevo servicio/flota" |

### Análisis de Sustitución vs Complemento

Cuando un puerto A pierde tráfico y un puerto B aparece en el mismo servicio:

```sql
SELECT
  CASE WHEN e.start >= '{p1_start}' AND e.start < '{p1_end}' THEN 'P1' ELSE 'P2' END as period,
  e.portname, COUNT(*) as escalas
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
WHERE f.imo IN ({service_imos})
  AND e.portname IN ('{port_A}', '{port_B}')
  AND f.fleet = 'containers'
  AND ((e.start >= '{p1_start}' AND e.start < '{p1_end}')
    OR (e.start >= '{p2_start}' AND e.start < '{p2_end}'))
GROUP BY period, e.portname
ORDER BY e.portname, period
```

| Resultado | Conclusión |
|-----------|------------|
| A desaparece, B aparece | "B **sustituye** a A" |
| A se mantiene, B aparece | "B **complementa** a A" |
| A reduce, B aparece | "Redistribución parcial" |

---

## Checklist de Diagnóstico

**Paso 1 (después de validación):** Identifica el patrón dominante.

**Output requerido:** `Patrón dominante: [X]. Confianza: [Alta/Media/Baja]`

| Patrón | Firma de datos | Query de verificación |
|--------|----------------|----------------------|
| **Route Alteration** | Distancia +25% DENTRO zona, TEU estable | `SELECT period, zone, AVG(prev_leg) ... GROUP BY period, zone` |
| **Node Degradation** | Mainline desaparece, feeders compensan | `SELECT period, category, COUNT(*) ... GROUP BY period, category` |
| **Operational Consolidation** | Menos escalas, mismo TEU, buques mayores | `SELECT period, COUNT(*), SUM(teus), AVG(teus) ... GROUP BY period` |
| **Client Abandonment** | Un operador cae >50%, otros estables | `SELECT period, operator, COUNT(*) ... GROUP BY period, operator` |
| **Data Artifact** | Missing >10% en un período, días difieren >15% | Usar resultados de Paso 0 |

---

## Paso 0: Validación de Calidad

**ANTES de analizar**, ejecuta query de validación:

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

### Output de Paso 0

```
Validación: [OK / WARNING / FAIL]
- Días: P1=XX, P2=XX [OK/NORMALIZAR]
- Missing prev_port: P1=X%, P2=X% [OK/WARNING]
- Confianza inicial: [Alta/Media/Baja]
```

---

## Contrato de Salida

El informe sigue esta estructura fija.

### 1. Resumen Ejecutivo (max 5 bullets)

- Las conclusiones principales en lenguaje de negocio
- Cada bullet responde: qué pasó, quién lo causó, magnitud del cambio

### 2. Tabla de Métricas Macro

| Dimensión | Métrica | P1 | P2 | Δ |
|-----------|---------|---:|---:|--:|
| Actividad | Escalas | | | |
| | TEU (K) | | | |
| Alcance | Distancia media (nm) | | | |
| | KTM (M) | | | |
| Ocupación | Horas medias | | | |

### 3. Mainline (cat 4-5: New Panamax + ULCV)

- **Qué cambió**: variación de escalas, TEU, distancia
- **Quién lo explica**: Top-5 operadores/servicios con % contribución
- **Conectividad por ruta**: qué rutas ganaron/perdieron
- **Investigación de servicios** (si aplica threshold):
  - Pérdidas: trazado por IMO → destino del tráfico
  - Ganancias: origen del servicio → sustitución vs complemento
- **Hipótesis**: Evidence/Hypothesis/Alternative/Confidence + Implicación comercial

### 4. Feeder/Regional (cat 1-3)

- **Qué cambió**: variación de escalas, TEU, distancia
- **Quién lo explica**: Top-5 operadores con % contribución
- **Hipótesis**: 1-2 hipótesis data-driven

### 5. Limitaciones y Calidad de Datos

### 6. Redistribución Regional (CONDICIONAL)

**Activar solo si:** Pérdida mainline >20%, Pérdida >30% con zona/ruta, Operador pierde >50% share

---

## Regla de Oro

**Incluir SIEMPRE en WHERE:**
```sql
f.fleet = 'containers' AND f.teus > 0
```
Sin este filtro, incluirás graneleros, tankers, etc.

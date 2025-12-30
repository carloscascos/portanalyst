---
name: portanalyst
description: Port traffic transformation analysis toolkit for maritime intelligence. Use when analyzing port traffic changes between periods, detecting carbon leakage (EU ETS), visualizing connectivity shifts, or investigating competitive dynamics. Provides Deep Dive (single port transformation), Group Dive (port group comparison), Heatmap (TEU x Distance visualization), Port Matrix (ULCV connectivity), and Connectivity Cube (323 ports x 15 quarters precalculated). Requires traffic-db MCP server.
---

# Port Analyst - Traffic Transformation Toolkit

**Version:** 1.0.0
**Created:** 2025-12-29
**Type:** Maritime Analytics - Comparative Analysis

## Overview

The **portanalyst** skill analyzes **how port traffic has CHANGED between two periods**. While `traffic-profile` answers "What's the port like NOW?", portanalyst answers "What CHANGED between P1 and P2?".

This skill is essential for:
- **Strategic planning**: Detect traffic shifts before competitors
- **Regulatory analysis**: Assess EU ETS carbon leakage
- **Competitive intelligence**: Track market share changes
- **Infrastructure planning**: Identify emerging capacity needs

## Quick Reference

| Herramienta | Pregunta que responde | Output |
|-------------|----------------------|--------|
| **Deep Dive** | ¿Qué cambió en [Puerto] entre P1 y P2? | Informe MD con hipótesis |
| **Group Dive** | ¿Cómo evolucionó [grupo A] vs [grupo B]? | Análisis comparativo |
| **Heatmap** | ¿Cómo cambió el perfil TEU × Distancia? | PNG 3 paneles |
| **Port Matrix** | ¿Cómo cambió la conectividad mainline? | PNG matriz 39×39 |
| **Cube Query** | Datos históricos rápidos (sin DB) | Numpy arrays |
| **Service Analysis** | ¿Cómo opera el servicio [X]? | Ficha + puertos + fiabilidad |

### Comandos Típicos

- `"Deep dive Valencia Q3 2023 vs Q3 2025"` → Análisis completo
- `"Heatmap Piraeus"` → Visualización de cambio de perfil
- `"Port matrix EUROMED"` → Conectividad ULCV regional
- `"Group dive EEA=0 vs EEA=0.5"` → Carbon leakage analysis
- `"Analiza el servicio AE3"` → Ficha del servicio liner
- `"Service analysis Guangdong Express"` → Ficha operacional completa

## When to Use This Skill

**Activate portanalyst when user asks:**
- "Compare [port] Q3 2023 vs Q3 2025"
- "Analyze [port] traffic transformation"
- "What changed at [port] since [period]?"
- "Carbon leakage analysis for [region]"
- "EEA vs non-EEA traffic comparison"
- "Port connectivity evolution"
- "Heatmap comparison for [port]"
- "ULCV connectivity matrix"
- "Analyze service [AE3, TA4, etc.]"
- "Service reliability for [service name]"
- "Which operators run [service]?"
- "Service frequency and punctuality"

**Do NOT use for:**
- Current port profile/snapshot (use `traffic-profile`)
- Single-period fleet composition (use `traffic-profile`)
- Real-time vessel tracking (different system)

## Architecture

### Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| **traffic-db MCP** | Primary database (escalas, v_fleet, ports) | `db_query`, `db_export_query_to_file` |
| **Connectivity Cube** | Precalculated 4D matrix (323 ports x 15 quarters) | `scripts/cube.py` |

### Visual Tools

| Tool | Output | Purpose |
|------|--------|---------|
| **Heatmap** | PNG (3 panels: T1, T2, Delta) | TEU x Distance distribution comparison |
| **Port Matrix** | PNG (3 panels: T1, T2, Delta) | 39x39 ULCV connectivity matrix |

### Methodologies

| Method | Focus | Output |
|--------|-------|--------|
| **Deep Dive** | Single port transformation | Structured MD report with hypotheses |
| **Group Dive** | Port group comparison | Inter-group flow analysis |

## Technique Details

### 1. Deep Dive (Single Port Transformation)

**Question:** "What happened to [Port] between P1 and P2?"

**Process:**
1. Load metrics for both periods
2. Calculate deltas (escalas, TEU, KTM)
3. Identify winners/losers (routes, operators, categories)
4. Generate hypotheses with evidence
5. Output structured report

**Methodology:** See `references/deepdive-methodology.md`

**Example:**
```
User: "Deep dive analysis of Piraeus Q3 2023 vs Q3 2025"

Output: ./analysis/Piraeus_Q3_2023_vs_Q3_2025.md
```

### 2. Group Dive (Port Group Comparison)

**Question:** "How did [group A] vs [group B] change between P1 and P2?"

**Groups supported:**
- Explicit list: `"Vigo, Leixoes, Bilbao"`
- By zone: `zone:ATLANTIC`
- By country: `country:ES`
- By EEA status: `EEA=0` (non-EEA) or `EEA=0.5` (EEA)

**Process:**
1. Aggregate metrics per group
2. Identify top contributors (which ports explain the change)
3. Analyze inter-group flows (for 2-group comparison)
4. Output comparative report

**Methodology:** See `references/groupdive-methodology.md`

**Example:**
```
User: "Group dive EEA=0 vs EEA=0.5 Q3 2023 vs Q3 2025"

Output: Analysis showing carbon leakage evidence
```

### 3. Heatmap (TEU x Distance Visualization)

**Question:** "How did the traffic profile (vessel size x route distance) change?"

**Process:**
1. Export bidirectional traffic data for both periods
2. Build 2D histograms (TEU bins x Distance bins)
3. Calculate statistical divergence (JSD, Hellinger, EMD)
4. Generate 3-panel PNG (T1, T2, Delta)
5. Identify hotspots (biggest changes)

**Script:** `scripts/heatmap.py`

**Example:**
```python
from scripts.heatmap import build_query, compare_periods

# Generate query
query = build_query("Valencia", 2023, "Q3", 2025, "Q3")

# Export and analyze
result = compare_periods(
    csv_path="data.csv",
    port="Valencia",
    year1=2023, period1="Q3",
    year2=2025, period2="Q3"
)

print(f"JSD: {result['jsd']:.3f} - {result['status']}")
# Output: ./analysis/default/Valencia_2023_Q3_vs_2025_Q3_bidirectional.png
```

### 4. Port Matrix (ULCV Connectivity)

**Question:** "How did mainline vessel (ULCV) connectivity between major ports change?"

**Process:**
1. Query top 30 EUROMED ports by KTM
2. Build origin-destination matrices for Cat 4-5 vessels
3. Compare T1 vs T2 matrices
4. Generate 3-panel PNG with hotspots
5. Identify top gains/losses

**Script:** `scripts/portmatrix.py`

**Scope:**
- 39 axis labels (30 top ports + "Otros EUROMED" + 8 external zones)
- Cat 4-5 vessels only (New Panamax, ULCV)
- External zones: ASIA, INDIA, MIDDLE EAST, SOUTH AFRICA, WEST AFRICA, NA ATLANTIC, CARIBE, SA ATLANTIC

### 5. Connectivity Cube (Precalculated Data)

**Purpose:** Fast access to historical connectivity data without live queries

**Structure:**
- Shape: `(323, 323, 15, 5)` = Origins x Destinations x Quarters x Categories
- Periods: Q1 2022 → Q3 2025 (15 quarters)
- Categories: 1=Feeder, 2=Panamax, 3=Post-Panamax, 4=New Panamax, 5=ULCV
- Metrics: calls, teus, nm, ktm

**Data location:** `~/proyectos/data/traffic/cube/cube_4d.npz`

**Schema:** See `references/cube-schema.md`

**Example:**
```python
from scripts.cube import load_default_cube

cube = load_default_cube()

# Get Piraeus → Asia mainline traffic in Q3 2025
pir_idx = cube.axis.index('Piraeus')
asia_idx = cube.axis.index('ASIA')
q3_2025_idx = cube.periods.index('2025 Q3')

mainline_ktm = cube.ktm[pir_idx, asia_idx, q3_2025_idx, 3:5].sum()
```

### 6. Service Analysis (Análisis de Servicio)

**Question:** "How does liner service [X] operate? What's its reliability?"

**Process:**
1. Identify service in econdb tables (shared_line_service, line_service)
2. Extract operators, alliance, rotation, trade lane
3. Analyze port call frequency and port classification
4. Calculate reliability at European gateway port
5. Compute ETS costs (since 2024)
6. List deployed fleet with vessel details
7. Generate structured ficha

**Data sources:**
- `econdb_shared_line_service` - Service metadata
- `econdb_line_service` - Operators and alliances
- `econdb_ship_service_ranges` - Vessel ↔ service assignment
- `escalas` - Actual port calls
- `v_escalas_metrics` - ETS costs

**Output:**
- **Ficha del Servicio**: Name, operators, alliance, rotation, frequency, reliability
- **Tabla de Puertos**: Calls, frequency, classification (Regular/Ocasional/Esporádico)
- **Flota Desplegada**: TEU, category, year, fuel, operator
- **ETS Analysis**: Quarterly costs, entry vs exit breakdown

**Methodology:** See `references/service-methodology.md`

**Example:**
```
User: "Analiza el servicio AE3"

Output:
## AE3 / Asia North Europe 3

**Operadores:** Maersk, Hapag-Lloyd (Gemini)
**Alianza:** Gemini
**Rotación:** 45 días
**Frecuencia:** 1 buque cada 7.2 días (semanal) en Algeciras
**Fiabilidad:** 54% ±12h en Algeciras (Moderada)
**Tipo EUROMED:** EXTRA
**Gateway entrada:** Algeciras (37 cruces)
**ETS 2024:** €8.5M

[Tabla de puertos...]
[Flota desplegada...]
```

## Output Paths

All outputs go to `./analysis/` relative to current working directory:

| Mode | Path | Naming |
|------|------|--------|
| Default (testing) | `./analysis/default/` | `default.{png,json,md}` |
| Production | `./analysis/{Port}/` | `{Port}_{P1}_vs_{P2}.{png,json,md}` |

## SQL Query Templates

Common query patterns are documented in `references/query-templates.md`.

**CRITICAL filters for container analysis:**
```sql
WHERE f.fleet = 'containers'
  AND f.teus > 0
```

## Workflow Examples

### Example 1: Complete Port Analysis

```
User: "Full analysis of Valencia Q3 2023 vs Q3 2025"

Steps:
1. Deep Dive → ./analysis/Valencia/Valencia_Q3_2023_vs_Q3_2025.md
2. Heatmap → ./analysis/Valencia/Valencia_Q3_2023_vs_Q3_2025.png
3. [Optional] Port Matrix for context
```

### Example 2: EU ETS Carbon Leakage

```
User: "Detect carbon leakage in Mediterranean Q3 2023 vs Q3 2025"

Steps:
1. Group Dive: EEA=0 vs EEA=0.5 → Identify if non-EEA growing faster
2. Top contributors → Which ports explain the shift
3. Inter-group flows → EEA→non-EEA vs non-EEA→EEA changes
4. Port Matrix → ULCV connectivity changes (Tangier Med, Port Said, Piraeus)
```

### Example 3: Quick Heatmap Check

```
User: "Heatmap for Piraeus traffic profile change"

Steps:
1. Export bidirectional data for Q3 2023 and Q3 2025
2. Run heatmap.py
3. Review PNG and JSON hotspots
```

## Cross-Reference: traffic-profile

| Need | Use |
|------|-----|
| Current fleet composition | `traffic-profile` |
| Which operators serve the port NOW | `traffic-profile` |
| What CHANGED between P1 and P2 | `portanalyst` |
| Carbon leakage analysis | `portanalyst` |
| Traffic transformation hypotheses | `portanalyst` |

**Complementary workflow:**
1. `traffic-profile` → Snapshot P1 (baseline)
2. `traffic-profile` → Snapshot P2 (current)
3. `portanalyst` → Comparative analysis (what changed, why, implications)

## File Structure

```
~/.claude/skills/portanalyst/
├── SKILL.md                    # This file
├── scripts/
│   ├── cube.py                 # Connectivity cube access
│   ├── heatmap.py              # TEU x Distance heatmaps
│   ├── portmatrix.py           # ULCV connectivity matrices
│   └── service.py              # Service analysis (liner services)
└── references/
    ├── deepdive-methodology.md # Deep dive workflow
    ├── groupdive-methodology.md# Group comparison workflow
    ├── cube-schema.md          # Cube structure documentation
    ├── query-templates.md      # Reusable SQL patterns
    └── service-methodology.md  # Service analysis workflow

~/proyectos/data/traffic/cube/  # EXTERNAL DATA
├── cube_4d.npz                 # 4D connectivity cube (~250 MB)
└── euromed_ports_active.csv    # Port index (315 ports)
```

## Dependencies

- **traffic-db MCP server** (database access)
- Python: numpy, matplotlib, scipy, pandas
- External data: `~/proyectos/data/traffic/cube/`

## Skill Metadata

- **Category:** Maritime Analytics - Comparative Analysis
- **Database:** traffic-db MCP (read-only)
- **Output:** PNG visualizations, JSON metrics, MD reports
- **Region:** EUROMED focus (315 ports + 8 external zones)
- **Architecture:** Database-first with precalculated cube
- **Version:** 1.0.0

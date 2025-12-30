# Connectivity Cube Schema

4D connectivity matrix for EUROMED container traffic analysis.

## Data Location

```
~/proyectos/data/traffic/cube/
├── cube_4d.npz                 # 4D connectivity cube (~250 MB)
├── cube_flat.csv               # Tabular version (3.6 MB, 81K rows)
└── euromed_ports_active.csv    # Port index (315 ports)
```

## cube_4d.npz

| Property   | Value                             |
|------------|-----------------------------------|
| Shape      | (323, 323, 15, 5)                 |
| Ports      | 315 EUROMED + 8 external zones    |
| Periods    | Q1 2022 → Q3 2025 (15 quarters)   |
| Categories | 1-5 (Feeder → ULCV)               |
| Metrics    | calls, teus, nm, ktm              |

### Structure

```python
import numpy as np

data = np.load('cube_4d.npz', allow_pickle=True)

# 4D arrays (origin, destination, period, category)
calls = data['calls']  # Number of calls
teus = data['teus']    # TEUs transported
nm = data['nm']        # Nautical miles
ktm = data['ktm']      # kTEU·miles (transport work)

# Axes
axis = data['axis']              # 323 port/zone names
periods = data['periods']        # 15 quarters
categories = data['categories']  # [1, 2, 3, 4, 5]
port_meta = data['port_meta'].item()  # {port: {country, zone, eea}}
```

### Vessel Categories

| Cat | Name          | TEU          |
|-----|---------------|--------------|
| 1   | Feeder        | < 3,000      |
| 2   | Panamax       | 3,000-5,000  |
| 3   | Post Panamax  | 5,000-10,000 |
| 4   | New Panamax   | 10,000-15,000|
| 5   | ULCV          | > 15,000     |

### External Zones

ASIA, INDIA, MIDDLE EAST, SOUTH AFRICA, WEST AFRICA, NA ATLANTIC, CARIBE, SA ATLANTIC

### Usage Example

```python
import numpy as np

data = np.load('cube_4d.npz', allow_pickle=True)
calls = data['calls']
axis = list(data['axis'])
periods = list(data['periods'])

# Get Rotterdam → Piraeus traffic in Q3 2025, categories 4-5
rot_idx = axis.index('Rotterdam')
pir_idx = axis.index('Piraeus')
q3_25_idx = periods.index('2025 Q3')

mainline_calls = calls[rot_idx, pir_idx, q3_25_idx, 3:5].sum()
```

---

## euromed_ports_active.csv

Port index defining the cube axis, sorted by activity (ktm).

| Column | Description |
|--------|-------------|
| `portname` | Port name |
| `zone` | Geographic zone (NORTH EUROPE, WEST MED, EAST MED, ATLANTIC) |
| `country` | Country |
| `calls` | Total accumulated calls (ranking) |
| `teus` | Total accumulated TEUs (ranking) |
| `ktm` | Total kTEU·miles (sorting metric) |
| `eea` | EEA flag: 0.5 = EEA port, 0.0 = non-EEA |

**Note on EEA**: UK ports have `eea=0.0` post-Brexit.

---

## cube_flat.csv

Tabular version of the cube for SQL queries (3.6 MB, 81K rows).

| Column | Type | Description |
|--------|------|-------------|
| origin | string | Origin port/zone |
| destination | string | Destination port/zone |
| period | string | Quarter (e.g., "2025 Q3") |
| category | int | Vessel category (1-5) |
| calls | int | Number of calls |
| teus | int | TEUs transported |
| nm | float | Nautical miles |
| ktm | float | kTEU·miles |

---

## Access Methods

### Python (numpy)

```python
from scripts.cube import load_default_cube

cube = load_default_cube()

# Outgoing traffic from Piraeus in Q3 2025
pir_idx = cube.axis.index('Piraeus')
q3_idx = cube.periods.index('2025 Q3')

outgoing = cube.calls[pir_idx, :, q3_idx, :].sum(axis=1)
```

### DuckDB (SQL)

```bash
# Outgoing traffic from Piraeus Q3 2025
duckdb -c "
SELECT destination, SUM(calls) as calls, SUM(teus) as teus
FROM 'cube_flat.csv'
WHERE origin = 'Piraeus' AND period = '2025 Q3'
GROUP BY destination
ORDER BY calls DESC
LIMIT 10
"

# Incoming traffic to Valencia (cat 4-5)
duckdb -c "
SELECT origin, SUM(calls) as calls, SUM(ktm) as ktm
FROM 'cube_flat.csv'
WHERE destination = 'Valencia'
  AND period = '2025 Q3'
  AND category IN (4, 5)
GROUP BY origin
ORDER BY calls DESC
"

# Quarterly evolution Piraeus - Asia
duckdb -c "
SELECT period, SUM(calls) as calls
FROM 'cube_flat.csv'
WHERE (origin = 'Piraeus' AND destination = 'ASIA')
   OR (origin = 'ASIA' AND destination = 'Piraeus')
GROUP BY period
ORDER BY period
"
```

---

## Period Index

| Index | Period  |
|-------|---------|
| 0     | 2022 Q1 |
| 1     | 2022 Q2 |
| 2     | 2022 Q3 |
| 3     | 2022 Q4 |
| 4     | 2023 Q1 |
| 5     | 2023 Q2 |
| 6     | 2023 Q3 |
| 7     | 2023 Q4 |
| 8     | 2024 Q1 |
| 9     | 2024 Q2 |
| 10    | 2024 Q3 |
| 11    | 2024 Q4 |
| 12    | 2025 Q1 |
| 13    | 2025 Q2 |
| 14    | 2025 Q3 |

---

## Notes

- **Cube is read-only**: Pre-calculated for fast access, not for real-time updates
- **External zones aggregate**: ASIA includes all Asian ports outside EUROMED
- **Direction matters**: `calls[A, B]` ≠ `calls[B, A]` (origin → destination)
- **Zero values**: Many cells are zero (sparse matrix) - use masked arrays when visualizing

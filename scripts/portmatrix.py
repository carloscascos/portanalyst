#!/usr/bin/env python3
"""
portmatrix.py - Matriz de conectividad puerto-a-puerto (SIMÉTRICA) - STANDALONE

Compara matrices de conectividad entre dos períodos para buques ULCV (cat 4-5).

3 primitivas:
  - snapshot(rows): construye matriz 39x39 desde resultados de query
  - compare(s1, s2): compara 2 matrices, retorna métricas + diferencia
  - render_symmetric(s1, s2, metrics): genera PNG con 3 paneles (T1, T2, Delta)

Ejes (39 entradas):
  - Top 30 puertos EUROMED por KTM (período más reciente)
  - "Otros EUROMED" (agregado de puertos menores)
  - 8 zonas externas: ASIA, INDIA, MIDDLE EAST, SOUTH AFRICA, WEST AFRICA,
    NA ATLANTIC, CARIBE, SA ATLANTIC

Matriz simétrica: muestra conexiones origen → destino donde al menos uno
de los extremos es EUROMED. Las zonas externas aparecen tanto en filas
como en columnas.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import sys

# =============================================================================
# STANDALONE CONFIGURATION
# =============================================================================
# Output directory: relative to current working directory
ANALYSIS_DIR = Path.cwd() / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "default"

# Path to cube data (external storage)
CUBE_DATA_DIR = Path.home() / "proyectos/data/traffic/cube"

# Import ConnectivityCube from local module (same directory)
try:
    from cube import ConnectivityCube
except ImportError:
    # Try adding scripts directory to path
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from cube import ConnectivityCube
    except ImportError:
        # ConnectivityCube not available - some functions will fail
        ConnectivityCube = None

# =============================================================================
# CONSTANTS
# =============================================================================

# EUROMED zones (for filtering port calls, NOT as matrix entries)
EUROMED_ZONES = ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')

# External zones to include in matrix (as aggregate entries)
EXTERNAL_ZONES = [
    'ASIA', 'INDIA', 'MIDDLE EAST', 'SOUTH AFRICA',
    'WEST AFRICA', 'NA ATLANTIC', 'CARIBE', 'SA ATLANTIC'
]

# Vessel categories to include (New Panamax and ULCV)
VESSEL_CATEGORIES = ('4 New Panamax', '5 ULCV')

# Number of top ports to include
TOP_N_PORTS = 1000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MatrixSnapshot:
    """Matriz de conectividad para un período."""
    label: str                          # Period label (e.g., "Q3 2024")
    direction: str                      # 'incoming' or 'outgoing'
    axis_labels: List[str]              # Row/column labels (39 entries)
    matrix_calls: np.ndarray            # 39x39 matrix of call counts
    matrix_teus: np.ndarray             # 39x39 matrix of TEU totals
    matrix_nm: np.ndarray               # 39x39 matrix of nautical miles
    matrix_ktm: np.ndarray              # 39x39 matrix of kilo-TEU-miles

    # Summary metrics
    total_calls: int
    total_teus: int
    total_nm: int                       # Total nautical miles
    total_ktm: int                      # Total kilo-TEU-miles

    # Port classification
    top_ports: List[str]                # Top 30 EUROMED ports by KTM
    otros_euromed_ports: List[str]      # Ports aggregated into "Otros EUROMED"


@dataclass
class MatrixMetrics:
    """Métricas de comparación entre dos matrices."""
    direction: str                      # 'incoming' or 'outgoing'

    # Absolute changes
    delta_calls: np.ndarray             # T2 - T1 calls
    delta_teus: np.ndarray              # T2 - T1 TEUs
    delta_nm: np.ndarray                # T2 - T1 nautical miles
    delta_ktm: np.ndarray               # T2 - T1 kilo-TEU-miles

    # Summary
    total_delta_calls: int
    total_delta_teus: int
    total_delta_nm: int                 # Total change in nautical miles
    total_delta_ktm: int                # Total change in kilo-TEU-miles

    # Hotspots (top changes by absolute delta) - now includes ktm
    top_gains: List[Tuple[str, str, int, int, int]]    # (origin, dest, delta_calls, delta_teus, delta_ktm)
    top_losses: List[Tuple[str, str, int, int, int]]   # (origin, dest, delta_calls, delta_teus, delta_ktm)


@dataclass
class MatrixAnalysis:
    """Resultado completo del análisis para exportar a JSON."""
    period1: str
    period2: str
    date_range1: Dict[str, str]
    date_range2: Dict[str, str]
    axis_labels: List[str]
    top_ports: List[str]

    # Snapshots for both periods and directions
    incoming_t1: MatrixSnapshot
    incoming_t2: MatrixSnapshot
    outgoing_t1: MatrixSnapshot
    outgoing_t2: MatrixSnapshot

    # Comparison metrics
    incoming_metrics: MatrixMetrics
    outgoing_metrics: MatrixMetrics

    # Output files
    files: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# AXIS LABEL GENERATION
# =============================================================================

def get_axis_labels(rows: List[Dict], top_n: int = TOP_N_PORTS) -> Tuple[List[str], List[str], List[str]]:
    """
    Determina las etiquetas del eje a partir de los resultados de query.

    Espera rows con: portname, zone, ktm (para ranking)

    Returns:
        (axis_labels, top_ports, otros_ports)
        - axis_labels: 39 entries (30 top ports + "Otros EUROMED" + 8 external zones)
        - top_ports: List of top 30 EUROMED port names
        - otros_ports: List of EUROMED ports not in top 30
    """
    # Aggregate KTM by port
    port_ktm = {}
    port_zone = {}

    for row in rows:
        port = row.get('portname')
        zone = row.get('zone')
        ktm = row.get('ktm', 0) or 0

        if port and zone in EUROMED_ZONES:
            port_ktm[port] = port_ktm.get(port, 0) + float(ktm)
            port_zone[port] = zone

    # Sort by KTM and get top N
    sorted_ports = sorted(port_ktm.items(), key=lambda x: x[1], reverse=True)
    top_ports = [p[0] for p in sorted_ports[:top_n]]
    otros_ports = [p[0] for p in sorted_ports[top_n:]]

    # Build axis labels: top 30 + "Otros EUROMED" + 8 external zones
    axis_labels = top_ports + ["Otros EUROMED"] + EXTERNAL_ZONES

    return axis_labels, top_ports, otros_ports


def map_to_axis(port: str, zone: str, top_ports: List[str]) -> Optional[str]:
    """
    Mapea un puerto a su posición en el eje.

    Returns:
        - Port name if in top_ports
        - "Otros EUROMED" if EUROMED but not in top
        - Zone name if in EXTERNAL_ZONES
        - None if unknown/excluded
    """
    if port in top_ports:
        return port
    elif zone in EUROMED_ZONES:
        return "Otros EUROMED"
    elif zone in EXTERNAL_ZONES:
        return zone
    else:
        return None  # Exclude (not in our scope)


# =============================================================================
# PRIMITIVA 1: snapshot()
# =============================================================================

def snapshot(rows: List[Dict], label: str, direction: str,
             axis_labels: List[str], top_ports: List[str]) -> MatrixSnapshot:
    """
    Crea matriz de conectividad desde resultados de query.

    Args:
        rows: Lista de dicts con origin, destination, origin_zone, dest_zone, calls, teus, nm, ktm
        label: Etiqueta del período (ej: "Q3 2024")
        direction: 'incoming' o 'outgoing'
        axis_labels: Lista de 39 etiquetas para los ejes
        top_ports: Lista de top 30 puertos para mapeo

    Returns:
        MatrixSnapshot con matrices 39×39 (calls, teus, nm, ktm)
    """
    n = len(axis_labels)
    matrix_calls = np.zeros((n, n), dtype=int)
    matrix_teus = np.zeros((n, n), dtype=int)
    matrix_nm = np.zeros((n, n), dtype=int)
    matrix_ktm = np.zeros((n, n), dtype=int)

    label_to_idx = {label: i for i, label in enumerate(axis_labels)}
    otros_ports = []  # Track which ports went to "Otros EUROMED"

    for row in rows:
        origin = row.get('origin')
        destination = row.get('destination')
        origin_zone = row.get('origin_zone')
        dest_zone = row.get('dest_zone')
        calls = int(row.get('calls', 0) or 0)
        teus = int(row.get('teus', 0) or 0)
        nm = int(row.get('nm', 0) or 0)
        ktm = int(row.get('ktm', 0) or 0)

        # Map to axis positions
        origin_mapped = map_to_axis(origin, origin_zone, top_ports)
        dest_mapped = map_to_axis(destination, dest_zone, top_ports)

        if origin_mapped is None or dest_mapped is None:
            continue

        if origin_mapped not in label_to_idx or dest_mapped not in label_to_idx:
            continue

        i = label_to_idx[origin_mapped]
        j = label_to_idx[dest_mapped]

        matrix_calls[i, j] += calls
        matrix_teus[i, j] += teus
        matrix_nm[i, j] += nm
        matrix_ktm[i, j] += ktm

        # Track otros ports
        if origin_mapped == "Otros EUROMED" and origin not in otros_ports:
            otros_ports.append(origin)
        if dest_mapped == "Otros EUROMED" and destination not in otros_ports:
            otros_ports.append(destination)

    return MatrixSnapshot(
        label=label,
        direction=direction,
        axis_labels=axis_labels,
        matrix_calls=matrix_calls,
        matrix_teus=matrix_teus,
        matrix_nm=matrix_nm,
        matrix_ktm=matrix_ktm,
        total_calls=int(matrix_calls.sum()),
        total_teus=int(matrix_teus.sum()),
        total_nm=int(matrix_nm.sum()),
        total_ktm=int(matrix_ktm.sum()),
        top_ports=top_ports,
        otros_euromed_ports=otros_ports
    )


# =============================================================================
# PRIMITIVA 2: compare()
# =============================================================================

def compare(snap1: MatrixSnapshot, snap2: MatrixSnapshot, n_hotspots: int = 5) -> MatrixMetrics:
    """
    Compara dos matrices y calcula métricas.

    Args:
        snap1: Snapshot período 1 (referencia)
        snap2: Snapshot período 2 (comparar)
        n_hotspots: Número de hotspots a identificar por categoría

    Returns:
        MatrixMetrics con deltas y hotspots (incluyendo nm y ktm)
    """
    delta_calls = snap2.matrix_calls - snap1.matrix_calls
    delta_teus = snap2.matrix_teus - snap1.matrix_teus
    delta_nm = snap2.matrix_nm - snap1.matrix_nm
    delta_ktm = snap2.matrix_ktm - snap1.matrix_ktm

    # Find top gains and losses
    axis_labels = snap1.axis_labels
    n = len(axis_labels)

    # Flatten and sort by delta_ktm (more meaningful than calls for transport work)
    changes = []
    for i in range(n):
        for j in range(n):
            if delta_calls[i, j] != 0 or delta_ktm[i, j] != 0:
                changes.append((
                    axis_labels[i],  # origin
                    axis_labels[j],  # destination
                    int(delta_calls[i, j]),
                    int(delta_teus[i, j]),
                    int(delta_ktm[i, j])
                ))

    # Sort by absolute delta_ktm (transport work is more meaningful than call count)
    changes.sort(key=lambda x: abs(x[4]), reverse=True)

    # Separate gains and losses
    gains = [c for c in changes if c[4] > 0][:n_hotspots]
    losses = [c for c in changes if c[4] < 0][:n_hotspots]

    return MatrixMetrics(
        direction=snap1.direction,
        delta_calls=delta_calls,
        delta_teus=delta_teus,
        delta_nm=delta_nm,
        delta_ktm=delta_ktm,
        total_delta_calls=int(delta_calls.sum()),
        total_delta_teus=int(delta_teus.sum()),
        total_delta_nm=int(delta_nm.sum()),
        total_delta_ktm=int(delta_ktm.sum()),
        top_gains=gains,
        top_losses=losses
    )


# =============================================================================
# PRIMITIVA 3: render()
# =============================================================================

def render(snap1_in: MatrixSnapshot, snap2_in: MatrixSnapshot,
           snap1_out: MatrixSnapshot, snap2_out: MatrixSnapshot,
           metrics_in: MatrixMetrics, metrics_out: MatrixMetrics,
           filename: str = None, dashboard: bool = False,
           output_dir: Path = None) -> str:
    """
    Genera PNG con 6 paneles para análisis de conectividad.

    Layout:
        Fila 1: INCOMING T1 | INCOMING T2 | INCOMING Delta
        Fila 2: OUTGOING T1 | OUTGOING T2 | OUTGOING Delta
        Fila 3: Resumen + Hotspots

    Args:
        snap1_in, snap2_in: Snapshots incoming T1, T2
        snap1_out, snap2_out: Snapshots outgoing T1, T2
        metrics_in, metrics_out: Métricas de comparación
        filename: Nombre archivo (default: auto)
        dashboard: Si True, guarda como 'matrix_dashboard.png'
        output_dir: Directorio de salida (default: ./analysis/default)

    Returns:
        Path del archivo guardado
    """
    # Use provided output_dir or default
    out_dir = output_dir if output_dir else OUTPUT_DIR

    # Layout: 6 heatmaps (2 filas × 3 cols) + tabla
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[4, 4, 1.5], hspace=0.25, top=0.92, bottom=0.05)

    axes_in = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_out = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_table = fig.add_subplot(gs[2, :])

    axis_labels = snap1_in.axis_labels
    n = len(axis_labels)

    # Prepare abbreviated labels for display
    def abbrev(label: str) -> str:
        if len(label) <= 12:
            return label
        # Common abbreviations
        abbrevs = {
            'Otros EUROMED': 'Otros EUR',
            'MIDDLE EAST': 'MID EAST',
            'SOUTH AFRICA': 'S. AFRICA',
            'WEST AFRICA': 'W. AFRICA',
            'NA ATLANTIC': 'NA ATL',
            'SA ATLANTIC': 'SA ATL'
        }
        return abbrevs.get(label, label[:10] + '.')

    short_labels = [abbrev(l) for l in axis_labels]

    # Get max values for color scaling
    max_calls = max(
        snap1_in.matrix_calls.max(),
        snap2_in.matrix_calls.max(),
        snap1_out.matrix_calls.max(),
        snap2_out.matrix_calls.max(),
        1
    )

    def plot_matrix(ax, matrix, title, cmap='Blues', show_delta=False, vmax=None):
        """Plot a single matrix heatmap."""
        if show_delta:
            # Diverging colormap for delta
            vmax_abs = max(abs(matrix.min()), abs(matrix.max()), 1)
            masked = np.ma.masked_where(matrix == 0, matrix)
            im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs, aspect='auto')
        else:
            # Sequential colormap with log scale for counts
            masked = np.ma.masked_where(matrix == 0, matrix)
            ax.set_facecolor('white')
            if matrix.max() > 0:
                im = ax.imshow(masked, cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax or max_calls), aspect='auto')
            else:
                im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(short_labels, fontsize=6)

        # Add grid lines to separate top30, otros, zones
        for pos in [TOP_N_PORTS - 0.5, TOP_N_PORTS + 0.5]:
            ax.axhline(y=pos, color='gray', linewidth=0.5, linestyle='--')
            ax.axvline(x=pos, color='gray', linewidth=0.5, linestyle='--')

        plt.colorbar(im, ax=ax, shrink=0.6, label='Escalas' if not show_delta else 'Δ')
        return im

    # Row 1: INCOMING
    plot_matrix(axes_in[0], snap1_in.matrix_calls, f'INCOMING {snap1_in.label}\n(n={snap1_in.total_calls:,})', 'Blues')
    plot_matrix(axes_in[1], snap2_in.matrix_calls, f'INCOMING {snap2_in.label}\n(n={snap2_in.total_calls:,})', 'Blues')
    plot_matrix(axes_in[2], metrics_in.delta_calls, f'Δ INCOMING\n(Δn={metrics_in.total_delta_calls:+,})', show_delta=True)

    axes_in[0].set_ylabel('Origen', fontsize=9)
    axes_in[0].set_xlabel('Destino', fontsize=9)

    # Row 2: OUTGOING
    plot_matrix(axes_out[0], snap1_out.matrix_calls, f'OUTGOING {snap1_out.label}\n(n={snap1_out.total_calls:,})', 'Oranges')
    plot_matrix(axes_out[1], snap2_out.matrix_calls, f'OUTGOING {snap2_out.label}\n(n={snap2_out.total_calls:,})', 'Oranges')
    plot_matrix(axes_out[2], metrics_out.delta_calls, f'Δ OUTGOING\n(Δn={metrics_out.total_delta_calls:+,})', show_delta=True)

    axes_out[0].set_ylabel('Origen', fontsize=9)
    axes_out[0].set_xlabel('Destino', fontsize=9)

    # Table: Summary + Hotspots
    ax_table.axis('off')

    def fmt_num(n):
        if abs(n) >= 1e6:
            return f'{n/1e6:.1f}M'
        elif abs(n) >= 1e3:
            return f'{n/1e3:.1f}K'
        return f'{n:,}'

    # Summary table
    summary_data = [
        ['', snap1_in.label, snap2_in.label, 'Δ', 'Δ%'],
        ['Escalas IN', f'{snap1_in.total_calls:,}', f'{snap2_in.total_calls:,}',
         f'{metrics_in.total_delta_calls:+,}',
         f'{(metrics_in.total_delta_calls/snap1_in.total_calls*100):+.1f}%' if snap1_in.total_calls else '-'],
        ['Escalas OUT', f'{snap1_out.total_calls:,}', f'{snap2_out.total_calls:,}',
         f'{metrics_out.total_delta_calls:+,}',
         f'{(metrics_out.total_delta_calls/snap1_out.total_calls*100):+.1f}%' if snap1_out.total_calls else '-'],
        ['TEUs IN', fmt_num(snap1_in.total_teus), fmt_num(snap2_in.total_teus),
         f'{metrics_in.total_delta_teus:+,}',
         f'{(metrics_in.total_delta_teus/snap1_in.total_teus*100):+.1f}%' if snap1_in.total_teus else '-'],
        ['TEUs OUT', fmt_num(snap1_out.total_teus), fmt_num(snap2_out.total_teus),
         f'{metrics_out.total_delta_teus:+,}',
         f'{(metrics_out.total_delta_teus/snap1_out.total_teus*100):+.1f}%' if snap1_out.total_teus else '-'],
    ]

    table1 = ax_table.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='left',
        cellLoc='right',
        colWidths=[0.08, 0.06, 0.06, 0.06, 0.05],
        bbox=[0, 0.2, 0.31, 0.8]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)

    for j in range(5):
        table1[(0, j)].set_facecolor('#e6e6e6')
        table1[(0, j)].set_text_props(fontweight='bold')

    # Hotspots tables
    def format_hotspots(hotspots: List[Tuple], title: str, x_offset: float, color: str):
        if not hotspots:
            return

        data = [[h[0][:15], h[1][:15], f'{h[2]:+d}', fmt_num(h[3])] for h in hotspots[:5]]

        table = ax_table.table(
            cellText=data,
            colLabels=['Origen', 'Destino', 'ΔEsc', 'ΔTEUs'],
            loc='left',
            cellLoc='left',
            colWidths=[0.10, 0.10, 0.05, 0.06],
            bbox=[x_offset, 0.2, 0.31, 0.8]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        for j in range(4):
            table[(0, j)].set_facecolor(color)
            table[(0, j)].set_text_props(fontweight='bold')

        ax_table.text(x_offset + 0.155, 1.0, title, ha='center', va='bottom',
                      fontsize=10, fontweight='bold', transform=ax_table.transAxes)

    format_hotspots(metrics_in.top_gains + metrics_out.top_gains, 'TOP GANANCIAS', 0.35, '#c8e6c9')
    format_hotspots(metrics_in.top_losses + metrics_out.top_losses, 'TOP PÉRDIDAS', 0.68, '#ffcdd2')

    # Title
    fig.suptitle(f'Matriz de Conectividad | {snap1_in.label} vs {snap2_in.label}',
                 fontsize=14, fontweight='bold')

    # Determine filename
    if dashboard:
        filename = "matrix_dashboard.png"
    elif filename is None:
        filename = f"matrix_{snap1_in.label}_vs_{snap2_in.label}.png".replace(" ", "_")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close()

    return str(output_path)


# =============================================================================
# PERIOD PARSING (from heatmap.py)
# =============================================================================

def parse_period(year: int, period: str = None) -> tuple:
    """
    Convierte año + período en rango de fechas.

    Args:
        year: Año (ej: 2023)
        period: M1-M12 (mes), Q1-Q4 (trimestre), None (año completo)

    Returns:
        (start_date, end_date, label) como strings
    """
    if period is None or period == "":
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        label = str(year)
        return start, end, label

    period = period.upper()

    if period.startswith("Q"):
        q = int(period[1])
        if q < 1 or q > 4:
            raise ValueError(f"Trimestre inválido: {period}. Usar Q1-Q4")
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 3
        start = f"{year}-{start_month:02d}-01"
        if end_month > 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{end_month:02d}-01"
        label = f"{year} {period}"

    elif period.startswith("M"):
        m = int(period[1:])
        if m < 1 or m > 12:
            raise ValueError(f"Mes inválido: {period}. Usar M1-M12")
        start = f"{year}-{m:02d}-01"
        if m == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{m + 1:02d}-01"
        label = f"{year} {period}"

    else:
        raise ValueError(f"Período inválido: {period}. Usar M1-M12 o Q1-Q4")

    return start, end, label


# =============================================================================
# SQL QUERY BUILDERS
# =============================================================================

def build_top_ports_query(start: str, end: str) -> str:
    """
    Genera query para obtener top 30 puertos EUROMED por KTM.
    """
    return f"""
SELECT
    e.portname,
    p.zone,
    COUNT(*) as calls,
    SUM(f.teus) as total_teus,
    ROUND(SUM(f.teus * (COALESCE(e.prev_leg, 0) + COALESCE(e.next_leg, 0)) / 2) / 1000, 0) as ktm
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
WHERE f.fleet = 'containers'
  AND f.category IN ('4 New Panamax', '5 ULCV')
  AND f.teus > 0
  AND p.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
  AND e.start >= '{start}' AND e.start < '{end}'
GROUP BY e.portname, p.zone
ORDER BY ktm DESC
LIMIT 50
""".replace('\n', ' ').strip()


def build_incoming_query(start: str, end: str) -> str:
    """
    Genera query para matriz incoming (prev_port → portname).
    Origen puede ser puerto EUROMED o zona externa.
    Destino siempre es puerto EUROMED.
    Incluye nm (millas) y ktm (kilo-TEU-miles).
    """
    return f"""
SELECT
    CASE
        WHEN p_prev.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
        THEN e.prev_port
        WHEN p_prev.zone IN ('ASIA', 'INDIA', 'MIDDLE EAST', 'SOUTH AFRICA', 'WEST AFRICA', 'NA ATLANTIC', 'CARIBE', 'SA ATLANTIC')
        THEN p_prev.zone
        ELSE NULL
    END as origin,
    e.portname as destination,
    p_prev.zone as origin_zone,
    p.zone as dest_zone,
    COUNT(*) as calls,
    SUM(f.teus) as teus,
    SUM(COALESCE(e.prev_leg, 0)) as nm,
    ROUND(SUM(f.teus * COALESCE(e.prev_leg, 0)) / 1000, 0) as ktm
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
LEFT JOIN ports p_prev ON e.prev_port = p_prev.portname
WHERE f.fleet = 'containers'
  AND f.category IN ('4 New Panamax', '5 ULCV')
  AND f.teus > 0
  AND p.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
  AND e.start >= '{start}' AND e.start < '{end}'
  AND e.prev_port IS NOT NULL
GROUP BY origin, e.portname, p_prev.zone, p.zone
HAVING origin IS NOT NULL
ORDER BY ktm DESC
""".replace('\n', ' ').strip()


def build_outgoing_query(start: str, end: str) -> str:
    """
    Genera query para matriz outgoing (portname → next_port).
    Origen siempre es puerto EUROMED.
    Destino puede ser puerto EUROMED o zona externa.
    Incluye nm (millas) y ktm (kilo-TEU-miles).
    """
    return f"""
SELECT
    e.portname as origin,
    CASE
        WHEN p_next.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
        THEN e.next_port
        WHEN p_next.zone IN ('ASIA', 'INDIA', 'MIDDLE EAST', 'SOUTH AFRICA', 'WEST AFRICA', 'NA ATLANTIC', 'CARIBE', 'SA ATLANTIC')
        THEN p_next.zone
        ELSE NULL
    END as destination,
    p.zone as origin_zone,
    p_next.zone as dest_zone,
    COUNT(*) as calls,
    SUM(f.teus) as teus,
    SUM(COALESCE(e.next_leg, 0)) as nm,
    ROUND(SUM(f.teus * COALESCE(e.next_leg, 0)) / 1000, 0) as ktm
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
LEFT JOIN ports p_next ON e.next_port = p_next.portname
WHERE f.fleet = 'containers'
  AND f.category IN ('4 New Panamax', '5 ULCV')
  AND f.teus > 0
  AND p.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
  AND e.start >= '{start}' AND e.start < '{end}'
  AND e.next_port IS NOT NULL
GROUP BY e.portname, destination, p.zone, p_next.zone
HAVING destination IS NOT NULL
ORDER BY ktm DESC
""".replace('\n', ' ').strip()


def build_symmetric_query(start: str, end: str) -> str:
    """
    Genera query para matriz simétrica (origen → destino).
    Incluye conexiones donde al menos uno de los extremos es EUROMED.
    Incluye nm (millas) y ktm (kilo-TEU-miles).
    """
    return f"""
SELECT
    CASE
        WHEN p_orig.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
        THEN e.portname
        WHEN p_orig.zone IN ('ASIA', 'INDIA', 'MIDDLE EAST', 'SOUTH AFRICA', 'WEST AFRICA', 'NA ATLANTIC', 'CARIBE', 'SA ATLANTIC')
        THEN p_orig.zone
        ELSE NULL
    END as origin,
    CASE
        WHEN p_next.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
        THEN e.next_port
        WHEN p_next.zone IN ('ASIA', 'INDIA', 'MIDDLE EAST', 'SOUTH AFRICA', 'WEST AFRICA', 'NA ATLANTIC', 'CARIBE', 'SA ATLANTIC')
        THEN p_next.zone
        ELSE NULL
    END as destination,
    p_orig.zone as origin_zone,
    p_next.zone as dest_zone,
    COUNT(*) as calls,
    SUM(f.teus) as teus,
    SUM(COALESCE(e.next_leg, 0)) as nm,
    ROUND(SUM(f.teus * COALESCE(e.next_leg, 0)) / 1000, 0) as ktm
FROM v_escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p_orig ON e.portname = p_orig.portname
LEFT JOIN ports p_next ON e.next_port = p_next.portname
WHERE f.fleet = 'containers'
  AND f.category IN ('4 New Panamax', '5 ULCV')
  AND f.teus > 0
  AND (p_orig.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED')
       OR p_next.zone IN ('NORTH EUROPE', 'ATLANTIC', 'WEST MED', 'EAST MED'))
  AND e.start >= '{start}' AND e.start < '{end}'
  AND e.next_port IS NOT NULL
GROUP BY origin, destination, p_orig.zone, p_next.zone
HAVING origin IS NOT NULL AND destination IS NOT NULL
ORDER BY ktm DESC
""".replace('\n', ' ').strip()


# =============================================================================
# RENDER SYMMETRIC (3 paneles)
# =============================================================================

def render_symmetric(snap1: MatrixSnapshot, snap2: MatrixSnapshot,
                     metrics: MatrixMetrics, filename: str = None,
                     dashboard: bool = False, output_dir: Path = None) -> str:
    """
    Genera PNG con 3 paneles para matriz simétrica.

    Layout:
        T1 | T2 | Delta
        Resumen + Hotspots

    Args:
        snap1, snap2: Snapshots T1, T2
        metrics: Métricas de comparación
        filename: Nombre archivo (default: auto)
        dashboard: Si True, guarda como 'matrix_dashboard.png'
        output_dir: Directorio de salida (default: ./analysis/default)

    Returns:
        Path del archivo guardado
    """
    # Use provided output_dir or default
    out_dir = output_dir if output_dir else OUTPUT_DIR

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 1.2], hspace=0.25, top=0.90, bottom=0.08)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_table = fig.add_subplot(gs[1, :])

    axis_labels = snap1.axis_labels
    n = len(axis_labels)

    # Abbreviated labels
    def abbrev(label: str) -> str:
        abbrevs = {
            'Otros EUROMED': 'Otros EUR', 'MIDDLE EAST': 'MID EAST',
            'SOUTH AFRICA': 'S.AFRICA', 'WEST AFRICA': 'W.AFRICA',
            'NA ATLANTIC': 'NA ATL', 'SA ATLANTIC': 'SA ATL'
        }
        return abbrevs.get(label, label[:12] if len(label) > 12 else label)

    short_labels = [abbrev(l) for l in axis_labels]
    max_calls = max(snap1.matrix_calls.max(), snap2.matrix_calls.max(), 1)

    def plot_matrix(ax, matrix, title, cmap='YlOrRd', show_delta=False):
        if show_delta:
            vmax_abs = max(abs(matrix.min()), abs(matrix.max()), 1)
            masked = np.ma.masked_where(matrix == 0, matrix)
            im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs, aspect='auto')
            cbar_label = 'Δ Escalas'
        else:
            masked = np.ma.masked_where(matrix == 0, matrix)
            ax.set_facecolor('white')
            if matrix.max() > 0:
                im = ax.imshow(masked, cmap=cmap, norm=LogNorm(vmin=1, vmax=max_calls), aspect='auto')
            else:
                im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
            cbar_label = 'Escalas'

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
        ax.set_yticklabels(short_labels, fontsize=5)

        for pos in [TOP_N_PORTS - 0.5, TOP_N_PORTS + 0.5]:
            ax.axhline(y=pos, color='gray', linewidth=0.5, linestyle='--')
            ax.axvline(x=pos, color='gray', linewidth=0.5, linestyle='--')

        plt.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)
        ax.set_xlabel('Destino', fontsize=9)
        ax.set_ylabel('Origen', fontsize=9)

    plot_matrix(axes[0], snap1.matrix_calls, f'{snap1.label}\n({snap1.total_calls:,} conexiones)')
    plot_matrix(axes[1], snap2.matrix_calls, f'{snap2.label}\n({snap2.total_calls:,} conexiones)')
    plot_matrix(axes[2], metrics.delta_calls, f'Δ Cambio\n({metrics.total_delta_calls:+,})', show_delta=True)

    # Table
    ax_table.axis('off')

    def fmt_num(n):
        if abs(n) >= 1e6: return f'{n/1e6:.1f}M'
        elif abs(n) >= 1e3: return f'{n/1e3:.1f}K'
        return f'{n:,}'

    summary = [
        ['', snap1.label, snap2.label, 'Δ', 'Δ%'],
        ['Conexiones', f'{snap1.total_calls:,}', f'{snap2.total_calls:,}',
         f'{metrics.total_delta_calls:+,}',
         f'{metrics.total_delta_calls/snap1.total_calls*100:+.1f}%' if snap1.total_calls else '-'],
        ['TEUs', fmt_num(snap1.total_teus), fmt_num(snap2.total_teus),
         fmt_num(metrics.total_delta_teus),
         f'{metrics.total_delta_teus/snap1.total_teus*100:+.1f}%' if snap1.total_teus else '-'],
    ]

    table1 = ax_table.table(cellText=summary[1:], colLabels=summary[0], loc='left', cellLoc='right',
                            colWidths=[0.08, 0.06, 0.06, 0.06, 0.05], bbox=[0.02, 0.2, 0.28, 0.7])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    for j in range(5):
        table1[(0, j)].set_facecolor('#e6e6e6')
        table1[(0, j)].set_text_props(fontweight='bold')

    # Hotspots
    if metrics.top_gains:
        gains = [[h[0][:14], h[1][:14], f'{h[2]:+d}'] for h in metrics.top_gains[:6]]
        t_gains = ax_table.table(cellText=gains, colLabels=['Origen', 'Destino', 'Δ'], loc='left', cellLoc='left',
                                 colWidths=[0.10, 0.10, 0.05], bbox=[0.35, 0.1, 0.25, 0.85])
        t_gains.auto_set_font_size(False)
        t_gains.set_fontsize(8)
        for j in range(3):
            t_gains[(0, j)].set_facecolor('#c8e6c9')
            t_gains[(0, j)].set_text_props(fontweight='bold')
        ax_table.text(0.475, 0.98, 'TOP GANANCIAS', ha='center', fontsize=10, fontweight='bold', transform=ax_table.transAxes)

    if metrics.top_losses:
        losses = [[h[0][:14], h[1][:14], f'{h[2]:+d}'] for h in metrics.top_losses[:6]]
        t_losses = ax_table.table(cellText=losses, colLabels=['Origen', 'Destino', 'Δ'], loc='left', cellLoc='left',
                                  colWidths=[0.10, 0.10, 0.05], bbox=[0.68, 0.1, 0.25, 0.85])
        t_losses.auto_set_font_size(False)
        t_losses.set_fontsize(8)
        for j in range(3):
            t_losses[(0, j)].set_facecolor('#ffcdd2')
            t_losses[(0, j)].set_text_props(fontweight='bold')
        ax_table.text(0.805, 0.98, 'TOP PÉRDIDAS', ha='center', fontsize=10, fontweight='bold', transform=ax_table.transAxes)

    fig.suptitle(f'Matriz de Conectividad | {snap1.label} vs {snap2.label} | ULCV (Cat 4-5)',
                 fontsize=13, fontweight='bold')

    if dashboard:
        filename = "matrix_dashboard.png"
    elif filename is None:
        filename = f"matrix_{snap1.label}_vs_{snap2.label}.png".replace(" ", "_")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(output_path)


# =============================================================================
# ANALYZE: Pipeline completo → JSON
# =============================================================================

def analyze(top_ports_csv: str, incoming_csv: str, outgoing_csv: str,
            year1: int, period1: str = None,
            year2: int = None, period2: str = None,
            output_dir: str = None) -> dict:
    """
    Pipeline completo: carga datos, compara períodos, genera visualización y JSON.

    Args:
        top_ports_csv: CSV con ranking de puertos (para determinar ejes)
        incoming_csv: CSV con datos incoming para ambos períodos
        outgoing_csv: CSV con datos outgoing para ambos períodos
        year1, period1: Primer período
        year2, period2: Segundo período
        output_dir: Carpeta de salida

    Returns:
        Dict JSON-serializable con métricas y rutas de archivos
    """
    import pandas as pd

    # Parse periods
    start1, end1, label1 = parse_period(year1, period1)
    start2, end2, label2 = parse_period(year2, period2)

    # Determine output directory
    is_default = output_dir is None
    if is_default:
        out_path = ANALYSIS_DIR / "default"
    else:
        out_path = ANALYSIS_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Load top ports data (from more recent period for axis ordering)
    df_top = pd.read_csv(top_ports_csv)
    axis_labels, top_ports, otros_ports = get_axis_labels(df_top.to_dict('records'))

    # Load incoming data
    df_in = pd.read_csv(incoming_csv)
    df_in['start'] = pd.to_datetime(df_in['start']) if 'start' in df_in.columns else None

    # Load outgoing data
    df_out = pd.read_csv(outgoing_csv)
    df_out['start'] = pd.to_datetime(df_out['start']) if 'start' in df_out.columns else None

    # Filter by period if start column exists
    if 'start' in df_in.columns:
        df_in1 = df_in[(df_in['start'] >= start1) & (df_in['start'] < end1)]
        df_in2 = df_in[(df_in['start'] >= start2) & (df_in['start'] < end2)]
    else:
        # Assume data is pre-filtered or has period column
        df_in1 = df_in[df_in.get('period', '') == label1] if 'period' in df_in.columns else df_in
        df_in2 = df_in[df_in.get('period', '') == label2] if 'period' in df_in.columns else df_in

    if 'start' in df_out.columns:
        df_out1 = df_out[(df_out['start'] >= start1) & (df_out['start'] < end1)]
        df_out2 = df_out[(df_out['start'] >= start2) & (df_out['start'] < end2)]
    else:
        df_out1 = df_out[df_out.get('period', '') == label1] if 'period' in df_out.columns else df_out
        df_out2 = df_out[df_out.get('period', '') == label2] if 'period' in df_out.columns else df_out

    # Create snapshots
    snap_in1 = snapshot(df_in1.to_dict('records'), label1, 'incoming', axis_labels, top_ports)
    snap_in2 = snapshot(df_in2.to_dict('records'), label2, 'incoming', axis_labels, top_ports)
    snap_out1 = snapshot(df_out1.to_dict('records'), label1, 'outgoing', axis_labels, top_ports)
    snap_out2 = snapshot(df_out2.to_dict('records'), label2, 'outgoing', axis_labels, top_ports)

    # Compare
    metrics_in = compare(snap_in1, snap_in2)
    metrics_out = compare(snap_out1, snap_out2)

    # Render
    if is_default:
        png_name = "matrix_default.png"
    else:
        png_name = f"matrix_{label1}_vs_{label2}.png".replace(" ", "_")

    png_path = render(snap_in1, snap_in2, snap_out1, snap_out2, metrics_in, metrics_out, png_name, output_dir=out_path)

    # Build JSON result
    result = {
        'period1': label1,
        'period2': label2,
        'date_range1': {'start': start1, 'end': end1},
        'date_range2': {'start': start2, 'end': end2},
        'axis_labels': axis_labels,
        'top_ports': top_ports,
        'incoming': {
            't1': {'calls': snap_in1.total_calls, 'teus': snap_in1.total_teus, 'nm': snap_in1.total_nm, 'ktm': snap_in1.total_ktm},
            't2': {'calls': snap_in2.total_calls, 'teus': snap_in2.total_teus, 'nm': snap_in2.total_nm, 'ktm': snap_in2.total_ktm},
            'delta': {'calls': metrics_in.total_delta_calls, 'teus': metrics_in.total_delta_teus, 'nm': metrics_in.total_delta_nm, 'ktm': metrics_in.total_delta_ktm},
            'hotspots': {
                'gains': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                          for h in metrics_in.top_gains],
                'losses': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                           for h in metrics_in.top_losses]
            }
        },
        'outgoing': {
            't1': {'calls': snap_out1.total_calls, 'teus': snap_out1.total_teus, 'nm': snap_out1.total_nm, 'ktm': snap_out1.total_ktm},
            't2': {'calls': snap_out2.total_calls, 'teus': snap_out2.total_teus, 'nm': snap_out2.total_nm, 'ktm': snap_out2.total_ktm},
            'delta': {'calls': metrics_out.total_delta_calls, 'teus': metrics_out.total_delta_teus, 'nm': metrics_out.total_delta_nm, 'ktm': metrics_out.total_delta_ktm},
            'hotspots': {
                'gains': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                          for h in metrics_out.top_gains],
                'losses': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                           for h in metrics_out.top_losses]
            }
        },
        'files': {'png': png_path}
    }

    # Save JSON
    if is_default:
        json_name = "matrix_default.json"
    else:
        json_name = f"matrix_{label1}_vs_{label2}.json".replace(" ", "_")
    json_path = out_path / json_name
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    result['files']['json'] = str(json_path)

    return result


def analyze_symmetric(top_ports_csv: str, data_csv: str,
                      year1: int, period1: str = None,
                      year2: int = None, period2: str = None,
                      output_dir: str = None,
                      cube_path: str = None) -> dict:
    """
    Pipeline simplificado para matriz simétrica.

    Args:
        top_ports_csv: CSV con ranking de puertos (para determinar ejes)
        data_csv: CSV con datos simétricos (origin, destination, calls, teus, period)
        year1, period1: Primer período
        year2, period2: Segundo período
        output_dir: Carpeta de salida
        cube_path: Ruta para guardar/actualizar ConnectivityCube (.npz)

    Returns:
        Dict JSON-serializable con métricas y rutas de archivos
    """
    import pandas as pd

    # Parse periods
    start1, end1, label1 = parse_period(year1, period1)
    start2, end2, label2 = parse_period(year2, period2)

    # Determine output directory
    is_default = output_dir is None
    if is_default:
        out_path = ANALYSIS_DIR / "default"
    else:
        out_path = ANALYSIS_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Load top ports data
    df_top = pd.read_csv(top_ports_csv)
    axis_labels, top_ports, otros_ports = get_axis_labels(df_top.to_dict('records'))

    # Load symmetric data
    df = pd.read_csv(data_csv)

    # Filter by period
    if 'period' in df.columns:
        df1 = df[df['period'] == label1]
        df2 = df[df['period'] == label2]
    else:
        df['start'] = pd.to_datetime(df['start'])
        df1 = df[(df['start'] >= start1) & (df['start'] < end1)]
        df2 = df[(df['start'] >= start2) & (df['start'] < end2)]

    # Create snapshots
    snap1 = snapshot(df1.to_dict('records'), label1, 'symmetric', axis_labels, top_ports)
    snap2 = snapshot(df2.to_dict('records'), label2, 'symmetric', axis_labels, top_ports)

    # Compare
    metrics = compare(snap1, snap2, n_hotspots=6)

    # Render
    if is_default:
        png_name = "matrix_default.png"
    else:
        png_name = f"matrix_{label1}_vs_{label2}.png".replace(" ", "_")

    png_path = render_symmetric(snap1, snap2, metrics, png_name, output_dir=out_path)

    # Build JSON result
    result = {
        'period1': label1,
        'period2': label2,
        'date_range1': {'start': start1, 'end': end1},
        'date_range2': {'start': start2, 'end': end2},
        'axis_labels': axis_labels,
        'top_ports': top_ports,
        'summary': {
            't1': {'calls': snap1.total_calls, 'teus': snap1.total_teus, 'nm': snap1.total_nm, 'ktm': snap1.total_ktm},
            't2': {'calls': snap2.total_calls, 'teus': snap2.total_teus, 'nm': snap2.total_nm, 'ktm': snap2.total_ktm},
            'delta': {'calls': metrics.total_delta_calls, 'teus': metrics.total_delta_teus, 'nm': metrics.total_delta_nm, 'ktm': metrics.total_delta_ktm},
        },
        'hotspots': {
            'gains': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                      for h in metrics.top_gains],
            'losses': [{'origin': h[0], 'dest': h[1], 'delta_calls': h[2], 'delta_teus': h[3], 'delta_ktm': h[4]}
                       for h in metrics.top_losses]
        },
        'files': {'png': png_path}
    }

    # Save JSON
    if is_default:
        json_name = "matrix_default.json"
    else:
        json_name = f"matrix_{label1}_vs_{label2}.json".replace(" ", "_")
    json_path = out_path / json_name
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    result['files']['json'] = str(json_path)

    # Save to ConnectivityCube if requested
    if cube_path and ConnectivityCube is not None:
        cube_file = Path(cube_path)
        if cube_file.exists():
            cube = ConnectivityCube(str(cube_file))
        else:
            cube = ConnectivityCube()

        # Add periods if not already present
        if label1 not in cube.periods:
            cube.add_period(label1, snap1.matrix_calls, snap1.matrix_teus,
                           snap1.matrix_nm, snap1.matrix_ktm, axis_labels)
        if label2 not in cube.periods:
            cube.add_period(label2, snap2.matrix_calls, snap2.matrix_teus,
                           snap2.matrix_nm, snap2.matrix_ktm)

        cube.save(str(cube_file))
        result['files']['cube'] = str(cube_file)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Port connectivity matrix comparison (symmetric)')
    parser.add_argument('--top-ports', required=True, help='CSV with top ports ranking')
    parser.add_argument('--data', required=True, help='CSV with symmetric connectivity data')
    parser.add_argument('--y1', type=int, required=True, help='First year')
    parser.add_argument('--p1', default=None, help='First period (Q1-Q4, M1-M12)')
    parser.add_argument('--y2', type=int, required=True, help='Second year')
    parser.add_argument('--p2', default=None, help='Second period (Q1-Q4, M1-M12)')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory')
    parser.add_argument('--default', action='store_true', help='Use analysis/default/')
    parser.add_argument('--cube', default=None, help='Path to ConnectivityCube (.npz) to save/update')

    args = parser.parse_args()

    output_dir = None if args.default else args.output_dir

    result = analyze_symmetric(
        top_ports_csv=args.top_ports,
        data_csv=args.data,
        year1=args.y1,
        period1=args.p1,
        year2=args.y2,
        period2=args.p2,
        output_dir=output_dir,
        cube_path=args.cube
    )

    print(json.dumps(result, indent=2))

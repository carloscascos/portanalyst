#!/usr/bin/env python3
"""
heatmap.py - Comparador de heatmaps de tráfico portuario (STANDALONE)

3 primitivas:
  - snapshot(rows): construye heatmap 2D desde resultados de query
  - compare(s1, s2): compara 2 heatmaps, retorna métricas + diferencia
  - render(s1, s2, metrics): genera PNG con 3 paneles

Uso desde Claude Code:

    # 1. Query directa a la base de datos
    result = db_query("SELECT ... WHERE port='London' AND ...")

    # 2. Crear snapshots
    s1 = snapshot(result1['rows'], "London", "Q3 2023", "prev_leg")
    s2 = snapshot(result2['rows'], "London", "Q3 2025", "prev_leg")

    # 3. Comparar y renderizar
    metrics = compare(s1, s2)
    render(s1, s2, metrics, "london_comparison.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os

# =============================================================================
# STANDALONE CONFIGURATION
# =============================================================================
# Output directory: relative to current working directory
ANALYSIS_DIR = Path.cwd() / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "default"

# =============================================================================
# COLOR STANDARDS - Consistent color schemes for all heatmaps
# =============================================================================
# Snapshots (T1, T2): Yellow → Red (intensity/volume)
CMAP_SNAPSHOT = 'YlOrRd'

# Delta (comparison): Blue (decrease) → Red (increase)
CMAP_DELTA = 'RdBu_r'

# =============================================================================
# SEMANTIC BINS - Variable width bins for meaningful segmentation
# =============================================================================

# TEU bins (vessel categories)
TEU_BINS = np.array([0, 3000, 5000, 10000, 15000, 25000], dtype=float)
TEU_LABELS = ['Feeder', 'Panamax', 'Post-Panamax', 'New Panamax', 'ULCV']

# Distance bins (route types)
DIST_BINS = np.array([0, 500, 1500, 4000, 8000, 15000], dtype=float)
DIST_LABELS = ['Coastal', 'Short-haul', 'Medium', 'Long-haul', 'Ultra-long']

# Fixed bins by default (semantic bins can be enabled via environment)
USE_FIXED_BINS = os.getenv("USE_FIXED_BINS", "true").lower() == "true"
DIST_BIN = int(os.getenv("DIST_BIN", 500))      # nm
TEU_BIN = int(os.getenv("TEU_BIN", 500))        # TEU
DIST_MAX = int(os.getenv("DIST_MAX", 15000))    # nm
TEU_MAX = int(os.getenv("TEU_MAX", 25000))      # TEU


def get_bins():
    """Returns (dist_bins, teu_bins) - semantic or fixed based on config."""
    if USE_FIXED_BINS:
        return (
            np.arange(0, DIST_MAX + DIST_BIN, DIST_BIN),
            np.arange(0, TEU_MAX + TEU_BIN, TEU_BIN)
        )
    return DIST_BINS, TEU_BINS


def get_bin_label(bin_type: str, bin_min: float, bin_max: float) -> str:
    """
    Returns human-readable label for a bin range.

    Args:
        bin_type: 'teu' or 'dist'
        bin_min, bin_max: Bin boundaries

    Returns:
        Label string like 'Feeder (0-3K TEU)' or 'Coastal (0-500nm)'
    """
    if USE_FIXED_BINS:
        # Fixed bins: just show range
        if bin_type == 'teu':
            return f"{bin_min/1000:.0f}-{bin_max/1000:.0f}K TEU"
        else:
            return f"{bin_min:.0f}-{bin_max:.0f}nm"

    # Semantic bins: find matching label
    if bin_type == 'teu':
        bins, labels = TEU_BINS, TEU_LABELS
        unit = 'TEU'
    else:
        bins, labels = DIST_BINS, DIST_LABELS
        unit = 'nm'

    for i in range(len(bins) - 1):
        if bins[i] == bin_min and bins[i+1] == bin_max:
            if bin_type == 'teu':
                return f"{labels[i]} ({bin_min/1000:.0f}-{bin_max/1000:.0f}K)"
            else:
                return f"{labels[i]} ({bin_min:.0f}-{bin_max:.0f}nm)"

    # Fallback
    if bin_type == 'teu':
        return f"{bin_min/1000:.0f}-{bin_max/1000:.0f}K TEU"
    return f"{bin_min:.0f}-{bin_max:.0f}nm"


@dataclass
class Snapshot:
    """Histograma 2D de un período."""
    label: str
    port: str
    leg_type: str
    histogram: np.ndarray
    n_calls: int
    dist_bins: np.ndarray
    teu_bins: np.ndarray
    total_teus: int
    total_nm: float
    total_ktm: float
    mean_dist: float  # Distancia media ponderada por TEU
    avg_hours: float = 0.0  # Horas medias de escala
    total_hours: float = 0.0  # Horas totales de operación
    total_hour_teu: float = 0.0  # Hora-TEU (tiempo ponderado por capacidad)

    # Campos bidireccionales (solo se usan cuando leg_type='bidirectional')
    incoming_histogram: np.ndarray = field(default=None, repr=False)
    outgoing_histogram: np.ndarray = field(default=None, repr=False)
    incoming_calls: int = 0
    outgoing_calls: int = 0
    incoming_teus: int = 0
    outgoing_teus: int = 0
    incoming_ktm: float = 0.0
    outgoing_ktm: float = 0.0
    directional_ratio: float = 1.0  # outgoing/incoming (TEUs)


@dataclass
class Metrics:
    """Métricas de comparación."""
    jsd: float
    hellinger: float
    emd: float
    diff: np.ndarray
    delta_calls: int

    # Campos bidireccionales (solo se usan cuando snapshots son bidireccionales)
    jsd_incoming: float = None
    jsd_outgoing: float = None
    diff_incoming: np.ndarray = field(default=None, repr=False)
    diff_outgoing: np.ndarray = field(default=None, repr=False)
    directional_shift: float = None  # Cambio en ratio O/I entre períodos

    @property
    def status(self) -> str:
        # Umbrales para bins fijos (30x50=1500 bins)
        if self.jsd < 0.10:
            return "ESTABLE"
        elif self.jsd < 0.25:
            return "CAMBIO MENOR"
        elif self.jsd < 0.40:
            return "CAMBIO MODERADO"
        else:
            return "CAMBIO SIGNIFICATIVO"


@dataclass
class Hotspot:
    """Un punto de interés identificado en el histograma."""
    rank: int
    direction: str  # 'incoming' o 'outgoing'
    segment: str    # 'small' o 'large'
    dist_range: tuple
    teu_range: tuple
    delta_pct: float
    change: str     # 'GANANCIA' o 'PÉRDIDA'
    dist_label: str = ""  # Human-readable distance label
    teu_label: str = ""   # Human-readable TEU label

    def to_dict(self) -> dict:
        return {
            'rank': self.rank,
            'direction': self.direction,
            'segment': self.segment,
            'dist_range': list(self.dist_range),
            'teu_range': list(self.teu_range),
            'dist_label': self.dist_label,
            'teu_label': self.teu_label,
            'delta_pct': round(self.delta_pct, 2),
            'change': self.change
        }


# Umbrales de segmentación por TEU (alineados con bins semánticos)
# Small = Feeder + Panamax (bins 0-3000, 3000-5000)
# Large = New Panamax + ULCV (bins 10000-15000, 15000+)
TEU_SMALL_MAX = 5000   # Feeders + Panamax: < 5000 TEU
TEU_LARGE_MIN = 10000  # New Panamax + ULCV: >= 10000 TEU


# =============================================================================
# PRIMITIVA 1: snapshot()
# =============================================================================

def snapshot(rows: List[Dict[str, Any]], port: str, label: str,
             leg_type: str = "bidirectional") -> Snapshot:
    """
    Crea snapshot (histograma 2D) desde resultados de query.

    Args:
        rows: Lista de dicts del resultado de db_query (result['rows'])
        port: Nombre del puerto (para metadatos)
        label: Etiqueta del período (ej: "Q3 2023")
        leg_type: 'bidirectional' (default), 'prev_leg', 'next_leg', o 'both'

    Returns:
        Snapshot con histograma 2D (y campos bidireccionales si leg_type='bidirectional')
    """
    dist_bins, teu_bins = get_bins()

    # Modo bidireccional: separar incoming y outgoing
    if leg_type == 'bidirectional':
        return _snapshot_bidirectional(rows, port, label, dist_bins, teu_bins)

    # Modo unidireccional (original)
    distances = []
    teus = []

    if leg_type == 'both':
        leg_cols = ['prev_leg', 'next_leg']
    else:
        leg_cols = [leg_type]

    for row in rows:
        teu = row.get('TEU') or row.get('teus') or row.get('teu')
        # Handle None, NaN, empty string
        try:
            teu = float(teu) if teu is not None and teu != '' else None
        except (ValueError, TypeError):
            teu = None
        if teu is None or teu != teu or teu <= 0:  # teu != teu checks for NaN
            continue

        for col in leg_cols:
            dist = row.get(col)
            # Handle None, NaN, empty string
            try:
                dist = float(dist) if dist is not None and dist != '' else None
            except (ValueError, TypeError):
                dist = None
            if dist is not None and dist == dist and dist >= 0:  # dist == dist excludes NaN
                distances.append(float(dist))
                teus.append(float(teu))

    # Calcular totales
    total_teus = int(sum(teus))
    total_nm = sum(distances)
    total_ktm = sum(t * d for t, d in zip(teus, distances)) / 1000  # Kilo-TEU-Miles
    mean_dist = sum(t * d for t, d in zip(teus, distances)) / total_teus if total_teus > 0 else 0

    # Crear histograma 2D
    if len(distances) > 0:
        hist, _, _ = np.histogram2d(distances, teus, bins=[dist_bins, teu_bins])
        hist = hist.T  # TEU en Y, distancia en X
    else:
        hist = np.zeros((len(teu_bins) - 1, len(dist_bins) - 1))

    return Snapshot(
        label=label,
        port=port,
        leg_type=leg_type,
        histogram=hist,
        n_calls=len(rows),
        dist_bins=dist_bins,
        teu_bins=teu_bins,
        total_teus=total_teus,
        total_nm=total_nm,
        total_ktm=total_ktm,
        mean_dist=mean_dist
    )


def _snapshot_bidirectional(rows: List[Dict[str, Any]], port: str, label: str,
                            dist_bins: np.ndarray, teu_bins: np.ndarray) -> Snapshot:
    """
    Crea snapshot bidireccional con histogramas separados para incoming/outgoing.

    Espera que cada row tenga:
        - 'direction': 'incoming' o 'outgoing'
        - 'leg': distancia del leg
        - 'teus' o 'TEU': capacidad del buque
        - 'duration' (opcional): horas de escala
    """
    # Separar datos por dirección
    in_distances, in_teus = [], []
    out_distances, out_teus = [], []
    # Para métricas de tiempo (solo incoming para no duplicar por escala)
    hours_list, hours_teus = [], []

    for row in rows:
        teu = row.get('TEU') or row.get('teus') or row.get('teu')
        # Handle None, NaN, empty string
        try:
            teu = float(teu) if teu is not None and teu != '' else None
        except (ValueError, TypeError):
            teu = None
        if teu is None or teu != teu or teu <= 0:  # teu != teu checks for NaN
            continue

        dist = row.get('leg')
        # Handle None, NaN, empty string, and negative values
        try:
            dist = float(dist) if dist is not None and dist != '' else None
        except (ValueError, TypeError):
            dist = None
        if dist is None or dist != dist or dist < 0:  # dist != dist checks for NaN
            continue

        direction = row.get('direction', 'incoming')
        if direction == 'incoming':
            in_distances.append(float(dist))
            in_teus.append(float(teu))
            # Capturar duration solo en incoming (cada escala aparece 2 veces)
            duration = row.get('duration')
            try:
                duration = float(duration) if duration is not None and duration != '' else None
            except (ValueError, TypeError):
                duration = None
            if duration is not None and duration == duration and duration >= 0:
                hours_list.append(duration)
                hours_teus.append(float(teu))
        else:  # outgoing
            out_distances.append(float(dist))
            out_teus.append(float(teu))

    # Métricas incoming
    incoming_teus = int(sum(in_teus))
    incoming_ktm = sum(t * d for t, d in zip(in_teus, in_distances)) / 1000 if in_teus else 0

    # Métricas outgoing
    outgoing_teus = int(sum(out_teus))
    outgoing_ktm = sum(t * d for t, d in zip(out_teus, out_distances)) / 1000 if out_teus else 0

    # Ratio direccional
    directional_ratio = outgoing_teus / incoming_teus if incoming_teus > 0 else 1.0

    # Histograma incoming
    if len(in_distances) > 0:
        in_hist, _, _ = np.histogram2d(in_distances, in_teus, bins=[dist_bins, teu_bins])
        in_hist = in_hist.T
    else:
        in_hist = np.zeros((len(teu_bins) - 1, len(dist_bins) - 1))

    # Histograma outgoing
    if len(out_distances) > 0:
        out_hist, _, _ = np.histogram2d(out_distances, out_teus, bins=[dist_bins, teu_bins])
        out_hist = out_hist.T
    else:
        out_hist = np.zeros((len(teu_bins) - 1, len(dist_bins) - 1))

    # Histograma combinado (para backward compat y métricas agregadas)
    all_distances = in_distances + out_distances
    all_teus = in_teus + out_teus
    total_teus = int(sum(all_teus))
    total_nm = sum(all_distances)
    total_ktm = incoming_ktm + outgoing_ktm
    mean_dist = sum(t * d for t, d in zip(all_teus, all_distances)) / total_teus if total_teus > 0 else 0

    # Métricas de tiempo
    total_hours = sum(hours_list) if hours_list else 0.0
    avg_hours = total_hours / len(hours_list) if hours_list else 0.0
    total_hour_teu = sum(h * t for h, t in zip(hours_list, hours_teus)) if hours_list else 0.0

    if len(all_distances) > 0:
        combined_hist, _, _ = np.histogram2d(all_distances, all_teus, bins=[dist_bins, teu_bins])
        combined_hist = combined_hist.T
    else:
        combined_hist = np.zeros((len(teu_bins) - 1, len(dist_bins) - 1))

    # Contar escalas únicas (cada fila es una escala, no duplicar por dirección)
    # En bidireccional, n_calls es el total de filas / 2 (cada escala genera 2 filas)
    n_calls = len(rows) // 2

    return Snapshot(
        label=label,
        port=port,
        leg_type='bidirectional',
        histogram=combined_hist,
        n_calls=n_calls,
        dist_bins=dist_bins,
        teu_bins=teu_bins,
        total_teus=total_teus,
        total_nm=total_nm,
        total_ktm=total_ktm,
        mean_dist=mean_dist,
        avg_hours=avg_hours,
        total_hours=total_hours,
        total_hour_teu=total_hour_teu,
        # Campos bidireccionales
        incoming_histogram=in_hist,
        outgoing_histogram=out_hist,
        incoming_calls=len(in_distances),
        outgoing_calls=len(out_distances),
        incoming_teus=incoming_teus,
        outgoing_teus=outgoing_teus,
        incoming_ktm=incoming_ktm,
        outgoing_ktm=outgoing_ktm,
        directional_ratio=directional_ratio
    )


# =============================================================================
# PRIMITIVA 2: compare()
# =============================================================================

def compare(snap1: Snapshot, snap2: Snapshot) -> Metrics:
    """
    Compara dos snapshots y calcula métricas.

    Args:
        snap1: Snapshot período 1 (referencia)
        snap2: Snapshot período 2 (comparar)

    Returns:
        Metrics con JSD, Hellinger, EMD, diff matrix
        (y métricas bidireccionales si los snapshots son bidireccionales)
    """
    h1 = snap1.histogram
    h2 = snap2.histogram

    # Normalizar
    eps = 1e-10
    h1_norm = (h1 + eps) / (h1.sum() + eps * h1.size)
    h2_norm = (h2 + eps) / (h2.sum() + eps * h2.size)

    # Jensen-Shannon Divergence
    jsd = jensenshannon(h1_norm.flatten(), h2_norm.flatten(), base=2) ** 2

    # Hellinger Distance
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(h1_norm) - np.sqrt(h2_norm)) ** 2))

    # Earth Mover's Distance (via marginales)
    h1_dist = h1_norm.sum(axis=0)
    h2_dist = h2_norm.sum(axis=0)
    h1_teu = h1_norm.sum(axis=1)
    h2_teu = h2_norm.sum(axis=1)

    dist_centers = (snap1.dist_bins[:-1] + snap1.dist_bins[1:]) / 2
    teu_centers = (snap1.teu_bins[:-1] + snap1.teu_bins[1:]) / 2

    emd_dist = wasserstein_distance(dist_centers, dist_centers, h1_dist, h2_dist)
    emd_teu = wasserstein_distance(teu_centers, teu_centers, h1_teu, h2_teu)
    emd = (emd_dist + emd_teu) / 2

    # Diferencia en porcentaje
    h1_pct = h1 / h1.sum() * 100 if h1.sum() > 0 else h1
    h2_pct = h2 / h2.sum() * 100 if h2.sum() > 0 else h2
    diff = h2_pct - h1_pct

    # Métricas bidireccionales (si aplica)
    jsd_incoming = None
    jsd_outgoing = None
    diff_incoming = None
    diff_outgoing = None
    directional_shift = None

    if snap1.leg_type == 'bidirectional' and snap2.leg_type == 'bidirectional':
        # Comparar incoming
        in1 = snap1.incoming_histogram
        in2 = snap2.incoming_histogram
        if in1 is not None and in2 is not None:
            in1_norm = (in1 + eps) / (in1.sum() + eps * in1.size)
            in2_norm = (in2 + eps) / (in2.sum() + eps * in2.size)
            jsd_incoming = jensenshannon(in1_norm.flatten(), in2_norm.flatten(), base=2) ** 2
            in1_pct = in1 / in1.sum() * 100 if in1.sum() > 0 else in1
            in2_pct = in2 / in2.sum() * 100 if in2.sum() > 0 else in2
            diff_incoming = in2_pct - in1_pct

        # Comparar outgoing
        out1 = snap1.outgoing_histogram
        out2 = snap2.outgoing_histogram
        if out1 is not None and out2 is not None:
            out1_norm = (out1 + eps) / (out1.sum() + eps * out1.size)
            out2_norm = (out2 + eps) / (out2.sum() + eps * out2.size)
            jsd_outgoing = jensenshannon(out1_norm.flatten(), out2_norm.flatten(), base=2) ** 2
            out1_pct = out1 / out1.sum() * 100 if out1.sum() > 0 else out1
            out2_pct = out2 / out2.sum() * 100 if out2.sum() > 0 else out2
            diff_outgoing = out2_pct - out1_pct

        # Cambio en ratio direccional
        directional_shift = snap2.directional_ratio - snap1.directional_ratio

    return Metrics(
        jsd=jsd,
        hellinger=hellinger,
        emd=emd,
        diff=diff,
        delta_calls=snap2.n_calls - snap1.n_calls,
        jsd_incoming=jsd_incoming,
        jsd_outgoing=jsd_outgoing,
        diff_incoming=diff_incoming,
        diff_outgoing=diff_outgoing,
        directional_shift=directional_shift
    )


# =============================================================================
# PRIMITIVA 3: render()
# =============================================================================

def generate_analysis(snap1: Snapshot, snap2: Snapshot, metrics: Metrics) -> str:
    """Genera texto de análisis de los resultados."""
    lines = []

    # Resumen
    lines.append(f"ANÁLISIS: {snap1.port} | {snap1.label} → {snap2.label} | {snap1.leg_type}")
    lines.append("=" * 70)

    # Métricas
    lines.append(f"JSD: {metrics.jsd:.3f} ({metrics.status})")
    lines.append(f"Hellinger: {metrics.hellinger:.3f} | EMD: {metrics.emd:.0f} nm")
    lines.append(f"Escalas: {snap1.n_calls} → {snap2.n_calls} (Δ{metrics.delta_calls:+d}, {metrics.delta_calls/snap1.n_calls*100:+.1f}%)")
    lines.append("")

    # Interpretación automática básica
    if metrics.jsd < 0.05:
        lines.append("→ Patrón ESTABLE. No se detectan cambios significativos.")
    elif metrics.jsd < 0.15:
        lines.append("→ CAMBIO MENOR detectado. Revisar diferencias en el heatmap.")
    else:
        lines.append("→ CAMBIO SIGNIFICATIVO. Analizar causas posibles:")
        lines.append("  • Cambio de rutas/servicios")
        lines.append("  • Shift en tamaño de buques")
        lines.append("  • Cambio de operadores")

    return "\n".join(lines)


def render(snap1: Snapshot, snap2: Snapshot, metrics: Metrics,
           filename: str = None, dashboard: bool = False, output_dir: Path = None) -> str:
    """
    Genera PNG con heatmaps + análisis de texto.

    Para snapshots unidireccionales: 3 paneles (T1, T2, Δ)
    Para snapshots bidireccionales: 6 paneles (2 filas × 3 cols)

    Args:
        snap1: Snapshot período 1
        snap2: Snapshot período 2
        metrics: Métricas de comparación
        filename: Nombre archivo (default: basado en parámetros)
        dashboard: Si True, guarda como 'dashboard.png' (para preview en VSCode)
        output_dir: Directorio de salida (default: ./analysis/default)

    Returns:
        Path del archivo guardado
    """
    # Use provided output_dir or default
    out_dir = output_dir if output_dir else OUTPUT_DIR

    # Delegar a render_bidirectional si los snapshots son bidireccionales
    if snap1.leg_type == 'bidirectional' and snap2.leg_type == 'bidirectional':
        return render_bidirectional(snap1, snap2, metrics, filename, dashboard, out_dir)

    # Layout unidireccional: 3 heatmaps arriba, tablas abajo, narrativa bajo título
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.2], hspace=0.35, top=0.80, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_table = fig.add_subplot(gs[1, :])

    h1 = snap1.histogram
    h2 = snap2.histogram
    h1_pct = h1 / h1.sum() * 100 if h1.sum() > 0 else h1
    h2_pct = h2 / h2.sum() * 100 if h2.sum() > 0 else h2

    vmax = max(h1_pct.max(), h2_pct.max())
    extent = [snap1.dist_bins[0], snap1.dist_bins[-1],
              snap1.teu_bins[0], snap1.teu_bins[-1]]

    # Panel 1: Período inicial
    # PowerNorm con gamma=0.5 para hacer visibles los valores bajos
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    h1_masked = np.ma.masked_where(h1_pct == 0, h1_pct)
    axes[0].set_facecolor('white')
    im1 = axes[0].imshow(h1_masked, origin='lower', aspect='auto',
                         extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes[0].set_title(f'{snap1.label}\n(n={snap1.n_calls})', fontweight='bold', fontsize=11)
    axes[0].set_xlabel('Distancia (nm)')
    axes[0].set_ylabel('TEU')
    plt.colorbar(im1, ax=axes[0], label='%', shrink=0.8)

    # Panel 2: Período final
    h2_masked = np.ma.masked_where(h2_pct == 0, h2_pct)
    axes[1].set_facecolor('white')
    im2 = axes[1].imshow(h2_masked, origin='lower', aspect='auto',
                         extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes[1].set_title(f'{snap2.label}\n(n={snap2.n_calls})', fontweight='bold', fontsize=11)
    axes[1].set_xlabel('Distancia (nm)')
    axes[1].set_ylabel('TEU')
    plt.colorbar(im2, ax=axes[1], label='%', shrink=0.8)

    # Panel 3: Diferencia
    diff_masked = np.ma.masked_where((h1_pct == 0) & (h2_pct == 0), metrics.diff)
    vmax_diff = max(abs(metrics.diff.min()), abs(metrics.diff.max()), 0.1)

    axes[2].set_facecolor('#f5f5f5')
    im3 = axes[2].imshow(diff_masked, origin='lower', aspect='auto',
                         extent=extent, cmap='RdBu_r',
                         vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title(f'Diferencia\n(Δn={metrics.delta_calls:+d})', fontweight='bold', fontsize=11)
    axes[2].set_xlabel('Distancia (nm)')
    axes[2].set_ylabel('TEU')
    plt.colorbar(im3, ax=axes[2], label='Δ%', shrink=0.8)

    # Panel de tabla: Métricas comparativas
    ax_table.axis('off')

    # Formatear números
    def fmt_num(n, decimals=1):
        if abs(n) >= 1e9:
            return f'{n/1e9:.{decimals}f} B'
        elif abs(n) >= 1e6:
            return f'{n/1e6:.{decimals}f} M'
        elif abs(n) >= 1e3:
            return f'{n/1e3:.{decimals}f} K'
        else:
            return f'{n:.0f}'

    def fmt_delta(v1, v2, decimals=1):
        delta = v2 - v1
        sign = '+' if delta >= 0 else ''
        return f'{sign}{fmt_num(delta, decimals)}'

    def fmt_pct(v1, v2):
        if v1 == 0:
            return '-'
        pct = (v2 - v1) / v1 * 100
        sign = '+' if pct >= 0 else ''
        return f'{sign}{pct:.1f}%'

    # Datos de la tabla (5 columnas)
    table_data = [
        ['Escalas', f'{snap1.n_calls:,}', f'{snap2.n_calls:,}', fmt_delta(snap1.n_calls, snap2.n_calls), fmt_pct(snap1.n_calls, snap2.n_calls)],
        ['Millas', fmt_num(snap1.total_nm, 0), fmt_num(snap2.total_nm, 0), fmt_delta(snap1.total_nm, snap2.total_nm, 0), fmt_pct(snap1.total_nm, snap2.total_nm)],
        ['TEUs', fmt_num(snap1.total_teus), fmt_num(snap2.total_teus), fmt_delta(snap1.total_teus, snap2.total_teus), fmt_pct(snap1.total_teus, snap2.total_teus)],
        ['KTM', fmt_num(snap1.total_ktm), fmt_num(snap2.total_ktm), fmt_delta(snap1.total_ktm, snap2.total_ktm), fmt_pct(snap1.total_ktm, snap2.total_ktm)],
    ]

    col_labels = ['', snap1.label, snap2.label, 'Δ', 'Δ%']

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='left',
        cellLoc='right',
        colWidths=[0.06, 0.06, 0.06, 0.06, 0.05],
        bbox=[0, 0, 0.29, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Estilo de cabecera
    for j in range(5):
        table[(0, j)].set_facecolor('#e6e6e6')
        table[(0, j)].set_text_props(fontweight='bold')

    # Primera columna alineada a la izquierda
    for i in range(5):
        table[(i, 0)].set_text_props(ha='left')

    # Tabla de métricas de comparación (a la derecha)
    dist_delta = snap2.mean_dist - snap1.mean_dist
    dist_direction = 'más largas' if dist_delta > 0 else 'más cortas'

    def interpret_jsd(v):
        if v < 0.05:
            return 'Estable'
        elif v < 0.15:
            return 'Cambio menor'
        elif v < 0.30:
            return 'Cambio moderado'
        else:
            return 'Cambio significativo'

    def interpret_hellinger(v):
        if v < 0.15:
            return 'Casi idénticos'
        elif v < 0.30:
            return 'Diferencias leves'
        elif v < 0.50:
            return 'Diferencias moderadas'
        else:
            return 'Muy distintos'

    def interpret_emd(v, dist_delta):
        if v < 200:
            return 'Mismas rutas'
        elif v < 500:
            return f'Ajuste regional ({dist_delta:+.0f} nm)'
        elif v < 1000:
            return f'Cambio de perfil ({dist_delta:+.0f} nm)'
        else:
            return f'Transformación ({dist_delta:+.0f} nm)'

    # Añadir espacios para padding visual
    pad = '  '
    metrics_data = [
        [f'{pad}JSD', f'{pad}{metrics.jsd:.3f}', f'{pad}{interpret_jsd(metrics.jsd)}{pad}'],
        [f'{pad}Hellinger', f'{pad}{metrics.hellinger:.3f}', f'{pad}{interpret_hellinger(metrics.hellinger)}{pad}'],
        [f'{pad}EMD', f'{pad}{metrics.emd:.0f} nm', f'{pad}{interpret_emd(metrics.emd, dist_delta)}{pad}'],
    ]

    metrics_labels = [f'{pad}Métrica', f'{pad}Valor', f'{pad}Interpretación{pad}']

    table2 = ax_table.table(
        cellText=metrics_data,
        colLabels=metrics_labels,
        loc='left',
        cellLoc='left',
        colWidths=[0.08, 0.08, 0.20],
        bbox=[0.36, 0, 0.36, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.4)

    # Estilo cabecera tabla 2
    for j in range(3):
        table2[(0, j)].set_facecolor('#e6e6e6')
        table2[(0, j)].set_text_props(fontweight='bold')

    # Generar narrativa de negocio
    def generate_narrative(jsd, emd, delta_pct, dist_delta, port):
        # Clasificar métricas
        jsd_level = 'alto' if jsd > 0.30 else ('moderado' if jsd > 0.15 else 'bajo')
        emd_level = 'alto' if emd > 500 else ('moderado' if emd > 200 else 'bajo')
        delta_level = 'caída' if delta_pct < -5 else ('subida' if delta_pct > 5 else 'estable')
        dist_direction = 'alargando' if dist_delta > 0 else 'acortando'
        dist_trend = 'más largas' if dist_delta > 0 else 'más cortas'

        # Matriz de narrativas
        if jsd_level == 'alto' and emd_level == 'alto' and delta_level == 'caída':
            diagnosis = "PÉRDIDA DE ROL"
            narrative = f"{port} muestra señales de pérdida de conectividad. El patrón de tráfico ha cambiado significativamente con rutas {dist_trend} ({dist_delta:+.0f} nm) y menos escalas."
            action = "Investigar: ¿Qué líneas/operadores han reducido o eliminado servicios? ¿Hay un hub competidor ganando cuota?"
        elif jsd_level == 'alto' and emd_level == 'alto' and delta_level == 'subida':
            diagnosis = "TRANSFORMACIÓN POSITIVA"
            narrative = f"{port} está ganando nuevos servicios. El patrón de conectividad se está expandiendo con rutas {dist_trend} ({dist_delta:+.0f} nm)."
            action = "Analizar: ¿Qué nuevos servicios? ¿Inversiones en infraestructura? ¿Cambio de alianzas?"
        elif jsd_level == 'alto' and emd_level in ['bajo', 'moderado']:
            diagnosis = "CAMBIO DE MIX"
            narrative = f"{port} mantiene sus rutas principales pero el mix de operadores o tamaños de buque ha cambiado sustancialmente."
            action = "Revisar: ¿Fusiones de navieras? ¿Cambio en alianzas? ¿Nuevos entrantes o salidas?"
        elif jsd_level == 'moderado' and emd_level in ['alto', 'moderado']:
            diagnosis = "RECONFIGURACIÓN"
            narrative = f"{port} muestra un shift en su patrón de conectividad. Las rutas se están {dist_direction} ({dist_delta:+.0f} nm en promedio)."
            action = "Evaluar: ¿Cambio en el balance regional vs deep-sea?"
        elif jsd_level == 'moderado':
            diagnosis = "AJUSTE OPERATIVO"
            narrative = f"{port} presenta cambios moderados en su patrón de tráfico, posiblemente estacionales o por ajustes de servicio."
            action = "Monitorizar: Comparar con períodos anteriores para confirmar si es tendencia o variación puntual."
        else:
            diagnosis = "ESTABILIDAD"
            narrative = f"{port} mantiene un patrón de tráfico estable. No se detectan cambios estructurales significativos."
            action = "Continuar monitorización rutinaria."

        return diagnosis, narrative, action

    delta_pct = (snap2.n_calls - snap1.n_calls) / snap1.n_calls * 100 if snap1.n_calls > 0 else 0
    dist_delta = snap2.mean_dist - snap1.mean_dist
    diagnosis, narrative, action = generate_narrative(metrics.jsd, metrics.emd, delta_pct, dist_delta, snap1.port)

    # Título principal
    title = f'{snap1.port} | {snap1.leg_type}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)

    # Panel de narrativa (debajo del título)
    narrative_text = f"{diagnosis}: {narrative}"
    fig.text(0.5, 0.91, narrative_text, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', edgecolor='#ffc107', linewidth=2),
             wrap=True)

    plt.tight_layout()

    # Determinar nombre de archivo
    if dashboard:
        filename = "dashboard.png"
    elif filename is None:
        filename = f"{snap1.port}_{snap1.label}_vs_{snap2.label}_{snap1.leg_type}.png".replace(" ", "_")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()

    return str(output_path)


def render_bidirectional(snap1: Snapshot, snap2: Snapshot, metrics: Metrics,
                         filename: str = None, dashboard: bool = False,
                         output_dir: Path = None) -> str:
    """
    Genera PNG con 6 paneles para análisis bidireccional.

    Layout:
        Fila 1: T1-INCOMING | T2-INCOMING | Δ INCOMING
        Fila 2: T1-OUTGOING | T2-OUTGOING | Δ OUTGOING
        Fila 3: Tabla de métricas
    """
    from matplotlib.colors import PowerNorm

    # Use provided output_dir or default
    out_dir = output_dir if output_dir else OUTPUT_DIR

    # Layout: 6 heatmaps (2 filas × 3 cols) + tabla
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 3, 1.2], hspace=0.30, top=0.88, bottom=0.08)

    # Crear axes para heatmaps
    axes_in = [fig.add_subplot(gs[0, i]) for i in range(3)]   # Fila incoming
    axes_out = [fig.add_subplot(gs[1, i]) for i in range(3)]  # Fila outgoing
    ax_table = fig.add_subplot(gs[2, :])

    extent = [snap1.dist_bins[0], snap1.dist_bins[-1],
              snap1.teu_bins[0], snap1.teu_bins[-1]]

    # Histogramas incoming
    in1 = snap1.incoming_histogram
    in2 = snap2.incoming_histogram
    in1_pct = in1 / in1.sum() * 100 if in1.sum() > 0 else in1
    in2_pct = in2 / in2.sum() * 100 if in2.sum() > 0 else in2

    # Histogramas outgoing
    out1 = snap1.outgoing_histogram
    out2 = snap2.outgoing_histogram
    out1_pct = out1 / out1.sum() * 100 if out1.sum() > 0 else out1
    out2_pct = out2 / out2.sum() * 100 if out2.sum() > 0 else out2

    # Escalas comunes
    vmax_in = max(in1_pct.max(), in2_pct.max())
    vmax_out = max(out1_pct.max(), out2_pct.max())
    vmax = max(vmax_in, vmax_out)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    # Fila 1: INCOMING
    in1_masked = np.ma.masked_where(in1_pct == 0, in1_pct)
    axes_in[0].set_facecolor('white')
    im = axes_in[0].imshow(in1_masked, origin='lower', aspect='auto',
                           extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes_in[0].set_title(f'INCOMING {snap1.label}\n(n={snap1.incoming_calls})', fontweight='bold', fontsize=10)
    axes_in[0].set_ylabel('TEU')
    plt.colorbar(im, ax=axes_in[0], label='%', shrink=0.8)

    in2_masked = np.ma.masked_where(in2_pct == 0, in2_pct)
    axes_in[1].set_facecolor('white')
    im = axes_in[1].imshow(in2_masked, origin='lower', aspect='auto',
                           extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes_in[1].set_title(f'INCOMING {snap2.label}\n(n={snap2.incoming_calls})', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=axes_in[1], label='%', shrink=0.8)

    # Diferencia incoming
    diff_in = metrics.diff_incoming if metrics.diff_incoming is not None else (in2_pct - in1_pct)
    diff_in_masked = np.ma.masked_where((in1_pct == 0) & (in2_pct == 0), diff_in)
    vmax_diff_in = max(abs(diff_in.min()), abs(diff_in.max()), 0.1)
    axes_in[2].set_facecolor('#f5f5f5')
    im = axes_in[2].imshow(diff_in_masked, origin='lower', aspect='auto',
                           extent=extent, cmap='RdBu_r', vmin=-vmax_diff_in, vmax=vmax_diff_in)
    jsd_in = metrics.jsd_incoming if metrics.jsd_incoming else 0
    axes_in[2].set_title(f'Δ INCOMING\n(JSD={jsd_in:.3f})', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=axes_in[2], label='Δ%', shrink=0.8)

    # Fila 2: OUTGOING
    out1_masked = np.ma.masked_where(out1_pct == 0, out1_pct)
    axes_out[0].set_facecolor('white')
    im = axes_out[0].imshow(out1_masked, origin='lower', aspect='auto',
                            extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes_out[0].set_title(f'OUTGOING {snap1.label}\n(n={snap1.outgoing_calls})', fontweight='bold', fontsize=10)
    axes_out[0].set_xlabel('Distancia (nm)')
    axes_out[0].set_ylabel('TEU')
    plt.colorbar(im, ax=axes_out[0], label='%', shrink=0.8)

    out2_masked = np.ma.masked_where(out2_pct == 0, out2_pct)
    axes_out[1].set_facecolor('white')
    im = axes_out[1].imshow(out2_masked, origin='lower', aspect='auto',
                            extent=extent, cmap=CMAP_SNAPSHOT, norm=norm)
    axes_out[1].set_title(f'OUTGOING {snap2.label}\n(n={snap2.outgoing_calls})', fontweight='bold', fontsize=10)
    axes_out[1].set_xlabel('Distancia (nm)')
    plt.colorbar(im, ax=axes_out[1], label='%', shrink=0.8)

    # Diferencia outgoing
    diff_out = metrics.diff_outgoing if metrics.diff_outgoing is not None else (out2_pct - out1_pct)
    diff_out_masked = np.ma.masked_where((out1_pct == 0) & (out2_pct == 0), diff_out)
    vmax_diff_out = max(abs(diff_out.min()), abs(diff_out.max()), 0.1)
    axes_out[2].set_facecolor('#f5f5f5')
    im = axes_out[2].imshow(diff_out_masked, origin='lower', aspect='auto',
                            extent=extent, cmap='RdBu_r', vmin=-vmax_diff_out, vmax=vmax_diff_out)
    jsd_out = metrics.jsd_outgoing if metrics.jsd_outgoing else 0
    axes_out[2].set_title(f'Δ OUTGOING\n(JSD={jsd_out:.3f})', fontweight='bold', fontsize=10)
    axes_out[2].set_xlabel('Distancia (nm)')
    plt.colorbar(im, ax=axes_out[2], label='Δ%', shrink=0.8)

    # Panel de tabla: Métricas bidireccionales
    ax_table.axis('off')

    def fmt_num(n, decimals=1):
        if abs(n) >= 1e9:
            return f'{n/1e9:.{decimals}f} B'
        elif abs(n) >= 1e6:
            return f'{n/1e6:.{decimals}f} M'
        elif abs(n) >= 1e3:
            return f'{n/1e3:.{decimals}f} K'
        else:
            return f'{n:.0f}'

    def fmt_delta(v1, v2, decimals=1):
        delta = v2 - v1
        sign = '+' if delta >= 0 else ''
        return f'{sign}{fmt_num(delta, decimals)}'

    def fmt_pct(v1, v2):
        if v1 == 0:
            return '-'
        pct = (v2 - v1) / v1 * 100
        sign = '+' if pct >= 0 else ''
        return f'{sign}{pct:.1f}%'

    # Tabla direccional (incoming vs outgoing)
    table_data = [
        ['INCOMING', f'{snap1.incoming_calls:,}', f'{snap2.incoming_calls:,}',
         fmt_delta(snap1.incoming_calls, snap2.incoming_calls),
         fmt_pct(snap1.incoming_calls, snap2.incoming_calls),
         fmt_num(snap1.incoming_ktm), fmt_num(snap2.incoming_ktm)],
        ['OUTGOING', f'{snap1.outgoing_calls:,}', f'{snap2.outgoing_calls:,}',
         fmt_delta(snap1.outgoing_calls, snap2.outgoing_calls),
         fmt_pct(snap1.outgoing_calls, snap2.outgoing_calls),
         fmt_num(snap1.outgoing_ktm), fmt_num(snap2.outgoing_ktm)],
        ['TOTAL', f'{snap1.n_calls:,}', f'{snap2.n_calls:,}',
         fmt_delta(snap1.n_calls, snap2.n_calls),
         fmt_pct(snap1.n_calls, snap2.n_calls),
         fmt_num(snap1.total_ktm), fmt_num(snap2.total_ktm)],
    ]

    col_labels = ['Dirección', f'Escalas {snap1.label}', f'Escalas {snap2.label}', 'Δ', 'Δ%',
                  f'KTM {snap1.label}', f'KTM {snap2.label}']

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='left',
        cellLoc='right',
        colWidths=[0.08, 0.08, 0.08, 0.06, 0.06, 0.08, 0.08],
        bbox=[0, 0, 0.52, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for j in range(7):
        table[(0, j)].set_facecolor('#e6e6e6')
        table[(0, j)].set_text_props(fontweight='bold')
    table[(0, 0)].set_text_props(ha='left')

    # Tabla de métricas JSD
    jsd_in_val = metrics.jsd_incoming if metrics.jsd_incoming else 0
    jsd_out_val = metrics.jsd_outgoing if metrics.jsd_outgoing else 0
    dir_shift = metrics.directional_shift if metrics.directional_shift else 0

    metrics_data = [
        ['JSD Incoming', f'{jsd_in_val:.3f}', 'Cambio en orígenes'],
        ['JSD Outgoing', f'{jsd_out_val:.3f}', 'Cambio en destinos'],
        ['JSD Agregado', f'{metrics.jsd:.3f}', 'Cambio total'],
        ['Ratio O/I T1', f'{snap1.directional_ratio:.2f}', 'Salidas/Entradas'],
        ['Ratio O/I T2', f'{snap2.directional_ratio:.2f}', f'Δ={dir_shift:+.2f}'],
    ]

    metrics_labels = ['Métrica', 'Valor', 'Descripción']

    table2 = ax_table.table(
        cellText=metrics_data,
        colLabels=metrics_labels,
        loc='left',
        cellLoc='left',
        colWidths=[0.10, 0.06, 0.14],
        bbox=[0.56, 0, 0.30, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.2)

    for j in range(3):
        table2[(0, j)].set_facecolor('#e6e6e6')
        table2[(0, j)].set_text_props(fontweight='bold')

    # Título y narrativa
    title = f'{snap1.port} | BIDIRECCIONAL'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)

    # Diagnóstico basado en ratio direccional
    if abs(dir_shift) > 0.2:
        if dir_shift > 0:
            diagnosis = "SHIFT A EXPORTADOR"
            narrative = f"{snap1.port} está aumentando su rol como origen (más salidas relativas a entradas)"
        else:
            diagnosis = "SHIFT A IMPORTADOR"
            narrative = f"{snap1.port} está aumentando su rol como destino (más entradas relativas a salidas)"
    elif max(jsd_in_val, jsd_out_val) > 0.30:
        if jsd_in_val > jsd_out_val:
            diagnosis = "CAMBIO EN ORÍGENES"
            narrative = f"{snap1.port} muestra cambio significativo en los puertos de origen de sus buques"
        else:
            diagnosis = "CAMBIO EN DESTINOS"
            narrative = f"{snap1.port} muestra cambio significativo en los puertos de destino de sus buques"
    elif metrics.jsd > 0.15:
        diagnosis = "RECONFIGURACIÓN"
        narrative = f"{snap1.port} muestra cambios moderados en sus patrones de conectividad"
    else:
        diagnosis = "ESTABILIDAD"
        narrative = f"{snap1.port} mantiene patrones de conectividad estables en ambas direcciones"

    fig.text(0.5, 0.91, f"{diagnosis}: {narrative}", ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2),
             wrap=True)

    plt.tight_layout()

    # Determinar nombre de archivo
    if dashboard:
        filename = "dashboard.png"
    elif filename is None:
        filename = f"{snap1.port}_{snap1.label}_vs_{snap2.label}_bidirectional.png".replace(" ", "_")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()

    return str(output_path)


# =============================================================================
# Parsing de períodos
# =============================================================================

def parse_period(year: int, period: str = None) -> tuple:
    """
    Convierte año + período en rango de fechas.

    Args:
        year: Año (ej: 2023)
        period: M1-M12 (mes), Q1-Q4 (trimestre), None (año completo)

    Returns:
        (start_date, end_date, label) como strings

    Ejemplos:
        parse_period(2023, "Q3") → ("2023-07-01", "2023-10-01", "2023 Q3")
        parse_period(2023, "M7") → ("2023-07-01", "2023-08-01", "2023 M7")
        parse_period(2023)       → ("2023-01-01", "2024-01-01", "2023")
    """
    if period is None or period == "":
        # Año completo (default)
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        label = str(year)
        return start, end, label

    period = period.upper()

    if period.startswith("Q"):
        # Trimestre Q1-Q4
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
        # Mes M1-M12
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
        raise ValueError(f"Período inválido: {period}. Usar M1-M12 o Q1-Q4 (sin período = año completo)")

    return start, end, label


# =============================================================================
# Helpers
# =============================================================================

def build_query(port: str, year1: int, period1: str = None,
                year2: int = None, period2: str = None,
                leg_type: str = "bidirectional") -> str:
    """
    Genera query SQL para comparar dos períodos.

    Args:
        port: Nombre del puerto
        year1: Primer año
        period1: Período del primer año (Q1-Q4, M1-M12, o None=año completo)
        year2: Segundo año
        period2: Período del segundo año
        leg_type: 'prev_leg', 'next_leg', 'both', o 'bidirectional'

    Returns:
        Query SQL para db_export_query_to_file()

    Ejemplos:
        build_query("London", 2023, "Q3", 2025, "Q3")  # Q3 vs Q3
        build_query("London", 2023, None, 2025, None)  # Año vs año
        build_query("London", 2023, "Q3", 2025, "Q3", "bidirectional")  # Bidireccional
    """
    start1, end1, _ = parse_period(year1, period1)
    start2, end2, _ = parse_period(year2, period2)

    date_filter = f"((e.start >= '{start1}' AND e.start < '{end1}') OR (e.start >= '{start2}' AND e.start < '{end2}'))"
    base_filter = f"e.portname = '{port}' AND {date_filter} AND f.fleet = 'containers' AND f.teus > 0"

    if leg_type == 'bidirectional':
        # UNION ALL para obtener ambas direcciones con columna 'direction'
        return f"""SELECT e.start, e.prev_leg as leg, 'incoming' as direction, f.teus as TEU FROM v_escalas e JOIN v_fleet f ON e.imo = f.imo WHERE {base_filter} UNION ALL SELECT e.start, e.next_leg as leg, 'outgoing' as direction, f.teus as TEU FROM v_escalas e JOIN v_fleet f ON e.imo = f.imo WHERE {base_filter} ORDER BY start"""
    else:
        # Query unidireccional original
        return f"""SELECT e.start, e.prev_leg, e.next_leg, f.teus as TEU FROM v_escalas e JOIN v_fleet f ON e.imo = f.imo WHERE {base_filter} ORDER BY e.start"""


def compare_periods(csv_path: str, port: str,
                    year1: int, period1: str = None,
                    year2: int = None, period2: str = None,
                    leg_type: str = "bidirectional",
                    output: str = None,
                    dashboard: bool = False) -> dict:
    """
    Compara dos períodos desde CSV exportado.

    Args:
        csv_path: Ruta al CSV exportado por MCP
        port: Nombre del puerto
        year1: Primer año
        period1: Período (Q1-Q4, M1-M12, o None=año completo)
        year2: Segundo año
        period2: Período
        leg_type: 'prev_leg', 'next_leg', 'both', o 'bidirectional'
        output: Nombre del archivo PNG (default: auto)

    Returns:
        dict con métricas y path del PNG
        (incluye métricas bidireccionales si leg_type='bidirectional')

    Ejemplos:
        compare_periods("data.csv", "London", 2023, "Q3", 2025, "Q3")  # Q3 vs Q3
        compare_periods("data.csv", "London", 2023, None, 2025, None)  # Año vs año
        compare_periods("data.csv", "London", 2023, "Q3", 2025, "Q3", "bidirectional")  # Bidireccional
    """
    import pandas as pd

    # Parsear períodos
    start1, end1, label1 = parse_period(year1, period1)
    start2, end2, label2 = parse_period(year2, period2)

    # Cargar y filtrar datos
    df = pd.read_csv(csv_path)
    df['start'] = pd.to_datetime(df['start'])

    df1 = df[(df['start'] >= start1) & (df['start'] < end1)]
    df2 = df[(df['start'] >= start2) & (df['start'] < end2)]

    # Crear snapshots
    s1 = snapshot(df1.to_dict('records'), port, label1, leg_type)
    s2 = snapshot(df2.to_dict('records'), port, label2, leg_type)

    # Comparar y renderizar
    metrics = compare(s1, s2)

    if output is None:
        output = f"{port}_{label1}_vs_{label2}_{leg_type}.png".replace(" ", "_")

    path = render(s1, s2, metrics, output, dashboard)

    # Resultado base
    result = {
        'jsd': metrics.jsd,
        'hellinger': metrics.hellinger,
        'emd': metrics.emd,
        'status': metrics.status,
        'n1': s1.n_calls,
        'n2': s2.n_calls,
        'delta': metrics.delta_calls,
        'output': path
    }

    # Añadir métricas bidireccionales si aplica
    if leg_type == 'bidirectional':
        result.update({
            'jsd_incoming': metrics.jsd_incoming,
            'jsd_outgoing': metrics.jsd_outgoing,
            'directional_shift': metrics.directional_shift,
            'incoming_calls_1': s1.incoming_calls,
            'incoming_calls_2': s2.incoming_calls,
            'outgoing_calls_1': s1.outgoing_calls,
            'outgoing_calls_2': s2.outgoing_calls,
            'ratio_1': s1.directional_ratio,
            'ratio_2': s2.directional_ratio
        })

    return result


# =============================================================================
# HOTSPOT IDENTIFICATION (TEU-based segmentation)
# =============================================================================

def identify_hotspots(snap1: Snapshot, snap2: Snapshot, metrics: Metrics,
                      n_per_segment: int = 3) -> dict:
    """
    Identifica los 12 hotspots: 3 pequeños + 3 grandes × 2 direcciones.

    Args:
        snap1, snap2: Snapshots de los períodos
        metrics: Métricas de comparación
        n_per_segment: Número de bins por segmento (default 3)

    Returns:
        Dict con estructura {incoming: {small: [...], large: [...]}, outgoing: {...}}
    """
    result = {
        'incoming': {'small': [], 'large': []},
        'outgoing': {'small': [], 'large': []}
    }

    for direction in ['incoming', 'outgoing']:
        # Seleccionar diff según dirección
        if direction == 'incoming':
            diff = metrics.diff_incoming if metrics.diff_incoming is not None else metrics.diff
        else:
            diff = metrics.diff_outgoing if metrics.diff_outgoing is not None else metrics.diff

        for segment, (teu_min, teu_max) in [('small', (0, TEU_SMALL_MAX)),
                                             ('large', (TEU_LARGE_MIN, TEU_MAX))]:
            # Crear máscara para el segmento TEU
            mask = np.zeros_like(diff, dtype=bool)
            for i in range(diff.shape[0]):
                bin_teu_min = snap1.teu_bins[i]
                bin_teu_max = snap1.teu_bins[i + 1]
                if bin_teu_min >= teu_min and bin_teu_max <= teu_max:
                    mask[i, :] = True

            # Aplicar máscara y encontrar top n
            diff_masked = np.where(mask, diff, 0)
            flat_indices = np.argsort(np.abs(diff_masked.flatten()))[-n_per_segment:][::-1]

            hotspots = []
            for rank, flat_idx in enumerate(flat_indices, 1):
                i, j = np.unravel_index(flat_idx, diff.shape)
                delta = diff[i, j]
                if delta == 0:
                    continue

                dist_min, dist_max = float(snap1.dist_bins[j]), float(snap1.dist_bins[j + 1])
                teu_min, teu_max = float(snap1.teu_bins[i]), float(snap1.teu_bins[i + 1])

                hotspots.append(Hotspot(
                    rank=rank,
                    direction=direction,
                    segment=segment,
                    dist_range=(dist_min, dist_max),
                    teu_range=(teu_min, teu_max),
                    delta_pct=float(delta),
                    change='GANANCIA' if delta > 0 else 'PÉRDIDA',
                    dist_label=get_bin_label('dist', dist_min, dist_max),
                    teu_label=get_bin_label('teu', teu_min, teu_max)
                ))

            result[direction][segment] = hotspots

    return result


# =============================================================================
# ANALYZE: Pipeline completo → JSON
# =============================================================================

def analyze(csv_path: str, port: str,
            year1: int, period1: str = None,
            year2: int = None, period2: str = None,
            output_dir: str = None) -> dict:
    """
    Pipeline completo: carga datos, compara, identifica hotspots, genera JSON.

    Args:
        csv_path: Ruta al CSV exportado
        port: Nombre del puerto
        year1, period1: Primer período
        year2, period2: Segundo período
        output_dir: Carpeta de salida (default: analysis/default or analysis/{port})

    Returns:
        Dict JSON-serializable con métricas, hotspots y rutas de archivos
    """
    import pandas as pd
    import json

    # Parsear períodos
    start1, end1, label1 = parse_period(year1, period1)
    start2, end2, label2 = parse_period(year2, period2)

    # Determinar carpeta de salida y naming
    is_default = output_dir is None
    if is_default:
        out_path = ANALYSIS_DIR / "default"
    else:
        out_path = ANALYSIS_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Cargar y filtrar datos
    df = pd.read_csv(csv_path)
    df['start'] = pd.to_datetime(df['start'])

    df1 = df[(df['start'] >= start1) & (df['start'] < end1)]
    df2 = df[(df['start'] >= start2) & (df['start'] < end2)]

    # Crear snapshots (bidirectional por defecto)
    s1 = snapshot(df1.to_dict('records'), port, label1, 'bidirectional')
    s2 = snapshot(df2.to_dict('records'), port, label2, 'bidirectional')

    # Comparar
    metrics = compare(s1, s2)

    # Identificar hotspots
    hotspots = identify_hotspots(s1, s2, metrics)

    # Generar PNG (nombres genéricos en default, específicos en carpeta de puerto)
    if is_default:
        png_name = "default.png"
    else:
        png_name = f"{port}_{label1}_vs_{label2}.png".replace(" ", "_")
    png_path = out_path / png_name
    render(s1, s2, metrics, str(png_path), output_dir=out_path)

    # Construir resultado JSON
    result = {
        'port': port,
        'period1': label1,
        'period2': label2,
        'date_range1': {'start': start1, 'end': end1},
        'date_range2': {'start': start2, 'end': end2},
        'metrics': {
            'jsd': round(metrics.jsd, 3),
            'jsd_incoming': round(metrics.jsd_incoming, 3) if metrics.jsd_incoming else None,
            'jsd_outgoing': round(metrics.jsd_outgoing, 3) if metrics.jsd_outgoing else None,
            'hellinger': round(metrics.hellinger, 3),
            'emd': round(metrics.emd, 0),
            'status': metrics.status
        },
        'summary': {
            'calls_1': s1.n_calls,
            'calls_2': s2.n_calls,
            'delta_calls': s2.n_calls - s1.n_calls,
            'teus_1': s1.total_teus,
            'teus_2': s2.total_teus,
            'ktm_1': round(s1.total_ktm, 0),
            'ktm_2': round(s2.total_ktm, 0),
            'avg_dist_1': round(s1.mean_dist, 0),
            'avg_dist_2': round(s2.mean_dist, 0),
            'avg_hours_1': round(s1.avg_hours, 1),
            'avg_hours_2': round(s2.avg_hours, 1),
            'total_hours_1': round(s1.total_hours, 0),
            'total_hours_2': round(s2.total_hours, 0),
            'hour_teu_1': round(s1.total_hour_teu, 0),
            'hour_teu_2': round(s2.total_hour_teu, 0)
        },
        'hotspots': {
            direction: {
                segment: [h.to_dict() for h in bins]
                for segment, bins in segments.items()
            }
            for direction, segments in hotspots.items()
        },
        'files': {
            'png': str(png_path)
        }
    }

    # Guardar JSON (nombres genéricos en default)
    if is_default:
        json_name = "default.json"
    else:
        json_name = f"{port}_{label1}_vs_{label2}.json".replace(" ", "_")
    json_path = out_path / json_name
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    result['files']['json'] = str(json_path)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Histogram comparison for port traffic')
    parser.add_argument('csv_path', help='Path to CSV file with port call data')
    parser.add_argument('--port', '-p', required=True, help='Port name')
    parser.add_argument('--y1', type=int, required=True, help='First year')
    parser.add_argument('--p1', default=None, help='First period (Q1-Q4, M1-M12)')
    parser.add_argument('--y2', type=int, required=True, help='Second year')
    parser.add_argument('--p2', default=None, help='Second period (Q1-Q4, M1-M12)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: analysis/default, or analysis/{port} if port given)')
    parser.add_argument('--default', action='store_true',
                        help='Use analysis/default/ regardless of port')

    args = parser.parse_args()

    # Determinar output_dir
    if args.default:
        output_dir = None  # Uses default
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.port  # Use port name as folder

    result = analyze(
        csv_path=args.csv_path,
        port=args.port,
        year1=args.y1,
        period1=args.p1,
        year2=args.y2,
        period2=args.p2,
        output_dir=output_dir
    )

    print(json.dumps(result, indent=2))

"""
Service Analysis - Análisis de servicios liner
Part of the Port Analyst toolkit

This module provides data structures and query builders for analyzing
liner services. The actual queries are executed via MCP traffic-db server.

Usage:
    The dataclasses define the expected output structure.
    Query builders generate SQL for MCP execution.

See references/service-methodology.md for full methodology.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ServiceSnapshot:
    """Service metadata from econdb tables"""
    name: str
    operators: list[str]
    alliance: str
    rotation_days: int
    active_vessels: int
    avg_teu: int
    trade_lane: str
    first_voyage: str
    last_voyage: str
    is_feeder: bool = False
    is_regular: bool = True


@dataclass
class PortFrequency:
    """Port call frequency and classification"""
    portname: str
    country: str
    zone: str
    calls: int
    frequency_days: float
    ratio: float
    classification: str  # Regular/Ocasional/Esporádico

    @classmethod
    def classify(cls, ratio: float) -> str:
        """Classify port based on ratio to max calls"""
        if ratio >= 0.5:
            return "Regular"
        elif ratio >= 0.2:
            return "Ocasional"
        else:
            return "Esporádico"


@dataclass
class ReliabilityMetrics:
    """Schedule reliability metrics at gateway port"""
    gateway_port: str
    intervals: int
    frequency_days: float
    std_days: float
    reliability_6h: float
    reliability_12h: float
    reliability_24h: float
    classification: str  # Excelente/Buena/Moderada/Baja

    @classmethod
    def classify(cls, reliability_12h: float) -> str:
        """Classify reliability based on ±12h metric"""
        if reliability_12h >= 90:
            return "Excelente"
        elif reliability_12h >= 70:
            return "Buena"
        elif reliability_12h >= 50:
            return "Moderada"
        else:
            return "Baja"


@dataclass
class VesselInfo:
    """Vessel details for fleet table"""
    name: str
    imo: int
    teus: int
    category: str
    year_built: Optional[int]
    fuel: Optional[str]
    operator: str
    calls: int


@dataclass
class ETSMetrics:
    """EU ETS cost metrics"""
    year: int
    quarter: int
    calls: int
    ets_entry: float
    ets_exit: float
    ets_total: float
    ets_avg_per_call: float


@dataclass
class ServiceAnalysis:
    """Complete service analysis result"""
    snapshot: ServiceSnapshot
    ports: list[PortFrequency]
    reliability: Optional[ReliabilityMetrics]
    euromed_type: str  # EXTRA/INTRA/NON-EUROMED
    gateway_entry: Optional[str]
    gateway_exit: Optional[str]
    ets_metrics: list[ETSMetrics] = field(default_factory=list)
    fleet: list[VesselInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# Query Builders
# =============================================================================

EUROMED_ZONES = "('NORTH EUROPE','ATLANTIC','WEST MED','EAST MED')"

def build_metadata_query(service_name: str) -> str:
    """Build query for service metadata"""
    return f"""
SELECT sls.id as service_id, sls.name as servicio, sls.duration as rotacion_dias,
       sls.active_vessels, sls.teu_avg, sls.trade_lane, sls.is_feeder, sls.is_regular,
       sls.start_date, sls.end_date
FROM econdb_shared_line_service sls
WHERE sls.name LIKE '%{service_name}%'
""".replace('\n', ' ').strip()


def build_operators_query(service_name: str) -> str:
    """Build query for operators and alliances"""
    return f"""
SELECT sls.name as servicio, ls.operator, ls.alliance, ls.code as operator_code,
       ls.first_voyage, ls.last_voyage
FROM econdb_shared_line_service sls
JOIN econdb_line_service ls ON ls.shared_line_service_id = sls.id
WHERE sls.name LIKE '%{service_name}%'
""".replace('\n', ' ').strip()


def build_ports_query(service_name: str) -> str:
    """Build query for ports with frequency"""
    return f"""
SELECT e.portname, p.country, p.zone, COUNT(*) as escalas,
       DATEDIFF(MAX(e.start), MIN(e.start)) as dias,
       ROUND(DATEDIFF(MAX(e.start), MIN(e.start)) / COUNT(*), 1) as dias_entre_escalas
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{service_name}%' AND f.fleet = 'containers'
GROUP BY e.portname, p.country, p.zone
ORDER BY escalas DESC
""".replace('\n', ' ').strip()


def build_gateway_query(service_name: str) -> str:
    """Build query for European gateway port (entry from outside EUROMED)"""
    return f"""
SELECT e.portname, COUNT(*) as entries
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN ports p_prev ON e.prev_port = p_prev.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{service_name}%'
  AND p.zone IN {EUROMED_ZONES}
  AND p_prev.zone NOT IN {EUROMED_ZONES}
GROUP BY e.portname
ORDER BY entries DESC LIMIT 1
""".replace('\n', ' ').strip()


def build_reliability_query(service_name: str, gateway_port: str) -> str:
    """Build query for reliability metrics at gateway port"""
    return f"""
WITH intervalos AS (
  SELECT e.portname, e.start,
    LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start) as escala_anterior,
    TIMESTAMPDIFF(HOUR, LAG(e.start) OVER (PARTITION BY e.portname ORDER BY e.start), e.start) as intervalo_horas
  FROM escalas e
  JOIN econdb_ship_service_ranges s ON e.imo = s.imo
    AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
  JOIN ports p_prev ON e.prev_port = p_prev.portname
  WHERE s.sls_name LIKE '%{service_name}%'
    AND e.portname = '{gateway_port}'
    AND p_prev.zone NOT IN {EUROMED_ZONES}
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
""".replace('\n', ' ').strip()


def build_euromed_type_query(service_name: str) -> str:
    """Build query to determine EUROMED type (EXTRA/INTRA/NON-EUROMED)"""
    return f"""
SELECT
  CASE
    WHEN COUNT(DISTINCT CASE WHEN p.zone IN {EUROMED_ZONES} THEN e.portname END) > 0
     AND COUNT(DISTINCT CASE WHEN p.zone NOT IN {EUROMED_ZONES} THEN e.portname END) > 0
    THEN 'EUROMED-EXTRA'
    WHEN COUNT(DISTINCT CASE WHEN p.zone IN {EUROMED_ZONES} THEN e.portname END) > 0
    THEN 'EUROMED-INTRA'
    ELSE 'NON-EUROMED'
  END as tipo_servicio,
  GROUP_CONCAT(DISTINCT p.zone ORDER BY p.zone) as zonas_tocadas
FROM escalas e
JOIN ports p ON e.portname = p.portname
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{service_name}%'
""".replace('\n', ' ').strip()


def build_ets_query(service_name: str) -> str:
    """Build query for ETS costs by quarter"""
    return f"""
SELECT YEAR(e.start) as año, QUARTER(e.start) as trimestre, COUNT(*) as escalas,
  ROUND(SUM(em.incoming_ets_cost_eur), 0) as ets_entrada,
  ROUND(SUM(em.outgoing_ets_cost_eur), 0) as ets_salida,
  ROUND(SUM(em.total_ets_cost_eur), 0) as ets_total,
  ROUND(AVG(em.total_ets_cost_eur), 0) as ets_medio_escala
FROM escalas e
JOIN v_escalas_metrics em ON e.imo = em.imo AND e.start = em.start
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{service_name}%' AND e.start >= '2024-01-01'
GROUP BY YEAR(e.start), QUARTER(e.start)
ORDER BY año, trimestre
""".replace('\n', ' ').strip()


def build_fleet_query(service_name: str) -> str:
    """Build query for deployed fleet"""
    return f"""
SELECT f.name as buque, f.imo, ROUND(f.teus, 0) as teus, f.category,
  f.Year as año_construccion, f.fuel, f.group as operador, COUNT(*) as escalas
FROM escalas e
JOIN v_fleet f ON e.imo = f.imo
JOIN econdb_ship_service_ranges s ON e.imo = s.imo
  AND e.start BETWEEN s.start_range AND COALESCE(s.end_range, '2099-12-31')
WHERE s.sls_name LIKE '%{service_name}%' AND f.fleet = 'containers'
GROUP BY f.name, f.imo, f.teus, f.category, f.Year, f.fuel, f.group
ORDER BY escalas DESC
""".replace('\n', ' ').strip()


# =============================================================================
# Markdown Renderer
# =============================================================================

def render_ficha(analysis: ServiceAnalysis) -> str:
    """Render service analysis as markdown ficha"""
    s = analysis.snapshot
    r = analysis.reliability

    lines = [
        f"## {s.name}",
        "",
        f"**Operadores:** {', '.join(s.operators)}",
        f"**Alianza:** {s.alliance}",
        f"**Rotación:** {s.rotation_days} días",
        f"**Buques activos:** {s.active_vessels}",
        f"**TEU medio:** {s.avg_teu:,}",
        f"**Trade lane:** {s.trade_lane}",
        f"**Tipo EUROMED:** {analysis.euromed_type}",
    ]

    if analysis.gateway_entry:
        lines.append(f"**Gateway entrada:** {analysis.gateway_entry}")

    if r:
        lines.extend([
            f"**Frecuencia:** 1 buque cada {r.frequency_days} días en {r.gateway_port}",
            f"**Fiabilidad:** {r.reliability_12h}% ±12h ({r.classification})",
        ])

    lines.extend([
        "",
        "### Puertos del Servicio",
        "",
        "| Puerto | País | Zona | Escalas | Frecuencia | Clasificación |",
        "|--------|------|------|--------:|:----------:|---------------|",
    ])

    for p in analysis.ports[:15]:  # Top 15 ports
        lines.append(
            f"| {p.portname} | {p.country} | {p.zone} | {p.calls} | "
            f"1 cada {p.frequency_days} días | {p.classification} |"
        )

    if analysis.fleet:
        lines.extend([
            "",
            "### Flota Desplegada",
            "",
            "| Buque | TEU | Cat | Año | Fuel | Operador | Escalas |",
            "|-------|----:|:---:|----:|------|----------|--------:|",
        ])
        for v in analysis.fleet[:10]:  # Top 10 vessels
            fuel = v.fuel or "-"
            year = v.year_built or "-"
            lines.append(
                f"| {v.name} | {v.teus:,} | {v.category} | {year} | "
                f"{fuel} | {v.operator} | {v.calls} |"
            )

    if analysis.ets_metrics:
        total_ets = sum(e.ets_total for e in analysis.ets_metrics)
        lines.extend([
            "",
            "### Costes ETS",
            "",
            f"**Total ETS (desde 2024):** €{total_ets:,.0f}",
        ])

    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Service Analysis - Generate SQL queries for liner service analysis"
    )
    parser.add_argument("service", help="Service name (e.g., 'AE3', 'Guangdong')")
    parser.add_argument("--query", "-q",
                       choices=["metadata", "operators", "ports", "gateway",
                               "reliability", "euromed", "ets", "fleet", "all"],
                       default="all",
                       help="Which query to generate")
    parser.add_argument("--gateway", "-g", help="Gateway port for reliability query")

    args = parser.parse_args()

    queries = {
        "metadata": build_metadata_query,
        "operators": build_operators_query,
        "ports": build_ports_query,
        "gateway": build_gateway_query,
        "euromed": build_euromed_type_query,
        "ets": build_ets_query,
        "fleet": build_fleet_query,
    }

    if args.query == "all":
        for name, func in queries.items():
            print(f"-- {name.upper()}")
            print(func(args.service))
            print()
    elif args.query == "reliability":
        if not args.gateway:
            print("Error: --gateway required for reliability query")
        else:
            print(build_reliability_query(args.service, args.gateway))
    else:
        print(queries[args.query](args.service))

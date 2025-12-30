#!/usr/bin/env python3
"""
cube.py - ConnectivityCube: Almacenamiento N×N×T×C para matrices de conectividad

Estructura de datos que almacena matrices de conectividad puerto-a-puerto
con dimensiones variables usando NumPy 4D arrays.

Dimensiones:
  - N (axis): Puertos/zonas (variable, 39-60+)
  - T (periods): Períodos temporales (trimestres, 4-16)
  - C (categories): Vessel categories 1-5 (Feeder, Feeder-max, Panamax, New-Panamax, ULCV)
  - Métricas: calls, teus, nm, ktm

Uso:
    cube = ConnectivityCube()
    cube.add_period('2022-Q3', calls, teus, nm, ktm, axis_labels)
    cube.add_period('2025-Q3', calls, teus, nm, ktm, axis_labels)
    cube.save('connectivity.npz')

    cube = ConnectivityCube('connectivity.npz')
    cube.get('Piraeus', 'Rotterdam', '2024-Q3')               # Todas las categorías
    cube.get('Piraeus', 'ASIA', '2025-Q3', category=5)        # Solo ULCV
    cube.get('Piraeus', 'ASIA', '2025-Q3', category=[4, 5])   # Mainline (4+5)
    cube.by_category('Piraeus', 'ASIA', '2025-Q3')            # Desglose por categoría
    cube.delta('2022-Q3', '2025-Q3', 'ktm')
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# =============================================================================
# STANDALONE CONFIGURATION - Path to external data
# =============================================================================
CUBE_DATA_DIR = Path.home() / "proyectos/data/traffic/cube"
DEFAULT_CUBE_PATH = CUBE_DATA_DIR / "cube_4d.npz"
DEFAULT_PORTS_PATH = CUBE_DATA_DIR / "euromed_ports_active.csv"


def load_default_cube() -> 'ConnectivityCube':
    """Load the connectivity cube from the default location."""
    if not DEFAULT_CUBE_PATH.exists():
        raise FileNotFoundError(
            f"Cube not found at {DEFAULT_CUBE_PATH}. "
            f"Ensure data is installed in {CUBE_DATA_DIR}"
        )
    return ConnectivityCube(str(DEFAULT_CUBE_PATH))


class ConnectivityCube:
    """Cubo N×N×T×C de conectividad con índices dinámicos.

    Dimensiones:
      - N (axis): Puertos/zonas
      - T (periods): Períodos temporales
      - C (categories): Vessel categories 1-5
      - Métricas: calls, teus, nm, ktm
    """

    METRICS = ('calls', 'teus', 'nm', 'ktm')

    # Vessel categories (1-5) - ranges from v_fleet
    CATEGORY_NAMES = {
        1: 'Feeder',        # <3,000 TEU
        2: 'Panamax',       # 3,000-5,000 TEU
        3: 'Post Panamax',  # 5,000-10,000 TEU
        4: 'New Panamax',   # 10,000-15,000 TEU
        5: 'ULCV'           # >15,000 TEU
    }

    # EEA countries (EU + Norway, Iceland, Liechtenstein)
    EEA_COUNTRIES = {
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
        'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
        'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
        'Spain', 'Sweden', 'Norway', 'Iceland', 'Liechtenstein'
    }

    def __init__(self, npz_path: str = None):
        """
        Inicializa el cubo, opcionalmente cargando desde archivo.

        Args:
            npz_path: Ruta al archivo .npz (opcional, usa DEFAULT_CUBE_PATH si None y existe)
        """
        # Matrices 4D: (N, N, T, C)
        self.calls: Optional[np.ndarray] = None
        self.teus: Optional[np.ndarray] = None
        self.nm: Optional[np.ndarray] = None
        self.ktm: Optional[np.ndarray] = None

        # Labels
        self.axis: List[str] = []        # N labels (puertos/zonas)
        self.periods: List[str] = []     # T labels (períodos)
        self.categories: List[int] = [1, 2, 3, 4, 5]  # C labels (vessel categories)

        # Port metadata: {portname: {'country': str, 'zone': str, 'eea': float}}
        self.port_meta: Dict[str, Dict] = {}

        # Índices internos
        self._idx: Dict[str, int] = {}   # axis label → index
        self._tidx: Dict[str, int] = {}  # period label → index
        self._cidx: Dict[int, int] = {}  # category → index

        if npz_path:
            self.load(npz_path)

    def _build_index(self):
        """Reconstruye los diccionarios de índices."""
        self._idx = {name: i for i, name in enumerate(self.axis)}
        self._tidx = {p: i for i, p in enumerate(self.periods)}
        self._cidx = {cat: i for i, cat in enumerate(self.categories)}

    def _resolve_category(self, category) -> Union[slice, int, list]:
        """
        Convierte category param a índice(s) de array.

        Args:
            category: None (todas), int (una), list/range (varias)

        Returns:
            slice, int, o list de índices
        """
        if category is None:
            return slice(None)  # Todas las categorías
        if isinstance(category, int):
            return self._cidx[category]  # 1-indexed → 0-indexed via dict
        if isinstance(category, (list, range)):
            return [self._cidx[c] for c in category]  # Lista de índices
        raise ValueError(f"category debe ser None, int, list o range, recibido: {type(category)}")

    # =========================================================================
    # I/O
    # =========================================================================

    def save(self, path: str):
        """
        Guarda el cubo en formato .npz.

        Args:
            path: Ruta del archivo de salida
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            calls=self.calls,
            teus=self.teus,
            nm=self.nm,
            ktm=self.ktm,
            axis=np.array(self.axis, dtype=object),
            periods=np.array(self.periods, dtype=object),
            categories=np.array(self.categories, dtype=np.int32),
            port_meta=np.array(self.port_meta, dtype=object)
        )

    def load(self, path: str):
        """
        Carga el cubo desde archivo .npz.

        Backward compatible: si el archivo es 3D, expande a 4D con una sola categoría.

        Args:
            path: Ruta del archivo
        """
        data = np.load(path, allow_pickle=True)

        self.axis = list(data['axis'])
        self.periods = list(data['periods'])

        # Load categories (backward compatible con cubos 3D)
        if 'categories' in data:
            self.categories = list(data['categories'])
        else:
            self.categories = [1, 2, 3, 4, 5]  # Default

        # Load matrices
        calls = data['calls']
        teus = data['teus']
        nm = data['nm']
        ktm = data['ktm']

        # Backward compatibility: expandir 3D a 4D si es necesario
        if calls.ndim == 3:
            # Cubo antiguo 3D → expandir a 4D con categoría única (suma)
            C = len(self.categories)
            N, _, T = calls.shape
            # Crear 4D con la data en la última categoría (5=ULCV para cubos antiguos de ULCV)
            self.calls = np.zeros((N, N, T, C), dtype=calls.dtype)
            self.teus = np.zeros((N, N, T, C), dtype=teus.dtype)
            self.nm = np.zeros((N, N, T, C), dtype=nm.dtype)
            self.ktm = np.zeros((N, N, T, C), dtype=ktm.dtype)
            # Poner datos en categoría 5 (ULCV) - índice 4
            self.calls[:, :, :, 4] = calls
            self.teus[:, :, :, 4] = teus
            self.nm[:, :, :, 4] = nm
            self.ktm[:, :, :, 4] = ktm
        else:
            self.calls = calls
            self.teus = teus
            self.nm = nm
            self.ktm = ktm

        # Load port metadata (backward compatible)
        if 'port_meta' in data:
            pm = data['port_meta']
            self.port_meta = pm.item() if pm.ndim == 0 else dict(pm)
        else:
            self.port_meta = {}

        self._build_index()

    def add_period(self, label: str,
                   calls: np.ndarray,
                   teus: np.ndarray,
                   nm: np.ndarray,
                   ktm: np.ndarray,
                   axis_labels: List[str] = None):
        """
        Añade un período al cubo.

        Si es el primer período, inicializa las matrices.
        Si ya existen períodos, extiende la dimensión T.

        Args:
            label: Etiqueta del período (ej: '2024-Q3')
            calls: Matriz N×N×C de escalas (o N×N para backward compat)
            teus: Matriz N×N×C de TEUs
            nm: Matriz N×N×C de millas náuticas
            ktm: Matriz N×N×C de kilo-TEU-miles
            axis_labels: Labels de los ejes (requerido si es el primer período)
        """
        if label in self.periods:
            raise ValueError(f"Período '{label}' ya existe en el cubo")

        N = calls.shape[0]
        C = len(self.categories)

        # Backward compatibility: expandir 2D a 3D si es necesario
        if calls.ndim == 2:
            # Matriz 2D → expandir a 3D con datos en categoría 5 (ULCV)
            calls_3d = np.zeros((N, N, C), dtype=calls.dtype)
            teus_3d = np.zeros((N, N, C), dtype=teus.dtype)
            nm_3d = np.zeros((N, N, C), dtype=nm.dtype)
            ktm_3d = np.zeros((N, N, C), dtype=ktm.dtype)
            calls_3d[:, :, 4] = calls  # Categoría 5 = índice 4
            teus_3d[:, :, 4] = teus
            nm_3d[:, :, 4] = nm
            ktm_3d[:, :, 4] = ktm
            calls, teus, nm, ktm = calls_3d, teus_3d, nm_3d, ktm_3d

        if calls.shape != (N, N, C):
            raise ValueError(f"Matriz debe ser (N, N, C)={(N, N, C)}, recibido: {calls.shape}")

        # Primer período: inicializar
        if self.calls is None:
            if axis_labels is None:
                raise ValueError("axis_labels requerido para el primer período")
            if len(axis_labels) != N:
                raise ValueError(f"axis_labels ({len(axis_labels)}) no coincide con matriz ({N})")

            self.axis = list(axis_labels)
            self.periods = [label]

            # Crear matrices 4D con T=1
            self.calls = calls[:, :, np.newaxis, :]
            self.teus = teus[:, :, np.newaxis, :]
            self.nm = nm[:, :, np.newaxis, :]
            self.ktm = ktm[:, :, np.newaxis, :]

        else:
            # Verificar dimensiones
            if N != len(self.axis):
                raise ValueError(
                    f"Dimensión N ({N}) no coincide con existente ({len(self.axis)}). "
                    "Use expand_axis() para añadir puertos."
                )

            self.periods.append(label)

            # Extender dimensión T
            self.calls = np.concatenate([self.calls, calls[:, :, np.newaxis, :]], axis=2)
            self.teus = np.concatenate([self.teus, teus[:, :, np.newaxis, :]], axis=2)
            self.nm = np.concatenate([self.nm, nm[:, :, np.newaxis, :]], axis=2)
            self.ktm = np.concatenate([self.ktm, ktm[:, :, np.newaxis, :]], axis=2)

        self._build_index()

    # =========================================================================
    # Port Metadata
    # =========================================================================

    def set_port_meta(self, portname: str, country: str, zone: str, eea: float = None):
        """
        Establece metadatos para un puerto.

        Args:
            portname: Nombre del puerto
            country: País
            zone: Zona marítima (NORTH EUROPE, WEST MED, EAST MED, ATLANTIC)
            eea: Flag EEA (0.5 si es europeo, 0.0 si no). Si None, se calcula automáticamente.
        """
        if eea is None:
            eea = 0.5 if country in self.EEA_COUNTRIES else 0.0

        self.port_meta[portname] = {
            'country': country,
            'zone': zone,
            'eea': eea
        }

    def set_port_meta_batch(self, meta_list: List[Dict]):
        """
        Establece metadatos para múltiples puertos.

        Args:
            meta_list: Lista de dicts con keys: portname, country, zone, eea (opcional)
        """
        for m in meta_list:
            self.set_port_meta(
                m['portname'],
                m['country'],
                m['zone'],
                m.get('eea')
            )

    def get_port_meta(self, portname: str) -> Optional[Dict]:
        """
        Obtiene metadatos de un puerto.

        Args:
            portname: Nombre del puerto

        Returns:
            Dict con country, zone, eea o None si no existe
        """
        return self.port_meta.get(portname)

    def ports_by_zone(self, zone: str) -> List[str]:
        """
        Obtiene lista de puertos en una zona.

        Args:
            zone: Zona marítima

        Returns:
            Lista de nombres de puertos
        """
        return [p for p, m in self.port_meta.items() if m.get('zone') == zone]

    def ports_by_country(self, country: str) -> List[str]:
        """
        Obtiene lista de puertos de un país.

        Args:
            country: Nombre del país

        Returns:
            Lista de nombres de puertos
        """
        return [p for p, m in self.port_meta.items() if m.get('country') == country]

    def ports_eea(self, eea: bool = True) -> List[str]:
        """
        Obtiene lista de puertos EEA o no-EEA.

        Args:
            eea: True para puertos EEA, False para no-EEA

        Returns:
            Lista de nombres de puertos
        """
        threshold = 0.25  # 0.5 for EEA, 0.0 for non-EEA
        if eea:
            return [p for p, m in self.port_meta.items() if m.get('eea', 0) > threshold]
        else:
            return [p for p, m in self.port_meta.items() if m.get('eea', 0) <= threshold]

    # =========================================================================
    # Queries
    # =========================================================================

    def get(self, origin: str, dest: str, period: str, metric: str = 'calls',
            category: Union[int, List[int], range, None] = None) -> float:
        """
        Obtiene un valor específico.

        Args:
            origin: Puerto/zona origen
            dest: Puerto/zona destino
            period: Período
            metric: Métrica ('calls', 'teus', 'nm', 'ktm')
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Valor de la celda (suma si múltiples categorías)
        """
        i = self._idx[origin]
        j = self._idx[dest]
        t = self._tidx[period]
        c = self._resolve_category(category)
        m = getattr(self, metric)

        if isinstance(c, slice):
            return m[i, j, t, :].sum()
        elif isinstance(c, int):
            return m[i, j, t, c]
        else:  # list
            return m[i, j, t, c].sum()

    def row(self, origin: str, period: str, metric: str = 'calls',
            category: Union[int, List[int], range, None] = None) -> np.ndarray:
        """
        Obtiene todas las conexiones salientes de un puerto.

        Args:
            origin: Puerto/zona origen
            period: Período
            metric: Métrica
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Array de N valores (destinos)
        """
        i = self._idx[origin]
        t = self._tidx[period]
        c = self._resolve_category(category)
        m = getattr(self, metric)

        if isinstance(c, slice):
            return m[i, :, t, :].sum(axis=1)
        elif isinstance(c, int):
            return m[i, :, t, c]
        else:  # list
            return m[i, :, t, :][:, c].sum(axis=1)

    def col(self, dest: str, period: str, metric: str = 'calls',
            category: Union[int, List[int], range, None] = None) -> np.ndarray:
        """
        Obtiene todas las conexiones entrantes a un puerto.

        Args:
            dest: Puerto/zona destino
            period: Período
            metric: Métrica
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Array de N valores (orígenes)
        """
        j = self._idx[dest]
        t = self._tidx[period]
        c = self._resolve_category(category)
        m = getattr(self, metric)

        if isinstance(c, slice):
            return m[:, j, t, :].sum(axis=1)
        elif isinstance(c, int):
            return m[:, j, t, c]
        else:  # list
            return m[:, j, t, :][:, c].sum(axis=1)

    def timeseries(self, origin: str, dest: str, metric: str = 'calls',
                   category: Union[int, List[int], range, None] = None) -> np.ndarray:
        """
        Obtiene la serie temporal de una conexión.

        Args:
            origin: Puerto/zona origen
            dest: Puerto/zona destino
            metric: Métrica
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Array de T valores (períodos)
        """
        i = self._idx[origin]
        j = self._idx[dest]
        c = self._resolve_category(category)
        m = getattr(self, metric)

        if isinstance(c, slice):
            return m[i, j, :, :].sum(axis=1)
        elif isinstance(c, int):
            return m[i, j, :, c]
        else:  # list
            return m[i, j, :, :][:, c].sum(axis=1)

    def matrix(self, period: str, metric: str = 'calls',
               category: Union[int, List[int], range, None] = None) -> np.ndarray:
        """
        Obtiene la matriz N×N completa de un período.

        Args:
            period: Período
            metric: Métrica
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Matriz N×N
        """
        t = self._tidx[period]
        c = self._resolve_category(category)
        m = getattr(self, metric)

        if isinstance(c, slice):
            return m[:, :, t, :].sum(axis=2)
        elif isinstance(c, int):
            return m[:, :, t, c]
        else:  # list
            return m[:, :, t, :][:, :, c].sum(axis=2)

    def delta(self, p1: str, p2: str, metric: str = 'calls',
              category: Union[int, List[int], range, None] = None) -> np.ndarray:
        """
        Calcula la matriz delta entre dos períodos.

        Args:
            p1: Período inicial
            p2: Período final
            metric: Métrica
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Matriz N×N con diferencias (p2 - p1)
        """
        m1 = self.matrix(p1, metric, category)
        m2 = self.matrix(p2, metric, category)
        return m2 - m1

    def top_n(self, p1: str, p2: str, metric: str = 'calls',
              n: int = 10, gains: bool = True,
              category: Union[int, List[int], range, None] = None) -> List[Tuple[str, str, float]]:
        """
        Obtiene las N conexiones con mayor cambio.

        Args:
            p1: Período inicial
            p2: Período final
            metric: Métrica
            n: Número de resultados
            gains: True para mayores ganancias, False para mayores pérdidas
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Lista de (origen, destino, delta)
        """
        d = self.delta(p1, p2, metric, category)
        flat = d.flatten()
        N = len(self.axis)

        if gains:
            idx = np.argsort(flat)[-n:][::-1]
        else:
            idx = np.argsort(flat)[:n]

        return [(self.axis[i // N], self.axis[i % N], flat[i]) for i in idx]

    def sum_by_axis(self, period: str, metric: str = 'calls',
                    direction: str = 'origin',
                    category: Union[int, List[int], range, None] = None) -> Dict[str, float]:
        """
        Calcula totales por puerto/zona.

        Args:
            period: Período
            metric: Métrica
            direction: 'origin' (suma filas) o 'dest' (suma columnas)
            category: Categoría(s) - None=todas, int=una, list/range=varias

        Returns:
            Dict {puerto: total}
        """
        m = self.matrix(period, metric, category)

        if direction == 'origin':
            totals = m.sum(axis=1)  # Suma por fila (destinos)
        else:
            totals = m.sum(axis=0)  # Suma por columna (orígenes)

        return {self.axis[i]: totals[i] for i in range(len(self.axis))}

    def by_category(self, origin: str, dest: str, period: str,
                    metric: str = 'calls') -> Dict[int, float]:
        """
        Retorna valores desglosados por categoría.

        Args:
            origin: Puerto/zona origen
            dest: Puerto/zona destino
            period: Período
            metric: Métrica

        Returns:
            Dict {1: x, 2: y, 3: z, 4: w, 5: v} con valor por categoría
        """
        i = self._idx[origin]
        j = self._idx[dest]
        t = self._tidx[period]
        m = getattr(self, metric)
        return {cat: float(m[i, j, t, c]) for cat, c in self._cidx.items()}

    def category_share(self, origin: str, dest: str, period: str,
                       metric: str = 'calls') -> Dict[int, float]:
        """
        Retorna porcentaje de cada categoría.

        Args:
            origin: Puerto/zona origen
            dest: Puerto/zona destino
            period: Período
            metric: Métrica

        Returns:
            Dict {1: %, 2: %, ...} con porcentaje por categoría
        """
        values = self.by_category(origin, dest, period, metric)
        total = sum(values.values())
        if total == 0:
            return {cat: 0.0 for cat in values}
        return {cat: v / total * 100 for cat, v in values.items()}

    # =========================================================================
    # Utilidades
    # =========================================================================

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Retorna (N, N, T, C)."""
        if self.calls is None:
            return (0, 0, 0, 0)
        return self.calls.shape

    def __repr__(self) -> str:
        N, _, T, C = self.shape
        return f"ConnectivityCube(axis={N}, periods={T}, categories={C}, metrics={self.METRICS})"

    def info(self) -> str:
        """Retorna información detallada del cubo."""
        N, _, T, C = self.shape
        lines = [
            f"ConnectivityCube",
            f"  Shape: {N} x {N} x {T} x {C}",
            f"  Axis ({N}): {self.axis[:5]}{'...' if N > 5 else ''}",
            f"  Periods ({T}): {self.periods}",
            f"  Categories ({C}): {self.categories}",
            f"  Metrics: {self.METRICS}",
        ]
        if T > 0:
            lines.append(f"  Total calls: {self.calls.sum():,.0f}")
            lines.append(f"  Total TEUs: {self.teus.sum():,.0f}")

        # Port metadata summary
        if self.port_meta:
            zones = {}
            eea_count = 0
            for p, m in self.port_meta.items():
                z = m.get('zone', 'Unknown')
                zones[z] = zones.get(z, 0) + 1
                if m.get('eea', 0) > 0:
                    eea_count += 1
            lines.append(f"  Port metadata: {len(self.port_meta)} ports")
            lines.append(f"    Zones: {dict(zones)}")
            lines.append(f"    EEA ports: {eea_count}, Non-EEA: {len(self.port_meta) - eea_count}")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ConnectivityCube utilities')
    parser.add_argument('file', nargs='?', default=str(DEFAULT_CUBE_PATH),
                        help=f'Path to .npz file (default: {DEFAULT_CUBE_PATH})')
    parser.add_argument('--info', action='store_true', help='Show cube info')
    parser.add_argument('--get', nargs=3, metavar=('ORIGIN', 'DEST', 'PERIOD'),
                        help='Get specific value')
    parser.add_argument('--delta', nargs=2, metavar=('P1', 'P2'),
                        help='Show top changes between periods')
    parser.add_argument('--metric', default='calls',
                        help='Metric to use (default: calls)')
    parser.add_argument('-n', type=int, default=10,
                        help='Number of results for top_n')

    args = parser.parse_args()

    cube = ConnectivityCube(args.file)

    if args.info or (not args.get and not args.delta):
        print(cube.info())

    if args.get:
        origin, dest, period = args.get
        value = cube.get(origin, dest, period, args.metric)
        print(f"{origin} -> {dest} ({period}): {value:,.0f} {args.metric}")

    if args.delta:
        p1, p2 = args.delta
        print(f"\nTop {args.n} gains ({p1} -> {p2}, {args.metric}):")
        for origin, dest, delta in cube.top_n(p1, p2, args.metric, args.n, gains=True):
            print(f"  {origin:20} -> {dest:20}: {delta:+,.0f}")

        print(f"\nTop {args.n} losses ({p1} -> {p2}, {args.metric}):")
        for origin, dest, delta in cube.top_n(p1, p2, args.metric, args.n, gains=False):
            print(f"  {origin:20} -> {dest:20}: {delta:+,.0f}")

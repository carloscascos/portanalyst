# Port Analyst - Claude Code Skill

## Repository

- **GitHub:** https://github.com/carloscascos/portanalyst
- **Local:** `~/.claude/skills/portanalyst/`

## Workflow: Commit & Deploy

Cada vez que se haga un commit y push, **actualizar también el ZIP**:

```bash
cd ~/.claude/skills/portanalyst
git add -A && git commit -m "mensaje" && git push
cd .. && zip -r portanalyst.zip portanalyst -x "*.git*" -x "*__pycache__*" -x "*.pyc"
```

El archivo `portanalyst.zip` está en `~/.claude/skills/` (nivel superior, junto a la carpeta).

## Estructura

```
~/.claude/skills/
├── portanalyst.zip             # Distribución (actualizar tras cada push)
└── portanalyst/
    ├── SKILL.md                # Definición principal
    ├── CLAUDE.md               # Este archivo
    ├── scripts/
    │   ├── cube.py             # Connectivity Cube
    │   ├── heatmap.py          # TEU × Distance heatmaps
    │   ├── portmatrix.py       # ULCV matrices
    │   └── service.py          # Service Analysis
    └── references/
        ├── deepdive-methodology.md
        ├── groupdive-methodology.md
        ├── cube-schema.md
        ├── query-templates.md
        └── service-methodology.md
```

## Dependencias

- **MCP Server:** traffic-db (MySQL via MCP)
- **Python:** numpy, matplotlib, scipy, pandas
- **External Data:** `~/proyectos/data/traffic/cube/cube_4d.npz`

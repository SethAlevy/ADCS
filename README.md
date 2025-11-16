# Small Satellite ADCS Simulation

A Python-based Attitude Determination and Control System (ADCS) simulation for small satellites in Low Earth Orbit. Implements detumbling (B-dot) and pointing (B-cross) control using magnetorquers, with sensor fusion algorithms (TRIAD, QUEST, EKF).

## Quick Start

Install dependencies with Poetry:

```powershell
poetry install
```

Run the simulation:

```powershell
poetry run python adcs.py
```

## Documentation

Full documentation including theoretical background, API reference, and usage examples:

**[https://adcs-a-drabik.com](https://adcs-a-drabik.com)**

## Features

- Orbital propagation (SGP4/TLE)
- Earth magnetic field modeling (IGRF)
- Magnetometer & Sun sensor simulation
- Sensor fusion (TRIAD, QUEST, EKF)
- Detumbling & pointing control
- Interactive visualization (matplotlib, plotly)

## Requirements

- Python 3.13
- Poetry for dependency management


## Contact

GitHub: [@arekdrabik](https://github.com/arekdrabik)


import sys
from pathlib import Path

# Add project root (ADCS) to sys.path so `core`, `spacecraft`, etc. import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

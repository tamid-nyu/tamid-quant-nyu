import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT.parent / "btc-kalshi-digital"))

"""Make top-level packages (`eval`, `scripts`, `src.motifql`) importable in tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

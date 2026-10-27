"""Make top-level packages (`eval`, `scripts`) importable in tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

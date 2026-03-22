# This makes feature_engineering importable from this folder
import sys
from pathlib import Path

# Add this folder to path so pickle can find feature_engineering module
sys.path.insert(0, str(Path(__file__).resolve().parent))
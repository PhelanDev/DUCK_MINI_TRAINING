"""
HD verification renderer for CyberDuck V2 Pro.
"""
from pathlib import Path
import sys

# Ensure root scripts can be imported or re-use implementation
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from scripts.render_v2_hd import render_views

if __name__ == "__main__":
    render_views()

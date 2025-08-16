#!/usr/bin/env python3
"""
Script to run the Multi-Modal Earnings Call Forecaster Dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit dashboard"""
    # Get the path to the dashboard
    dashboard_path = Path(__file__).parent / "src" / "visualization" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        print("Please ensure the project structure is correct.")
        sys.exit(1)
    
    print("🚀 Starting Multi-Modal Earnings Call Forecaster Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        # Run the Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

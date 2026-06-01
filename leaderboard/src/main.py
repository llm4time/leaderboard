from pathlib import Path
import subprocess
import sys


def run():
  home_path = Path(__file__).with_name("home.py")
  subprocess.run([sys.executable, "-m", "streamlit", "run", str(home_path)])


if __name__ == "__main__":
  run()

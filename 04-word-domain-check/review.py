from pathlib import Path
import sys

from experiment import main


if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parent, '1.9.4.1'))

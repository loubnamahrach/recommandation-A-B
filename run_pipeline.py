"""Commande unique de relance : python run_pipeline.py [--promote]"""
import sys

from src import analyze, data, promote, simulate, train


def main():
    data.main()
    train.main()
    simulate.main()
    analyze.main()
    if "--promote" in sys.argv:
        promote.main()


if __name__ == "__main__":
    main()

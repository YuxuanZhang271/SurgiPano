"""Stitch-seg + persistent TAPIR + EMDQ-inspired smooth residual deformation."""
from pipeline import run

if __name__ == '__main__':
    run(nonrigid=True)

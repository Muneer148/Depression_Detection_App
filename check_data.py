"""Backward-compatible entry point for the dataset audit utility."""
from scripts.audit_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())

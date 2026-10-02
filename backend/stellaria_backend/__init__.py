"""Stellaria backend: FastAPI server around the DDSP-SVC engine.

The API process never imports torch; heavy work runs in subprocesses.
"""

__version__ = "0.1.0"

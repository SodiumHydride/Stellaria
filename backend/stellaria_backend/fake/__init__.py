"""Stand-ins for the engine scripts used with ``--fake-engine`` (no torch, no GPU).

They take the same command line as ``preprocess.py`` / ``train_reflow.py``, honour the same
files (``STOP``, ``train.max_steps``, ``model_<step>.pt``) and write real tfevents files, so
the backend runs exactly the same code paths as with the real engine.
"""

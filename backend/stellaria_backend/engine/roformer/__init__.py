"""Roformer vocal separation (BS-Roformer, Mel-Band-Roformer), vendored from
Music-Source-Separation-Training (see NOTICE.txt).

``config`` reads MSST configs without torch (the API process uses it); ``inference`` and the
model modules import torch and only run inside the separation worker.
"""

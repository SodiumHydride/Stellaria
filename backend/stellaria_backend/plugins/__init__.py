"""User plugins: folders with a script (``command`` runtime) or a ``process(job)`` entry
(``python`` runtime) that the canvas can use as mix processors or effects.

Internal interfaces (creation-v2 §6): ``PluginRegistry`` (``ctx.plugins``), ``run_plugin`` and
``check_python``. The API process never imports torch: interpreters are probed in subprocesses.
"""

from .pycheck import check_python
from .registry import PluginRegistry
from .runner import run_plugin

__all__ = ["PluginRegistry", "check_python", "run_plugin"]

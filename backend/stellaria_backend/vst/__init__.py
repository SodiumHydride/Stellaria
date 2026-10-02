"""VST3 / Audio Unit hosting through the optional ``vsthost`` add-on (pedalboard, GPL-3.0).

* ``registry.VstRegistry`` (``ctx.vst``, API process): folders, scanning, cache, editor jobs.
  Never imports pedalboard: plugins are only touched in short-lived ``worker.py`` subprocesses.
* ``processing`` (inside the ``dsp`` subprocess): ``apply()`` runs a plugin on audio.

Deliberately empty: importing ``stellaria_backend.vst.processing`` must stay light.
"""

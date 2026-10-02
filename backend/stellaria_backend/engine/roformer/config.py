"""Music-Source-Separation-Training config files, read without torch.

* ``load_config``: ``yaml.safe_load`` plus the tags MSST configs carry (``!!python/tuple`` →
  tuple; any other ``!!python/…`` or local ``!tag`` is read as the plain value it wraps).
* ``describe``: which architecture a config builds (``num_bands`` → Mel-Band-Roformer,
  ``freqs_per_bands`` → BS-Roformer), what the model is for (``separation``: it gives the vocals;
  ``cleanup``: it gives the dry voice without reverb, echo or noise), which output that is, and
  the inference settings. Configs this app cannot run raise ``ConfigProblem`` with a
  ``separation.skip.*`` code.
* ``cleanup_kind``: what a cleanup checkpoint removes, from its file name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
ARCHS = ("bs_roformer", "mel_band_roformer")
VOCALS = ("vocals", "vocal")
DRY = ("dry", "noreverb", "no_reverb", "dereverb", "deverb")  # the clean voice of a cleanup model
DEFAULT_CHUNK = 352800  # 8 s at 44.1 kHz

_COMMON_PARAMS = {
    "dim", "depth", "stereo", "num_stems", "time_transformer_depth", "freq_transformer_depth",
    "linear_transformer_depth", "dim_head", "heads", "attn_dropout", "ff_dropout", "flash_attn", "dim_freqs_in",
    "stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "zero_dc", "mask_estimator_depth",
    "multi_stft_resolution_loss_weight", "multi_stft_resolutions_window_sizes", "multi_stft_hop_size",
    "multi_stft_normalized", "mlp_expansion_factor", "use_torch_checkpoint", "skip_connection", "use_pope",
}
# constructor parameters a YAML config may set, per architecture (the window functions are
# callables, which a config cannot express)
PARAMS: dict[str, frozenset[str]] = {
    "bs_roformer": frozenset(_COMMON_PARAMS | {"freqs_per_bands"}),
    "mel_band_roformer": frozenset(_COMMON_PARAMS | {"num_bands", "sample_rate", "match_input_audio_length"}),
}
CALLABLE_PARAMS = frozenset({"stft_window_fn", "multi_stft_window_fn"})


class ConfigProblem(Exception):
    """A config this app cannot use: ``code`` (``separation.skip.*``) and its message params."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(f"{code} {params}")
        self.code = code
        self.params = {k: str(v) for k, v in params.items()}


class _Loader(yaml.SafeLoader):
    pass


def _tuple(loader: _Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node, deep=True))  # type: ignore[arg-type]


def _plain(loader: _Loader, _suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)  # type: ignore[arg-type]


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)
_Loader.add_multi_constructor("!", _plain)


def parse_config(text: str) -> dict[str, Any]:
    try:
        data = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigProblem("separation.skip.bad_config", reason=str(exc).splitlines()[0]) from exc
    if not isinstance(data, dict):
        raise ConfigProblem("separation.skip.bad_config", reason="not a mapping")
    return data


def load_config(path: Path | str) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigProblem("separation.skip.bad_config", reason=str(exc)) from exc
    return parse_config(text)


@dataclass
class RoformerSpec:
    arch: str  # 'bs_roformer' | 'mel_band_roformer'
    instruments: list[str]
    outputs: list[str]  # what the model returns, in order
    # index into ``outputs`` of what this app keeps (the vocals; the dry voice); -1: the model
    # returns the other part (the accompaniment; the reverb), so the kept part = mix − it
    target: int
    role: str  # 'separation' | 'cleanup'
    sample_rate: int
    chunk_size: int
    num_overlap: int
    batch_size: int
    stereo: bool
    use_amp: bool


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _guess_arch(config: dict[str, Any]) -> str:
    model = _section(config, "model")
    if "htdemucs" in config:
        return "htdemucs"
    if "num_subbands" in model:
        return "mdx23c"
    return "unknown"


def architecture(config: dict[str, Any], hint: str | None = None) -> str:
    """``bs_roformer`` or ``mel_band_roformer``; ``hint`` is a model type named next to the
    checkpoint (e.g. by MSST-GUI's model list)."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigProblem("separation.skip.bad_config", reason="no 'model' section")
    if hint and hint not in ARCHS:
        raise ConfigProblem("separation.skip.arch", arch=hint)
    if "num_bands" in model:
        arch = "mel_band_roformer"
    elif "freqs_per_bands" in model:
        arch = "bs_roformer"
    elif hint:
        arch = hint
    else:
        raise ConfigProblem("separation.skip.arch", arch=_guess_arch(config))
    unknown = sorted(set(model) - PARAMS[arch])
    if model.get("use_pope"):  # needs a package this app does not ship
        unknown.append("use_pope")
    if unknown:
        raise ConfigProblem("separation.skip.params", params=", ".join(unknown))
    return arch


def model_params(config: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments of the model class (lists become tuples, as the tuple tags intend)."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in _section(config, "model").items()}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _target(outputs: list[str], instruments: list[str]) -> tuple[str, int]:
    """(role, index of the kept output): the vocals, else the dry voice; -1 for a two-instrument
    model that returns the other one (an instrumental model, a reverb model)."""
    lowered = [o.lower() for o in outputs]
    names = [i.lower() for i in instruments]
    for role, wanted in (("separation", VOCALS), ("cleanup", DRY)):
        index = next((i for i, name in enumerate(lowered) if name in wanted), None)
        if index is not None:
            return role, index
    if len(outputs) == 1 and len(names) == 2 and lowered[0] in names:
        for role, wanted in (("separation", VOCALS), ("cleanup", DRY)):
            if any(n in wanted for n in names):
                return role, -1
    raise ConfigProblem("separation.skip.no_vocals", instruments=", ".join(instruments or outputs))


def cleanup_kind(name: str) -> str:
    """What a cleanup checkpoint removes, guessed from its file name: ``denoise``,
    ``dereverb_echo`` or ``dereverb`` (configs say only 'dry')."""
    lowered = name.lower()
    if "noise" in lowered:
        return "denoise"
    if "echo" in lowered:
        return "dereverb_echo"
    return "dereverb"


def describe(config: dict[str, Any], hint: str | None = None) -> RoformerSpec:
    arch = architecture(config, hint)
    model, audio = _section(config, "model"), _section(config, "audio")
    training, inference = _section(config, "training"), _section(config, "inference")
    instruments = [str(i) for i in training.get("instruments") or []]
    target = training.get("target_instrument")
    outputs = [str(target)] if target else list(instruments)
    if not outputs:
        raise ConfigProblem("separation.skip.bad_config", reason="no instruments")
    num_stems = _int(model.get("num_stems", 1), 1)
    if num_stems != len(outputs):
        raise ConfigProblem("separation.skip.bad_config", reason=f"num_stems {num_stems} != {len(outputs)} outputs")
    role, target = _target(outputs, instruments)
    sample_rate = _int(audio.get("sample_rate") or model.get("sample_rate"), 44100)
    chunk = _int(inference.get("chunk_size") or audio.get("chunk_size"), 0)
    if chunk <= 0:
        dim_t = _int(inference.get("dim_t") or audio.get("dim_t"), 0)
        hop = _int(model.get("stft_hop_length") or audio.get("hop_length"), 441)
        chunk = hop * (dim_t - 1) if dim_t > 1 else DEFAULT_CHUNK
    num_overlap = max(1, _int(inference.get("num_overlap"), 2))
    if chunk < 20 or chunk // num_overlap < 1:
        raise ConfigProblem("separation.skip.bad_config", reason=f"chunk_size {chunk}")
    return RoformerSpec(
        arch=arch, instruments=instruments, outputs=outputs, target=target, role=role, sample_rate=sample_rate,
        chunk_size=chunk, num_overlap=num_overlap, batch_size=max(1, _int(inference.get("batch_size"), 1)),
        stereo=bool(model.get("stereo", False)), use_amp=bool(training.get("use_amp", True)),
    )

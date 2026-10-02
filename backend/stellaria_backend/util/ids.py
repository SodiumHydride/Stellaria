"""Short random IDs with a type prefix (``chr_``, ``clp_``, ``mdl_`` ...)."""

import secrets

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def new_id(prefix: str, length: int = 10) -> str:
    return f"{prefix}_" + "".join(secrets.choice(_ALPHABET) for _ in range(length))

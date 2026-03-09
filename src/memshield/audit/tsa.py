"""RFC 3161 Time-Stamp Authority (TSA) client for the MemShield audit log."""
from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)


def stamp(data_hash: bytes, tsa_url: str) -> str | None:
    """Send *data_hash* to an RFC 3161 TSA and return the timestamp token.

    The TSA call is treated as *additive evidence* — failures are logged as
    warnings and ``None`` is returned rather than raising an exception.  The
    caller must never block an audit write because of a TSA failure.

    Args:
        data_hash: 32-byte SHA-256 digest of the data to timestamp.
        tsa_url: HTTP(S) URL of the RFC 3161 Time-Stamp Authority endpoint.

    Returns:
        Base64-encoded DER timestamp token on success, or ``None`` on any
        failure (including network errors and missing ``rfc3161ng`` library).
    """
    try:
        import rfc3161ng  # type: ignore[import]
    except ImportError:
        logger.warning(
            "rfc3161ng is not installed; TSA stamping is disabled. "
            "Install it with: pip install 'memshield[audit]'"
        )
        return None

    try:
        stamper = rfc3161ng.RemoteTimestamper(
            tsa_url,
            hashname="sha256",
            include_tsa_certificate=True,
        )
        token_der: bytes = stamper(digest=data_hash, nonce=True)
        return base64.b64encode(token_der).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.warning("TSA stamping failed (url=%s): %s", tsa_url, exc)
        return None

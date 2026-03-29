r"""Patch DJITelloPy's video URL before PyAV opens the UDP stream.

``BackgroundFrameRead`` decodes H.264 as fast as FFmpeg allows, while gesture code
often processes frames slower. The default FFmpeg UDP FIFO is small, so packets
pile up (``Circular buffer overrun``). After that backlog drains, the stream is
"live" and bit errors from Wi‑Fi/UDP are more likely to abort ``container.decode``,
which kills the background thread — OpenCV then appears frozen.

``djitellopy.tello.Tello.get_udp_video_address`` documents
``?overrun_nonfatal=1&fifo_size=...`` in a comment but does not enable them.

**FIFO size:** defaults to **2 MiB** — enough for most Tello bursts. A previous
**16 MiB** default triggered ``av.error.MemoryError`` (Errno 12) on some Windows
PyAV/FFmpeg builds when opening the UDP input.

Override with environment variable **`MLX_TELLO_UDP_FIFO_BYTES`** (integer bytes,
clamped to **256 KiB … 8 MiB**). Example:

``$env:MLX_TELLO_UDP_FIFO_BYTES = '4194304'``

Import this module **before** ``from djitellopy import Tello``.
"""

from __future__ import annotations

import os

_applied = False

_DEFAULT_FIFO = 2 * 1024 * 1024
_MIN_FIFO = 256 * 1024
_MAX_FIFO = 8 * 1024 * 1024


def _ffmpeg_udp_query() -> str:
    n = _DEFAULT_FIFO
    raw = os.environ.get("MLX_TELLO_UDP_FIFO_BYTES", "").strip()
    if raw:
        try:
            n = int(raw, 10)
        except ValueError:
            n = _DEFAULT_FIFO
    n = max(_MIN_FIFO, min(n, _MAX_FIFO))
    return f"fifo_size={n}&overrun_nonfatal=1"


def apply_mlx_djitellopy_udp_video_patch() -> None:
    global _applied
    if _applied:
        return
    from djitellopy import tello as tello_mod

    def get_udp_video_address(self) -> str:
        return (
            f"udp://@{self.VS_UDP_IP}:{self.vs_udp_port}?{_ffmpeg_udp_query()}"
        )

    tello_mod.Tello.get_udp_video_address = get_udp_video_address  # type: ignore[method-assign]
    _applied = True


apply_mlx_djitellopy_udp_video_patch()

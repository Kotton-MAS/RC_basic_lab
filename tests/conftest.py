"""テスト共通のヘルパー.

PNG の解像度は「保存後のファイルから実測する」。``rcParams`` を読むだけでは
「設定はしたが savefig には効いていない」を検出できないため、PNG の ``pHYs``
チャンク (物理解像度) を直接読む。
"""

from __future__ import annotations

import struct
from pathlib import Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_METERS_PER_INCH = 0.0254
_UNIT_METER = 1


def png_dpi(path: Path) -> float:
    """PNG の ``pHYs`` チャンクから dpi を実測する。

    Raises:
        ValueError: PNG でない / ``pHYs`` が無い / 単位がメートルでない場合。
    """
    data = path.read_bytes()
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError(f"PNG ではありません: {path}")
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs":
            pixels_per_unit_x, _, unit = struct.unpack(">IIB", body)
            if unit != _UNIT_METER:
                raise ValueError(f"pHYs の単位がメートルではありません: {unit}")
            return pixels_per_unit_x * _METERS_PER_INCH
        offset += 12 + length
    raise ValueError(f"pHYs チャンクがありません: {path}")


__all__ = ["png_dpi"]

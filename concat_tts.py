#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concat_tts.py
Module ghép nối dạng sóng âm thanh (Concatenative Synthesis Engine).
- Chuẩn hóa biên độ (RMS matching).
- Thuật toán làm mịn biên nối Cross-fade đa mức (0, 5, 15, 30 ms).
- Ghi nhận chính xác mốc thời gian biên nối (boundary markers) phục vụ trực quan hóa.
- Sinh ra: raw_concat.wav (ghép thô) và smoothed concat audio.
"""

import numpy as np
import soundfile as sf

def load_unit(path: str, sr: int = 16000) -> np.ndarray:
    """Tải một file âm thanh đơn vị, chuyển sang mono float32 và xác thực Fs."""
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == sr, f"Tần số lấy mẫu không khớp ({fs} != {sr}) tại file {path}"
    return x

def rms(x: np.ndarray) -> float:
    """Tính Root Mean Square của tín hiệu."""
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    """Cân bằng mức năng lượng theo ngưỡng target và chống clipping."""
    y = x * (target / rms(x))
    return np.clip(y, -0.99, 0.99)

def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Hòa trộn mượt hai đoạn tín hiệu a và b tại vùng gối đầu gồm n mẫu.
    Sử dụng linear ramp alpha từ 0.0 đến 1.0.
    Nếu n == 0, ghép trực tiếp không hòa trộn (raw concatenation).
    """
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_units(paths: list[str], sr: int = 16000, xfade_ms: float = 15.0) -> tuple[np.ndarray, list[float]]:
    """
    Ghép nối chuỗi file âm thanh theo đường dẫn với thời lượng crossfade chỉ định.
    Trả về:
        y: mảng tín hiệu âm thanh sau khi ghép nối.
        boundaries: danh sách thời điểm (tính bằng giây) của các biên ghép nối.
    """
    if not paths:
        return np.array([], dtype=np.float32), []
        
    xs = [match_rms(load_unit(p, sr)) for p in paths]
    y = xs[0]
    n = int(sr * xfade_ms / 1000.0)
    
    # Ghi nhận vị trí biên nối (ranh giới giữa các từ)
    boundaries = []
    current_samples = len(y)
    
    for x in xs[1:]:
        # Vị trí biên nối xấp xỉ điểm giữa vùng hòa trộn
        boundary_time = (current_samples - n / 2.0) / sr
        boundaries.append(max(0.0, boundary_time))
        
        y = crossfade(y, x, n)
        current_samples = len(y)
        
    return y, boundaries

def synthesize_raw_baseline(paths: list[str], sr: int = 16000) -> tuple[np.ndarray, list[float]]:
    """Ghép nối thô (raw concatenation) không sử dụng cross-fade (xfade_ms=0)."""
    return concat_units(paths, sr=sr, xfade_ms=0.0)

if __name__ == "__main__":
    from inventory import SpeechInventory
    from normalize import normalize_queue_text
    
    inv = SpeechInventory()
    tokens = normalize_queue_text("Mời khách hàng số 23 đến quầy 5.")
    # Chọn cố định candidate 1 cho baseline
    raw_paths = [inv.get_candidate_paths(w)[0] for w in tokens]
    
    y_raw, bounds = synthesize_raw_baseline(raw_paths)
    sf.write("audio_output/raw_concat.wav", y_raw, 16000)
    print(f"=== Đã tạo raw_concat.wav ({len(y_raw)/16000:.3f}s, {len(bounds)} boundaries) ===")

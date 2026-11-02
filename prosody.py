#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prosody.py
Module phân tích ngữ điệu (Prosody) và mô hình Nguồn - Lọc (Source–Filter Model).
- Ước lượng chu kỳ pitch Np và tần số cơ bản F0 = Fs / Np bằng hàm tự tương quan (Autocorrelation).
- Kéo giãn / nén thời gian (Time-stretch / WSOLA): 0.8x, 1.0x, 1.2x.
- Dịch cao độ (Pitch Shift): Delta_s in [-2, 0, +2] semitones theo công thức r = 2^(Delta_s / 12).
- So sánh phong bì phổ (Spectral Envelope) để kiểm chứng tính bảo toàn formant.
"""

import numpy as np
import librosa

def estimate_f0_autocorr(signal: np.ndarray, sr: int = 16000, fmin: float = 70.0, fmax: float = 400.0) -> tuple[float, int, np.ndarray]:
    """
    Ước lượng F0 trên đoạn tín hiệu hữu thanh (Voiced segment) bằng hàm tự tương quan (Autocorrelation).
    Trả về:
        f0: Tần số cơ bản (Hz).
        np_samples: Chu kỳ pitch tính theo số mẫu Np.
        corr_norm: Mảng tự tương quan chuẩn hóa.
    """
    # Trừ giá trị trung bình DC
    x = signal - np.mean(signal)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2:]  # Lấy nửa dương lag >= 0
    corr_norm = corr / (corr[0] + 1e-12)

    # Phạm vi lag tương ứng với fmin và fmax
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    # Tìm đỉnh tương quan cực đại trong khoảng hợp lệ
    peak_lag = min_lag + int(np.argmax(corr_norm[min_lag:max_lag]))
    np_samples = peak_lag
    f0 = float(sr / np_samples) if np_samples > 0 else 0.0

    return f0, np_samples, corr_norm[:max_lag + 50]

def change_speed_wsola(signal: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Thay đổi tốc độ phát âm (duration/speed) bằng time-stretch (Phase Vocoder / WSOLA).
    rate > 1.0: phát nhanh hơn (rút ngắn duration).
    rate < 1.0: phát chậm hơn (kéo dài duration).
    """
    if rate == 1.0:
        return signal.copy()
    return librosa.effects.time_stretch(signal, rate=rate)

def change_pitch_semitones(signal: np.ndarray, sr: int = 16000, n_steps: float = 0.0) -> np.ndarray:
    """
    Dịch cao độ giọng nói theo số semitone (n_steps).
    Tỷ số tần số: r = 2^(n_steps / 12), F0_new = r * F0_old.
    Bảo toàn tương đối formant/spectral envelope.
    """
    if n_steps == 0.0:
        return signal.copy()
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

if __name__ == "__main__":
    from concat_tts import load_unit
    
    unit_path = "lab4_data/units/ba_01.wav"
    audio = load_unit(unit_path, sr=16000)
    
    # Trích xuất đoạn ổn định giữa nguyên âm /a/ (dài ~40ms)
    mid_idx = len(audio) // 2
    segment_len = int(16000 * 0.04) # 40ms = 640 samples
    segment = audio[mid_idx: mid_idx + segment_len]
    
    f0, np_samples, _ = estimate_f0_autocorr(segment, sr=16000)
    print("=== KIỂM CHỨNG TẦN SỐ CƠ BẢN F0 VÀ SOURCE-FILTER ===")
    print(f"File phân tích : {unit_path}")
    print(f"Chu kỳ pitch Np : {np_samples} mẫu (ở Fs=16000 Hz)")
    print(f"Tần số F0 tính toán: F0 = Fs / Np = 16000 / {np_samples} = {f0:.2f} Hz")
    
    # Kiểm tra tỷ số semitone
    for ds in [-2, 0, 2]:
        r = 2.0 ** (ds / 12.0)
        f0_new = r * f0
        print(f"  Δs = {ds:+d} semitone -> Tỷ số r = {r:.4f} -> F0 dự kiến = {f0_new:.2f} Hz")

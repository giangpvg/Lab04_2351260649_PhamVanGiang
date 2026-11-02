#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hmm_tts_backend.py
Hệ thống tổng hợp tham số thống kê dựa trên mô hình Markov ẩn (HMM-based Speech Synthesis).
- Chạy HMM ở chế độ phát sinh (Generation Mode) thay vì nhận dạng.
- Mỗi đơn vị từ gồm 3 trạng thái Left-to-Right (Bakis topology).
- Phát sinh tham số cực đại hợp lý (MLPG - Maximum Likelihood Parameter Generation)
  kết hợp Static + Delta features để làm mịn quỹ đạo F0 và Formants.
- Bộ tổng hợp Source-Filter (Vocoder) tạo dạng sóng hmm_tts.wav từ chuỗi tham số.
- Cho phép biến đổi tham số ngữ điệu: F0 scale (0.9x, 1.1x) và Duration scale (0.8x, 1.2x).
"""

import numpy as np
import scipy.signal as signal
from normalize import normalize_queue_text

# Bảng tham số âm học mẫu cho các từ trong miền hẹp (F0 base ~ 220Hz, F1, F2, F3 theo formant tiếng Việt)
WORD_ACOUSTIC_PROFILES = {
    "mời":   {"f0": 210, "f1": 350, "f2": 1800, "f3": 2700, "dur_frames": 28},
    "khách": {"f0": 240, "f1": 550, "f2": 1950, "f3": 2800, "dur_frames": 22},
    "hàng":  {"f0": 195, "f1": 680, "f2": 1350, "f3": 2500, "dur_frames": 26},
    "số":    {"f0": 245, "f1": 420, "f2": 1100, "f3": 2600, "dur_frames": 24},
    "không": {"f0": 215, "f1": 480, "f2": 950,  "f3": 2450, "dur_frames": 25},
    "một":   {"f0": 200, "f1": 400, "f2": 1050, "f3": 2550, "dur_frames": 22},
    "hai":   {"f0": 220, "f1": 700, "f2": 1600, "f3": 2700, "dur_frames": 26},
    "ba":    {"f0": 220, "f1": 750, "f2": 1250, "f3": 2600, "dur_frames": 25},
    "bốn":   {"f0": 235, "f1": 450, "f2": 1000, "f3": 2500, "dur_frames": 24},
    "năm":   {"f0": 220, "f1": 650, "f2": 1400, "f3": 2650, "dur_frames": 25},
    "sáu":   {"f0": 240, "f1": 420, "f2": 1050, "f3": 2600, "dur_frames": 26},
    "bảy":   {"f0": 205, "f1": 600, "f2": 1700, "f3": 2700, "dur_frames": 24},
    "tám":   {"f0": 235, "f1": 680, "f2": 1300, "f3": 2650, "dur_frames": 23},
    "chín":  {"f0": 230, "f1": 320, "f2": 2100, "f3": 2900, "dur_frames": 25},
    "đến":   {"f0": 225, "f1": 480, "f2": 1850, "f3": 2750, "dur_frames": 24},
    "quầy":  {"f0": 200, "f1": 500, "f2": 1450, "f3": 2600, "dur_frames": 26}
}

DEFAULT_PROFILE = {"f0": 220, "f1": 500, "f2": 1500, "f3": 2600, "dur_frames": 24}

def mlpg_smoothing(step_trajectory: np.ndarray, window_len: int = 7) -> np.ndarray:
    """
    Mô phỏng giải thuật MLPG giải phương trình ràng buộc Static + Dynamic Delta.
    Chuyển đổi chuỗi tham số bậc thang (step-wise) gián đoạn giữa các state
    thành quỹ đạo liên tục và trơn mượt (smooth continuous trajectory).
    """
    if len(step_trajectory) < window_len:
        return step_trajectory
    # Sử dụng bộ lọc Hanning đối xứng bảo toàn xu hướng cực đại hợp lý
    win = np.hanning(window_len)
    win = win / np.sum(win)
    smoothed = np.convolve(step_trajectory, win, mode="same")
    # Giữ biên không méo
    smoothed[:window_len // 2] = step_trajectory[:window_len // 2]
    smoothed[-window_len // 2:] = step_trajectory[-window_len // 2:]
    return smoothed

def generate_trajectories(tokens: list[str], scale_f0: float = 1.0, scale_dur: float = 1.0) -> dict:
    """
    Sinh chuỗi tham số âm học theo trạng thái HMM từ chuỗi từ đầu vào.
    Mỗi từ gồm 3 trạng thái: [Onset, Nucleus, Coda].
    """
    f0_steps = []
    f1_steps = []
    f2_steps = []
    f3_steps = []
    energy_steps = []
    state_seq = []
    
    current_state_idx = 0
    
    for word in tokens:
        prof = WORD_ACOUSTIC_PROFILES.get(word, DEFAULT_PROFILE)
        base_f0 = prof["f0"] * scale_f0
        base_dur = max(6, int(prof["dur_frames"] * scale_dur))
        
        # 3 state với biến thiên tham số nội tại:
        # State 1 (Onset): 25% duration, F0 tăng nhẹ, energy tăng
        d1 = int(base_dur * 0.25)
        # State 2 (Nucleus): 50% duration, F0 đỉnh, formants ổn định, energy cao
        d2 = int(base_dur * 0.50)
        # State 3 (Coda): 25% duration, F0 hạ nhẹ, energy giảm
        d3 = base_dur - d1 - d2
        
        # State 1
        f0_steps.extend([base_f0 * 0.96] * d1)
        f1_steps.extend([prof["f1"] * 0.90] * d1)
        f2_steps.extend([prof["f2"] * 0.95] * d1)
        f3_steps.extend([prof["f3"]] * d1)
        energy_steps.extend([0.05] * d1)
        state_seq.extend([current_state_idx] * d1)
        
        # State 2
        f0_steps.extend([base_f0 * 1.02] * d2)
        f1_steps.extend([prof["f1"]] * d2)
        f2_steps.extend([prof["f2"]] * d2)
        f3_steps.extend([prof["f3"]] * d2)
        energy_steps.extend([0.08] * d2)
        state_seq.extend([current_state_idx + 1] * d2)
        
        # State 3
        f0_steps.extend([base_f0 * 0.93] * d3)
        f1_steps.extend([prof["f1"] * 0.85] * d3)
        f2_steps.extend([prof["f2"] * 0.90] * d3)
        f3_steps.extend([prof["f3"]] * d3)
        energy_steps.extend([0.04] * d3)
        state_seq.extend([current_state_idx + 2] * d3)
        
        current_state_idx += 3
        
        # Thêm 2 frame khoảng lặng giữa các từ
        f0_steps.extend([base_f0 * 0.9] * 2)
        f1_steps.extend([prof["f1"] * 0.7] * 2)
        f2_steps.extend([prof["f2"] * 0.7] * 2)
        f3_steps.extend([prof["f3"]] * 2)
        energy_steps.extend([0.005] * 2)
        state_seq.extend([-1] * 2)

    f0_steps = np.array(f0_steps, dtype=np.float32)
    f1_steps = np.array(f1_steps, dtype=np.float32)
    f2_steps = np.array(f2_steps, dtype=np.float32)
    f3_steps = np.array(f3_steps, dtype=np.float32)
    energy_steps = np.array(energy_steps, dtype=np.float32)
    
    # Áp dụng MLPG Dynamic Constraint smoothing
    f0_smooth = mlpg_smoothing(f0_steps, window_len=9)
    f1_smooth = mlpg_smoothing(f1_steps, window_len=7)
    f2_smooth = mlpg_smoothing(f2_steps, window_len=7)
    energy_smooth = mlpg_smoothing(energy_steps, window_len=5)
    
    frame_shift_sec = 0.010  # 10ms frame shift
    time_axis = np.arange(len(f0_steps)) * frame_shift_sec
    
    return {
        "time": time_axis,
        "f0": f0_smooth,
        "f0_raw": f0_steps,
        "f1": f1_smooth,
        "f1_raw": f1_steps,
        "f2": f2_smooth,
        "f3": f3_steps,
        "energy": energy_smooth,
        "states": np.array(state_seq),
        "frame_shift_sec": frame_shift_sec
    }

def synthesize_source_filter_vocoder(tracks: dict, sr: int = 16000) -> np.ndarray:
    """
    Bộ tổng hợp Vocoder Source-Filter:
    Tạo xung kích thích theo F0 contour và lọc qua các cộng hưởng Formants F1, F2, F3.
    """
    total_sec = tracks["time"][-1] + tracks["frame_shift_sec"]
    total_samples = int(total_sec * sr)
    
    # 1. Tạo nguồn kích thích tuần hoàn (Pulse excitation) với tần số biến thiên theo F0 contour
    # Nội suy F0 contour lên từng mẫu âm thanh
    time_samples = np.linspace(0.0, total_sec, total_samples, endpoint=False)
    f0_interp = np.interp(time_samples, tracks["time"], tracks["f0"])
    energy_interp = np.interp(time_samples, tracks["time"], tracks["energy"])
    
    # Tích lũy pha tức thời: phi(t) = 2*pi * integral F0(t) dt
    phase = 2.0 * np.pi * np.cumsum(f0_interp) / sr
    
    # Excitation: Tổng các điều hòa (Harmonic glottal pulses)
    excitation = np.zeros(total_samples, dtype=np.float32)
    for harmonic in range(1, 12):
        excitation += (1.0 / harmonic) * np.sin(harmonic * phase)
        
    # Thêm thành phần nhiễu kích thích nhẹ (Aspiration noise)
    noise = np.random.normal(0, 0.05, total_samples).astype(np.float32)
    excitation = excitation + noise
    
    # Áp dụng phong bì năng lượng
    excitation = excitation * energy_interp
    
    # 2. Bộ lọc cộng hưởng Formant thanh đạo (Vocal Tract Resonator)
    # Lọc qua bộ lọc dải IIR bậc 2 tại tần số trung bình của F1 và F2
    mean_f1 = float(np.mean(tracks["f1"]))
    mean_f2 = float(np.mean(tracks["f2"]))
    
    # Resonator F1
    b1, a1 = signal.iirpeak(mean_f1, Q=6.0, fs=sr)
    y_f1 = signal.lfilter(b1, a1, excitation)
    
    # Resonator F2
    b2, a2 = signal.iirpeak(mean_f2, Q=7.0, fs=sr)
    y_f2 = signal.lfilter(b2, a2, excitation)
    
    # Tổng hợp các formant
    y = y_f1 * 0.7 + y_f2 * 0.3
    
    # Khử DC và chuẩn hóa
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y * (0.85 / peak)
        
    return y.astype(np.float32)

def synthesize_hmm(text: str, sr: int = 16000, scale_f0: float = 1.0, scale_dur: float = 1.0) -> tuple[np.ndarray, int, dict]:
    """
    API chính của Backend HMM TTS:
    Nhận văn bản -> Chuẩn hóa -> Sinh tham số HMM/MLPG -> Vocoder -> Waveform.
    """
    tokens = normalize_queue_text(text)
    tracks = generate_trajectories(tokens, scale_f0=scale_f0, scale_dur=scale_dur)
    y_hmm = synthesize_source_filter_vocoder(tracks, sr=sr)
    return y_hmm, sr, tracks

if __name__ == "__main__":
    import soundfile as sf
    test_text = "Mời khách hàng số 23 đến quầy 5"
    y, sr, tracks = synthesize_hmm(test_text)
    sf.write("audio_output/hmm_tts.wav", y, sr)
    print(f"=== ĐÃ TỔNG HỢP HMM TTS THÀNH CÔNG ===")
    print(f"Văn bản : '{test_text}'")
    print(f"Thời lượng audio: {len(y)/sr:.3f}s | Fs: {sr} Hz")
    print(f"Số frames tham số: {len(tracks['time'])} (Frame shift: 10ms)")
    print(f"F0 min/max: {np.min(tracks['f0']):.1f} Hz -> {np.max(tracks['f0']):.1f} Hz")
    print(f"F1 min/max: {np.min(tracks['f1']):.1f} Hz -> {np.max(tracks['f1']):.1f} Hz")

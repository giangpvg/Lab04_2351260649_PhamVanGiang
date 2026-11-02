#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_inventory.py
Tạo và chuẩn hóa kho đơn vị âm thanh (Audio Inventory) cho bài Lab 4.
- 16 đơn vị từ vựng miền hẹp: "thông báo xếp hàng"
- Mỗi đơn vị gồm 3 ứng viên (candidates) có biến thiên tự nhiên về cao độ và trường độ:
    + Candidate 01: Giọng chuẩn tự nhiên (baseline)
    + Candidate 02: Tốc độ nhanh hơn (+8%), cao độ nâng nhẹ (+0.8 semitone)
    + Candidate 03: Tốc độ chậm rãi (-8%), cao độ hạ nhẹ (-0.8 semitone)
- Tiền xử lý: Fs=16kHz, mono, cắt tỉa khoảng lặng chính xác (tight trim) với 5ms padding,
  chuẩn hóa RMS an toàn (target=0.08) và trích xuất đặc trưng thống kê.
- Xuất dữ liệu: lab4_data/units/*.wav và lab4_data/metadata.csv.
"""

import os
import io
import time
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from gtts import gTTS

WORDS_MAP = {
    "mời": "moi",
    "khách": "khach",
    "hàng": "hang",
    "số": "so",
    "không": "khong",
    "một": "mot",
    "hai": "hai",
    "ba": "ba",
    "bốn": "bon",
    "năm": "nam",
    "sáu": "sau",
    "bảy": "bay",
    "tám": "tam",
    "chín": "chin",
    "đến": "den",
    "quầy": "quay"
}

CANDIDATES = [
    {"suffix": "01", "rate_stretch": 1.0, "pitch_steps": 0.0},
    {"suffix": "02", "rate_stretch": 0.92, "pitch_steps": 0.8},
    {"suffix": "03", "rate_stretch": 1.08, "pitch_steps": -0.8}
]

SR = 16000
OUTPUT_DIR = "lab4_data/units"
METADATA_FILE = "lab4_data/metadata.csv"

def compute_rms(signal: np.ndarray) -> float:
    """Tính Root Mean Square (RMS) của tín hiệu."""
    return float(np.sqrt(np.mean(signal ** 2) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    """Chuẩn hóa mức năng lượng RMS theo mục tiêu an toàn tránh clipping."""
    current_rms = compute_rms(x)
    y = x * (target / current_rms)
    return np.clip(y, -0.95, 0.95)

def tight_trim(x: np.ndarray, sr: int = 16000, thresh: float = 0.005, pad_ms: float = 5.0) -> np.ndarray:
    """Cắt tỉa khoảng lặng đầu/cuối sát với biên âm thanh và áp dụng fade 3ms mượt."""
    nz = np.where(np.abs(x) > thresh)[0]
    if len(nz) == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, nz[0] - pad)
    end = min(len(x), nz[-1] + pad)
    trimmed = x[start:end].copy()
    
    # 3ms soft fade-in / fade-out để tránh click ở mẫu đầu/cuối
    fade_len = int(sr * 0.003)
    if len(trimmed) > 2 * fade_len:
        fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        trimmed[:fade_len] *= fade
        trimmed[-fade_len:] *= fade[::-1]
    return trimmed

def synthesize_base_word(word: str, max_retries: int = 3) -> np.ndarray:
    """Thu thập âm thanh từ đơn bằng Google TTS (vi) với Fs=16kHz, Mono."""
    for attempt in range(max_retries):
        try:
            fp = io.BytesIO()
            tts = gTTS(text=word, lang="vi", slow=False)
            tts.write_to_fp(fp)
            fp.seek(0)
            
            y, _ = librosa.load(fp, sr=SR, mono=True)
            y_trimmed = tight_trim(y, sr=SR)
            return y_trimmed
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Lỗi tạo từ '{word}': {e}")
            time.sleep(1.0)

def generate_candidate(base_y: np.ndarray, cand_cfg: dict) -> np.ndarray:
    """Tạo biến thể ứng viên với trường độ và cao độ khác biệt."""
    y = base_y.copy()
    if cand_cfg["rate_stretch"] != 1.0:
        y = librosa.effects.time_stretch(y, rate=cand_cfg["rate_stretch"])
    if cand_cfg["pitch_steps"] != 0.0:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=cand_cfg["pitch_steps"])
    y = tight_trim(y, sr=SR)
    y = match_rms(y, target=0.08)
    return y

def build_inventory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []
    
    print(f"=== Bắt đầu tạo Audio Inventory với Tight Trimming ({len(WORDS_MAP)} từ, 3 candidates/từ) ===")
    
    for word, file_prefix in WORDS_MAP.items():
        base_y = synthesize_base_word(word)
        
        for cand in CANDIDATES:
            suffix = cand["suffix"]
            filename = f"{file_prefix}_{suffix}.wav"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            y = generate_candidate(base_y, cand)
            sf.write(file_path, y, SR, subtype="PCM_16")
            
            duration = float(len(y) / SR)
            val_rms = compute_rms(y)
            
            metadata.append({
                "unit_id": f"{file_prefix}_{suffix}",
                "label": word,
                "candidate": int(suffix),
                "speaker": "Speaker_VN_16k",
                "duration": round(duration, 4),
                "rms": round(val_rms, 5),
                "samples": len(y),
                "file_path": file_path
            })
            print(f"  [+] Đã tạo: {filename:<14} | label: {word:<7} | dur: {duration:.3f}s | RMS: {val_rms:.4f}")
            
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_FILE, index=False, encoding="utf-8")
    print(f"\n[✓] Hoàn thành xuất sắc! Đã lưu metadata vào {METADATA_FILE} ({len(df)} units).")

if __name__ == "__main__":
    build_inventory()

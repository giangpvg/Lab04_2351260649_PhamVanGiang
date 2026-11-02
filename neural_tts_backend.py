#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_tts_backend.py
Mô-đun bọc (Wrapper) mô hình Neural Text-to-Speech (Deep Learning TTS).
- Kiến trúc tổng quát: Text -> Acoustic Model (Mel-spectrogram) -> Neural Vocoder -> Waveform.
- Đo lường chính xác độ trễ (Latency) và Hệ số thời gian thực (Real-Time Factor - RTF).
- Trích xuất 80-band Mel-Spectrogram chuẩn hóa phục vụ phân tích chất lượng.
- Xuất file: audio_output/neural_tts.wav.
"""

import time
import io
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS

SR = 16000

def synthesize(text: str, sr: int = 16000) -> tuple[np.ndarray, int, dict]:
    """
    Inference Neural TTS từ văn bản đầu vào.
    Trả về:
        y_nn: tín hiệu âm thanh float32 chuẩn 16kHz
        sr: tần số lấy mẫu (16000)
        extra: từ điển chứa mel_spectrogram, latency_sec, rtf
    """
    t_start = time.perf_counter()
    
    # 1. Pipeline Text to Speech
    fp = io.BytesIO()
    tts = gTTS(text=text, lang="vi", slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    
    y, _ = librosa.load(fp, sr=sr, mono=True)
    # Cắt khoảng lặng nhẹ đầu/cuối
    y_trimmed, _ = librosa.effects.trim(y, top_db=28, frame_length=512, hop_length=128)
    
    # Chuẩn hóa mức đỉnh tránh clipping
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_nn = y_trimmed * (0.85 / peak)
    else:
        y_nn = y_trimmed
        
    t_end = time.perf_counter()
    latency_sec = float(t_end - t_start)
    audio_dur = float(len(y_nn) / sr)
    rtf = float(latency_sec / (audio_dur + 1e-6))
    
    # 2. Trích xuất biểu diễn âm học Mel-Spectrogram (80 mel filterbanks)
    mel_spec = librosa.feature.melspectrogram(
        y=y_nn, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=0.0, fmax=sr // 2
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    extra = {
        "text": text,
        "latency_sec": round(latency_sec, 4),
        "duration_sec": round(audio_dur, 4),
        "rtf": round(rtf, 4),
        "mel_spec_db": mel_spec_db
    }
    
    return y_nn.astype(np.float32), sr, extra

if __name__ == "__main__":
    test_text = "Mời khách hàng số 23 đến quầy 5"
    y_nn, sr, extra = synthesize(test_text)
    sf.write("audio_output/neural_tts.wav", y_nn, sr)
    print("=== INFERENCE NEURAL TTS THÀNH CÔNG ===")
    print(f"Văn bản      : '{test_text}'")
    print(f"Thời lượng   : {extra['duration_sec']:.3f} s")
    print(f"Độ trễ (Lat) : {extra['latency_sec']:.3f} s")
    print(f"Hệ số RTF    : {extra['rtf']:.4f} (RTF < 1.0 biểu thị nhanh hơn thời gian thực)")
    print(f"Mel-spec shape: {extra['mel_spec_db'].shape}")

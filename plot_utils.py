#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_utils.py
Module trực quan hóa tín hiệu, đặc trưng và kết quả thực nghiệm (300 DPI).
Bao gồm:
1. plot_inventory_statistics: Histogram phân bố Duration và RMS.
2. plot_boundary_smoothing_comparison: Waveform và Spectrogram tại biên nối (Raw vs. Cross-fade).
3. plot_unit_selection_trellis: Biểu diễn lưới chi phí Unit Selection và Viterbi path.
4. plot_hmm_parameter_trajectories: Quỹ đạo F0, F1/F2 (Step-wise vs. MLPG smoothed).
5. plot_backend_comparison: Đối sánh Waveform và Spectrogram giữa Concat, HMM và Neural TTS.
6. plot_crossfade_experiment: So sánh các mức crossfade 0, 5, 15, 30 ms.
7. plot_prosody_modifications: Phân tích pitch shift và speed stretch.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

# Thiết lập style đồ họa khoa học, đẹp mắt
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.4
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_inventory_statistics(metadata_csv: str = "lab4_data/metadata.csv", save_path: str = "figures/fig1_inventory_stats.png"):
    """Vẽ histogram phân bố trường độ (Duration) và năng lượng (RMS) của Inventory."""
    df = pd.read_csv(metadata_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Histogram Duration
    axes[0].hist(df["duration"], bins=14, color="#1f77b4", edgecolor="black", alpha=0.75, rwidth=0.85)
    axes[0].axvline(df["duration"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['duration'].mean():.3f}s")
    axes[0].set_title("Phân bố Thời lượng Đơn vị (Duration)")
    axes[0].set_xlabel("Thời lượng (giây)")
    axes[0].set_ylabel("Số lượng mẫu (Units)")
    axes[0].grid(True, linestyle="--")
    axes[0].legend()
    
    # 2. Histogram RMS
    axes[1].hist(df["rms"], bins=10, color="#2ca02c", edgecolor="black", alpha=0.75, rwidth=0.85)
    axes[1].axvline(df["rms"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['rms'].mean():.4f}")
    axes[1].set_title("Phân bố Năng lượng RMS (Energy Consistency)")
    axes[1].set_xlabel("Mức năng lượng RMS")
    axes[1].set_ylabel("Số lượng mẫu (Units)")
    axes[1].grid(True, linestyle="--")
    axes[1].legend()
    
    plt.suptitle("Thống kê Đặc trưng Tập mẫu Âm thanh (Audio Inventory Stats)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_boundary_comparison(y_raw: np.ndarray, y_smooth: np.ndarray, boundaries: list[float], sr: int = 16000, save_path: str = "figures/fig2_spectrogram_boundaries.png"):
    """Vẽ Waveform và Spectrogram so sánh biên nối giữa ghép thô và làm mịn bằng Cross-fade."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    t_raw = np.linspace(0, len(y_raw) / sr, len(y_raw))
    t_smooth = np.linspace(0, len(y_smooth) / sr, len(y_smooth))
    
    # 1. Waveform Raw
    axes[0].plot(t_raw, y_raw, color="#d62728", alpha=0.8)
    axes[0].set_title("(a) Waveform Ghép Thô (Raw Concatenation - xfade=0ms)")
    axes[0].set_ylabel("Biên độ")
    axes[0].grid(True, linestyle=":")
    for b in boundaries:
        axes[0].axvline(b, color="black", linestyle="--", alpha=0.6)
        
    # 2. Spectrogram Raw
    D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw, n_fft=512, hop_length=128)), ref=np.max)
    img1 = librosa.display.specshow(D_raw, sr=sr, hop_length=128, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma")
    axes[1].set_title("(b) Spectrogram Ghép Thô (Xuất hiện bậc nhảy phổ và click tại biên)")
    axes[1].set_ylabel("Tần số (Hz)")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")
    for b in boundaries:
        axes[1].axvline(b, color="cyan", linestyle="--", alpha=0.7)

    # 3. Waveform Smooth
    axes[2].plot(t_smooth, y_smooth, color="#1f77b4", alpha=0.8)
    axes[2].set_title("(c) Waveform Sau Làm Mịn Biên (Cross-fade 15ms)")
    axes[2].set_ylabel("Biên độ")
    axes[2].grid(True, linestyle=":")
    for b in boundaries:
        axes[2].axvline(b, color="black", linestyle="--", alpha=0.6)

    # 4. Spectrogram Smooth
    D_smooth = librosa.amplitude_to_db(np.abs(librosa.stft(y_smooth, n_fft=512, hop_length=128)), ref=np.max)
    img2 = librosa.display.specshow(D_smooth, sr=sr, hop_length=128, x_axis="time", y_axis="linear", ax=axes[3], cmap="magma")
    axes[3].set_title("(d) Spectrogram Sau Làm Mịn (Vùng chuyển tiếp phổ êm ái, triệt tiêu xung biên)")
    axes[3].set_xlabel("Thời gian (giây)")
    axes[3].set_ylabel("Tần số (Hz)")
    fig.colorbar(img2, ax=axes[3], format="%+2.0f dB")
    for b in boundaries:
        axes[3].axvline(b, color="cyan", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_unit_selection_trellis(step_details: list[dict], save_path: str = "figures/fig3_unit_selection_cost.png"):
    """Vẽ biểu đồ chi phí từng bước của thuật toán Unit Selection."""
    steps = [d["step"] for d in step_details]
    labels = [d["file"] for d in step_details]
    c_target = [d["target_cost"] for d in step_details]
    c_join = [d["join_cost"] for d in step_details]
    c_total = [d["cum_cost"] for d in step_details]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(steps))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, c_join, width, label="Join Cost (D_spec + RMS jump)", color="#ff7f0e", alpha=0.85)
    rects2 = ax1.bar(x + width/2, c_target, width, label="Target Cost (Dur/RMS dev)", color="#2ca02c", alpha=0.85)
    
    ax1.set_xlabel("Đơn vị trong câu thông báo")
    ax1.set_ylabel("Chi phí từng bước (Step Cost)", color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Trục phụ biểu diễn tổng chi phí tích lũy
    ax2 = ax1.twinx()
    ax2.plot(x, c_total, color="#d62728", marker="o", linewidth=2.5, label="Cumulative Total Cost")
    ax2.set_ylabel("Tổng chi phí tích lũy (Viterbi Cost)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    
    # Kết hợp legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.title("Phân Tích Chi Phí Lựa Chọn Đơn Vị (Unit Selection Cost Breakdown)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_hmm_parameter_trajectories(tracks: dict, save_path: str = "figures/fig4_hmm_trajectories.png"):
    """Vẽ quỹ đạo tham số HMM: So sánh Step-wise thô (không delta) và MLPG trơn mượt (có delta)."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    t = tracks["time"]
    
    # 1. F0 Contour
    axes[0].plot(t, tracks["f0_raw"], color="gray", linestyle="--", alpha=0.7, label="Step-wise (Không ràng buộc Delta)")
    axes[0].plot(t, tracks["f0"], color="#d62728", linewidth=2.2, label="MLPG Trajectory (Có ràng buộc Dynamic Delta)")
    axes[0].set_title("(a) Quỹ đạo Tần số cơ bản F0 (Pitch Contour)")
    axes[0].set_ylabel("F0 (Hz)")
    axes[0].grid(True, linestyle=":")
    axes[0].legend(loc="upper right")
    
    # 2. Formant Trajectories (F1 & F2)
    axes[1].plot(t, tracks["f1_raw"], color="lightgray", linestyle="--")
    axes[1].plot(t, tracks["f1"], color="#1f77b4", linewidth=2.0, label="Formant F1 (Âm sắc - Chiều cao nguyên âm)")
    axes[1].plot(t, tracks["f2"], color="#2ca02c", linewidth=2.0, label="Formant F2 (Độ lùi của lưỡi)")
    axes[1].set_title("(b) Quỹ đạo Formant F1 và F2 theo thời gian")
    axes[1].set_ylabel("Tần số (Hz)")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(loc="upper right")
    
    # 3. Energy Trajectory & State Alignment
    axes[2].plot(t, tracks["energy"], color="#9467bd", linewidth=2.0, label="Energy Envelope")
    axes[2].set_title("(c) Phong bì Năng lượng (Energy Trajectory)")
    axes[2].set_xlabel("Thời gian (giây)")
    axes[2].set_ylabel("Biên độ")
    axes[2].grid(True, linestyle=":")
    axes[2].legend(loc="upper right")
    
    plt.suptitle("Khảo sát Phát sinh Tham số Âm học HMM (HMM-based MLPG Synthesis)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_backend_comparison(y_concat: np.ndarray, y_hmm: np.ndarray, y_neural: np.ndarray, sr: int = 16000, save_path: str = "figures/fig5_backend_comparison.png"):
    """Đối sánh Waveform và Spectrogram giữa 3 phương pháp TTS: Concat, HMM và Neural."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    signals = [
        ("Concatenative TTS (Best Unit Selection)", y_concat, "#1f77b4"),
        ("Statistical Parametric (HMM + Vocoder)", y_hmm, "#2ca02c"),
        ("Neural TTS (Deep Learning Pretrained)", y_neural, "#d62728")
    ]
    
    for i, (name, sig, color) in enumerate(signals):
        # Waveform
        t = np.linspace(0, len(sig) / sr, len(sig))
        axes[i, 0].plot(t, sig, color=color, alpha=0.85)
        axes[i, 0].set_title(f"Waveform: {name}")
        axes[i, 0].set_ylabel("Biên độ")
        axes[i, 0].grid(True, linestyle=":")
        if i == 2:
            axes[i, 0].set_xlabel("Thời gian (giây)")
            
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=128)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=128, x_axis="time", y_axis="linear", ax=axes[i, 1], cmap="viridis")
        axes[i, 1].set_title(f"Spectrogram: {name}")
        axes[i, 1].set_ylabel("Tần số (Hz)")
        if i == 2:
            axes[i, 1].set_xlabel("Thời gian (giây)")
            
    plt.suptitle("So Sánh Toàn Diện 3 Thế Hệ Công Nghệ TTS (Concat vs. HMM vs. Neural)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_crossfade_experiment(xfade_dict: dict[int, np.ndarray], sr: int = 16000, save_path: str = "figures/fig6_crossfade_analysis.png"):
    """Vẽ phóng to biên nối tại các mức crossfade 0, 5, 15, 30 ms."""
    fig, axes = plt.subplots(len(xfade_dict), 1, figsize=(12, 10), sharex=True)
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    
    for idx, (xfade_ms, audio) in enumerate(xfade_dict.items()):
        # Trích xuất 100ms quanh biên nối đầu tiên (vùng ~0.5s)
        sample_mid = int(0.56 * sr)
        half_win = int(0.08 * sr)
        snippet = audio[sample_mid - half_win : sample_mid + half_win]
        t = np.linspace(-80, 80, len(snippet))
        
        axes[idx].plot(t, snippet, color=colors[idx], label=f"Cross-fade = {xfade_ms} ms")
        axes[idx].set_title(f"Biên nối tại xfade = {xfade_ms} ms {'(Gốc - Click gián đoạn)' if xfade_ms==0 else '(Tối ưu chuẩn)' if xfade_ms==15 else '(Quá dài - nhòe phụ âm)' if xfade_ms==30 else ''}")
        axes[idx].set_ylabel("Biên độ")
        axes[idx].grid(True, linestyle=":")
        axes[idx].legend(loc="upper right")
        
    axes[-1].set_xlabel("Thời gian tương đối quanh biên (ms)")
    plt.suptitle("Thực Nghiệm 1: Khảo Sát Tác Động Của Thời Lượng Cross-fade Tại Biên Ghép", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

def plot_prosody_modifications(speed_dict: dict, pitch_dict: dict, sr: int = 16000, save_path: str = "figures/fig7_prosody_modifications.png"):
    """Trực quan hóa biến đổi tốc độ và cao độ (Prosody Modification)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Speed Waveforms
    for speed, y in speed_dict.items():
        t = np.linspace(0, len(y) / sr, len(y))
        axes[0, 0].plot(t, y, label=f"Speed {speed}x ({len(y)/sr:.2f}s)", alpha=0.75)
    axes[0, 0].set_title("(a) Biến đổi Tốc độ nói (Speed Stretch 0.8x, 1.0x, 1.2x)")
    axes[0, 0].set_xlabel("Thời gian (giây)")
    axes[0, 0].set_ylabel("Biên độ")
    axes[0, 0].grid(True, linestyle=":")
    axes[0, 0].legend()
    
    # 2. Speed Spectrogram (1.2x)
    D_fast = librosa.amplitude_to_db(np.abs(librosa.stft(speed_dict[1.2], n_fft=512, hop_length=128)), ref=np.max)
    librosa.display.specshow(D_fast, sr=sr, hop_length=128, x_axis="time", y_axis="linear", ax=axes[0, 1], cmap="inferno")
    axes[0, 1].set_title("(b) Spectrogram ở tốc độ 1.2x (Bảo toàn Formants)")
    axes[0, 1].set_ylabel("Tần số (Hz)")
    
    # 3. Pitch Waveforms
    for ds, y in pitch_dict.items():
        t = np.linspace(0, len(y) / sr, len(y))
        axes[1, 0].plot(t[:800], y[:800], label=f"Δs = {ds:+d} st", alpha=0.75)
    axes[1, 0].set_title("(c) Dịch cao độ Pitch Shift (Phóng to chu kỳ sóng 50ms)")
    axes[1, 0].set_xlabel("Thời gian (giây)")
    axes[1, 0].set_ylabel("Biên độ")
    axes[1, 0].grid(True, linestyle=":")
    axes[1, 0].legend()
    
    # 4. Pitch Spectrogram (+2 st)
    D_pitch = librosa.amplitude_to_db(np.abs(librosa.stft(pitch_dict[2], n_fft=512, hop_length=128)), ref=np.max)
    librosa.display.specshow(D_pitch, sr=sr, hop_length=128, x_axis="time", y_axis="linear", ax=axes[1, 1], cmap="inferno")
    axes[1, 1].set_title("(d) Spectrogram khi dịch cao độ +2 semitone")
    axes[1, 1].set_xlabel("Thời gian (giây)")
    axes[1, 1].set_ylabel("Tần số (Hz)")
    
    plt.suptitle("Thực Nghiệm 3 & 4: Điều Khiển Ngữ Điệu & Mô Hình Source–Filter", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [✓] Đã lưu: {save_path}")

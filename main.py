#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
Chương trình thực thi giao diện dòng lệnh (CLI) cho Hệ thống Tổng hợp Tiếng nói (TTS) - Lab 4.
Sinh viên: Phạm Văn Giang — MSSV: 2351260649
Môn học: CSE457 - Xử lý âm thanh và tiếng nói

Cách sử dụng:
  python3 main.py --all                                        # Chạy toàn bộ pipeline và 5 thực nghiệm
  python3 main.py --text "Mời khách hàng số 23 đến quầy 5"     # Tổng hợp chuỗi văn bản
  python3 main.py --file input.txt --backend neural            # Tổng hợp trực tiếp từ tệp văn bản .txt
  python3 main.py --file input.txt --backend all               # Chạy đối sánh cả 3 backend từ file .txt
"""

import argparse
import os
import sys
import soundfile as sf

from normalize import normalize_queue_text
from inventory import SpeechInventory
from concat_tts import concat_units
from unit_selection import UnitSelector
from hmm_tts_backend import synthesize_hmm
from neural_tts_backend import synthesize as synthesize_neural
from experiments import run_all_experiments

SR = 16000
OUTPUT_DIR = "audio_output"

def synthesize_text_cli(text: str, backend: str = "concat", xfade_ms: float = 15.0, output_name: str | None = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text = text.strip()
    if not text:
        print("[-] Cảnh báo: Nội dung văn bản rỗng!")
        return

    print("=" * 70)
    print("HỆ THỐNG TỔNG HỢP TIẾNG NÓI CSE457 - LAB 4")
    print(f"Văn bản đầu vào : '{text}'")
    print(f"Backend lựa chọn: {backend.upper()}")
    print("=" * 70)
    
    tokens = normalize_queue_text(text)
    print(f"[1] Text Normalization -> Tokens: {tokens}")
    
    prefix = output_name if output_name else "cli"

    # 1. Concatenative Backend
    if backend in ["concat", "all"]:
        try:
            inv = SpeechInventory()
            selector = UnitSelector(sr=SR, lambda_join=1.0)
            lattice = inv.get_candidate_lattice(tokens)
            best_paths, total_cost, details = selector.select_best_units(lattice)
            y_concat, bounds = concat_units(best_paths, sr=SR, xfade_ms=xfade_ms)
            out_concat = os.path.join(OUTPUT_DIR, f"{prefix}_concat.wav")
            sf.write(out_concat, y_concat, SR)
            print(f"[✓] Concatenative TTS (Best Unit Selection):")
            print(f"    - File xuất: {out_concat} ({len(y_concat)/SR:.2f}s)")
            print(f"    - Tổng chi phí Viterbi: {total_cost:.4f}")
        except Exception as e:
            print(f"[-] Concatenative TTS không khả dụng cho văn bản này: {e}")
            print("    (Lưu ý: Concatenative chỉ hỗ trợ từ vựng miền hẹp của Lab 4).")

    # 2. HMM Backend
    if backend in ["hmm", "all"]:
        try:
            y_hmm, sr, tracks = synthesize_hmm(text, sr=SR)
            out_hmm = os.path.join(OUTPUT_DIR, f"{prefix}_hmm.wav")
            sf.write(out_hmm, y_hmm, sr)
            print(f"[✓] HMM-based Parametric TTS:")
            print(f"    - File xuất: {out_hmm} ({len(y_hmm)/sr:.2f}s)")
            print(f"    - Số frame tham số: {len(tracks['time'])}")
        except Exception as e:
            print(f"[-] HMM TTS gặp lỗi: {e}")

    # 3. Neural TTS Backend (Hỗ trợ tiếng Việt tổng quát mọi văn bản)
    if backend in ["neural", "all"]:
        try:
            y_nn, sr, extra = synthesize_neural(text, sr=SR)
            out_nn = os.path.join(OUTPUT_DIR, f"{prefix}_neural.wav")
            sf.write(out_nn, y_nn, sr)
            print(f"[✓] Neural TTS (Deep Learning - Hỗ trợ mọi văn bản tiếng Việt):")
            print(f"    - File xuất: {out_nn} ({len(y_nn)/sr:.2f}s)")
            print(f"    - Độ trễ (Latency): {extra['latency_sec']*1000:.1f} ms | RTF: {extra['rtf']:.4f}")
        except Exception as e:
            print(f"[-] Neural TTS gặp lỗi: {e}")

    print("=" * 70)
    print(f"Tổng hợp hoàn tất! Âm thanh được lưu tại thư mục: {OUTPUT_DIR}/")

def main():
    parser = argparse.ArgumentParser(description="Chương trình CLI Tổng hợp tiếng nói CSE457 - Lab 4")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ 5 thực nghiệm và xuất tất cả biểu đồ/audio")
    parser.add_argument("--text", type=str, default=None, help="Câu văn bản cần tổng hợp")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn đến tệp văn bản (.txt) cần tổng hợp")
    parser.add_argument("--backend", type=str, choices=["concat", "hmm", "neural", "all"], default="neural", help="Phương pháp tổng hợp (mặc định: neural)")
    parser.add_argument("--xfade", type=float, default=15.0, help="Thời lượng cross-fade (ms) cho Concat TTS")
    parser.add_argument("--output", type=str, default=None, help="Tên tiền tố cho file âm thanh đầu ra")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    elif args.file:
        if not os.path.exists(args.file):
            print(f"[-] Lỗi: Không tìm thấy tệp '{args.file}'!")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        out_prefix = args.output if args.output else f"output_{base_name}"
        synthesize_text_cli(text=content, backend=args.backend, xfade_ms=args.xfade, output_name=out_prefix)
    elif args.text:
        synthesize_text_cli(text=args.text, backend=args.backend, xfade_ms=args.xfade, output_name=args.output)
    else:
        # Mặc định chạy câu case study
        default_sentence = "Mời khách hàng số 23 đến quầy 5"
        synthesize_text_cli(text=default_sentence, backend="all", xfade_ms=args.xfade)

if __name__ == "__main__":
    main()

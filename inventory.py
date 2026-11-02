#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventory.py
Quản lý và thống kê kho đơn vị âm thanh (Audio Inventory Manager).
- Tải metadata từ CSV.
- Cung cấp danh sách đường dẫn ứng viên (candidate paths) cho từng từ.
- Tạo lưới ứng viên (Candidate Lattice) cho chuỗi token đầu vào.
- Tính toán thống kê và kiểm tra bất thường dữ liệu (Duration, RMS).
"""

import os
import pandas as pd
import numpy as np

class SpeechInventory:
    """Lớp quản lý kho đơn vị âm thanh phục vụ Unit Selection và Concatenation."""
    def __init__(self, metadata_path: str = "lab4_data/metadata.csv"):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {metadata_path}")
        self.metadata_path = metadata_path
        self.df = pd.read_csv(metadata_path)
        
        # Nhóm ứng viên theo nhãn từ vựng
        self.units_map: dict[str, list[dict]] = {}
        for _, row in self.df.iterrows():
            lbl = str(row["label"])
            if lbl not in self.units_map:
                self.units_map[lbl] = []
            self.units_map[lbl].append(row.to_dict())

    def get_candidate_paths(self, word: str) -> list[str]:
        """Trả về danh sách đường dẫn file âm thanh của tất cả candidate của một từ."""
        if word not in self.units_map:
            raise KeyError(f"Từ vựng '{word}' không tồn tại trong Inventory!")
        return [c["file_path"] for c in self.units_map[word]]

    def get_candidate_lattice(self, tokens: list[str]) -> list[list[str]]:
        """
        Tạo lưới candidate (Lattice) cho chuỗi token đầu vào.
        Trả về danh sách các cột, mỗi cột là danh sách đường dẫn candidate tại vị trí đó.
        """
        lattice = []
        for word in tokens:
            lattice.append(self.get_candidate_paths(word))
        return lattice

    def get_statistics(self) -> dict:
        """Thống kê mô tả về trường độ (duration) và năng lượng (RMS)."""
        dur = self.df["duration"]
        rms_val = self.df["rms"]
        return {
            "total_units": len(self.df),
            "vocab_size": self.df["label"].nunique(),
            "duration": {
                "mean": float(dur.mean()),
                "std": float(dur.std()),
                "min": float(dur.min()),
                "max": float(dur.max())
            },
            "rms": {
                "mean": float(rms_val.mean()),
                "std": float(rms_val.std()),
                "min": float(rms_val.min()),
                "max": float(rms_val.max())
            }
        }

if __name__ == "__main__":
    inv = SpeechInventory()
    stats = inv.get_statistics()
    print("=== THỐNG KÊ INVENTORY ===")
    print(f"Tổng số units: {stats['total_units']} | Từ vựng: {stats['vocab_size']}")
    print(f"Duration (s) : {stats['duration']['mean']:.3f} ± {stats['duration']['std']:.3f} "
          f"[{stats['duration']['min']:.3f} -> {stats['duration']['max']:.3f}]")
    print(f"RMS          : {stats['rms']['mean']:.4f} ± {stats['rms']['std']:.4f} "
          f"[{stats['rms']['min']:.4f} -> {stats['rms']['max']:.4f}]")

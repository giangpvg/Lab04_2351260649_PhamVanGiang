#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize.py
Module chuẩn hóa văn bản (Text Normalization) cho hệ thống TTS miền hẹp.
- Case study: Hệ thống thông báo gọi số thứ tự xếp hàng tại quầy.
- Ánh xạ số nguyên/mã số thành chuỗi từ phát âm (Digit by digit reading).
- Làm sạch ký tự đặc biệt, dấu câu, chuyển về chữ thường chuẩn hóa.
"""

import re

DIGIT_MAP = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín"
}

PUNCTUATION_REGEX = re.compile(r"[.,!?;:\"'()\[\]{}—–\-_]")

def clean_text(text: str) -> str:
    """Làm sạch dấu câu và ký tự đặc biệt, chuyển về chữ thường."""
    text = text.lower()
    text = PUNCTUATION_REGEX.sub(" ", text)
    # Rút gọn khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_queue_text(text: str) -> list[str]:
    """
    Chuẩn hóa văn bản miền hẹp thông báo quầy.
    Mọi chuỗi số được phân tách và đọc từng chữ số.
    Ví dụ: 'Mời khách hàng số 23 đến quầy 5.' ->
           ['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']
    """
    cleaned = clean_text(text)
    tokens = []
    for token in cleaned.split():
        if token.isdigit():
            # Đọc từng chữ số trong mã số quầy/số thứ tự
            for char in token:
                if char in DIGIT_MAP:
                    tokens.append(DIGIT_MAP[char])
        else:
            tokens.append(token)
    return tokens

if __name__ == "__main__":
    test_cases = [
        "Mời khách hàng số 23 đến quầy 5.",
        "Mời khách hàng số 104 đến quầy 2!",
        "Mời khách hàng số 79 đến quầy 8..."
    ]
    print("=== KIỂM THỬ MODULE TEXT NORMALIZATION ===")
    for tc in test_cases:
        res = normalize_queue_text(tc)
        print(f"Input : '{tc}'")
        print(f"Output: {res}\n")

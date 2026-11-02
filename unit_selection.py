#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unit_selection.py
Thuật toán lựa chọn đơn vị âm thanh tối ưu (Unit Selection Engine).
- Tính Target Cost (đo sai lệch duration và RMS so với mục tiêu ngữ âm).
- Tính Join Cost (đo mức độ bất liên tục phổ log-magnitude D_spec qua biên 25ms, Hanning window).
- Tìm kiếm đường đi chi phí nhỏ nhất (Minimum Cost Path) qua lưới Candidate bằng Dynamic Programming (Viterbi).
- Xuất chuỗi file âm thanh tối ưu cho: best_concat.wav.
"""

import numpy as np
import soundfile as sf
from concat_tts import load_unit, rms

def compute_join_cost(a: np.ndarray, b: np.ndarray, sr: int = 16000, ms: float = 25.0, nfft: int = 512) -> float:
    """
    Tính Join Cost đo mức độ sai khác phổ log-magnitude tại vùng biên ghép nối.
    D_spec = (1/K) * sum_k [ L_left[k] - L_right[k] ]^2
    """
    n = int(sr * ms / 1000.0)
    n = min(n, len(a), len(b))
    if n < 8:
        return 0.0
        
    wa = np.hanning(n)
    wb = np.hanning(n)
    
    A = np.fft.rfft(a[-n:] * wa, n=nfft)
    B = np.fft.rfft(b[:n] * wb, n=nfft)
    
    La = 20.0 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20.0 * np.log10(np.maximum(np.abs(B), 1e-8))
    
    d_spec = float(np.mean((La - Lb) ** 2))
    
    # Bổ sung sai khác RMS ở biên để phạt bước nhảy năng lượng
    rms_a = rms(a[-n:])
    rms_b = rms(b[:n])
    d_rms = 50.0 * abs(rms_a - rms_b)
    
    return d_spec + d_rms

def compute_target_cost(unit_audio: np.ndarray, target_dur: float, target_rms: float = 0.08, sr: int = 16000) -> float:
    """
    Tính Target Cost đo độ sai lệch về trường độ và năng lượng mong muốn.
    """
    dur = len(unit_audio) / sr
    cur_rms = rms(unit_audio)
    
    # Trọng số độ lệch duration và RMS
    cost_dur = ((dur - target_dur) / (target_dur + 1e-6)) ** 2
    cost_rms = ((cur_rms - target_rms) / (target_rms + 1e-6)) ** 2
    return float(1.5 * cost_dur + 1.0 * cost_rms)

class UnitSelector:
    """Bộ giải thuật tìm kiếm chuỗi đơn vị tối ưu qua ma trận quy hoạch động (Viterbi)."""
    def __init__(self, sr: int = 16000, lambda_join: float = 1.0):
        self.sr = sr
        self.lambda_join = lambda_join
        self._audio_cache: dict[str, np.ndarray] = {}

    def get_audio(self, path: str) -> np.ndarray:
        if path not in self._audio_cache:
            self._audio_cache[path] = load_unit(path, self.sr)
        return self._audio_cache[path]

    def select_best_units(self, candidate_lattice: list[list[str]], target_durs: list[float] | None = None) -> tuple[list[str], float, list[dict]]:
        """
        Tìm kiếm chuỗi ứng viên có tổng chi phí nhỏ nhất (Target Cost + Join Cost).
        lattice: Danh sách các vị trí j (từ 0..J-1), mỗi vị trí là danh sách đường dẫn file candidates.
        Trả về:
            best_paths: danh sách đường dẫn candidate tối ưu
            total_cost: tổng chi phí nhỏ nhất
            step_details: chi tiết cost từng bước để lập bảng phân tích
        """
        J = len(candidate_lattice)
        if J == 0:
            return [], 0.0, []

        # Khởi tạo target duration mặc định nếu không truyền vào (lấy trung bình các candidate tại vị trí đó)
        if target_durs is None:
            target_durs = []
            for j in range(J):
                durs = [len(self.get_audio(p)) / self.sr for p in candidate_lattice[j]]
                target_durs.append(float(np.mean(durs)))

        # DP tables: Viterbi cost và Backpointer
        V = []  # V[j][c] = min cost to reach candidate c at step j
        BP = [] # BP[j][c] = index of candidate at step j-1 that gave min cost

        # Bước khởi tạo (j = 0)
        first_costs = []
        for c, p in enumerate(candidate_lattice[0]):
            audio = self.get_audio(p)
            c_target = compute_target_cost(audio, target_durs[0], sr=self.sr)
            first_costs.append(c_target)
        V.append(first_costs)
        BP.append([-1] * len(candidate_lattice[0]))

        # Các bước tiếp theo (j = 1 .. J-1)
        for j in range(1, J):
            curr_candidates = candidate_lattice[j]
            prev_candidates = candidate_lattice[j - 1]
            curr_costs = []
            curr_bp = []

            for c_curr, p_curr in enumerate(curr_candidates):
                audio_curr = self.get_audio(p_curr)
                c_target = compute_target_cost(audio_curr, target_durs[j], sr=self.sr)

                best_prev_cost = float("inf")
                best_prev_idx = -1

                for c_prev, p_prev in enumerate(prev_candidates):
                    audio_prev = self.get_audio(p_prev)
                    c_join = compute_join_cost(audio_prev, audio_curr, sr=self.sr)
                    
                    total_step_cost = V[j - 1][c_prev] + self.lambda_join * c_join + c_target
                    if total_step_cost < best_prev_cost:
                        best_prev_cost = total_step_cost
                        best_prev_idx = c_prev

                curr_costs.append(best_prev_cost)
                curr_bp.append(best_prev_idx)

            V.append(curr_costs)
            BP.append(curr_bp)

        # Truy vết ngược (Backtracking)
        best_last_idx = int(np.argmin(V[-1]))
        min_total_cost = float(V[-1][best_last_idx])

        best_path_indices = [best_last_idx]
        for j in range(J - 1, 0, -1):
            prev_idx = BP[j][best_path_indices[-1]]
            best_path_indices.append(prev_idx)
        best_path_indices.reverse()

        best_paths = [candidate_lattice[j][idx] for j, idx in enumerate(best_path_indices)]

        # Ghi nhận chi tiết từng bước nối
        step_details = []
        for j in range(J):
            p = best_paths[j]
            audio = self.get_audio(p)
            c_tar = compute_target_cost(audio, target_durs[j], sr=self.sr)
            c_jn = 0.0
            if j > 0:
                prev_audio = self.get_audio(best_paths[j - 1])
                c_jn = compute_join_cost(prev_audio, audio, sr=self.sr)
            step_details.append({
                "step": j + 1,
                "file": p.split("/")[-1],
                "target_cost": round(c_tar, 4),
                "join_cost": round(c_jn, 4),
                "cum_cost": round(V[j][best_path_indices[j]], 4)
            })

        return best_paths, min_total_cost, step_details

if __name__ == "__main__":
    from inventory import SpeechInventory
    from normalize import normalize_queue_text
    
    inv = SpeechInventory()
    tokens = normalize_queue_text("Mời khách hàng số 23 đến quầy 5.")
    lattice = inv.get_candidate_lattice(tokens)
    
    selector = UnitSelector()
    best_paths, total_c, details = selector.select_best_units(lattice)
    
    print(f"=== KẾT QUẢ UNIT SELECTION (VITERBI) ===")
    print(f"Tổng chi phí tối ưu: {total_c:.4f}\n")
    for d in details:
        print(f"Step {d['step']:2d} | Unit: {d['file']:<14} | C_target: {d['target_cost']:.4f} | C_join: {d['join_cost']:.4f} | Total: {d['cum_cost']:.4f}")

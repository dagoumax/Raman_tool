#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raman Spectroscopy Gas Concentration Analyzer

SIF/TXT -> Baseline -> Peak -> Concentration -> Smooth -> Plot
Dependencies: numpy, scipy, matplotlib, tkinter (Python 3.10+)
"""

import os, re, struct, json, csv, threading, queue, time, traceback
from datetime import datetime
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import matplotlib
try:    matplotlib.use('TkAgg', force=True)
except TypeError: matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import MaxNLocator, FuncFormatter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

# ============================================================
class SifFile:
    def __init__(self, filepath):
        self.filepath = filepath; self.filename = os.path.basename(filepath)
        self.data = open(filepath, 'rb').read(); self.laser_wl = 532.0
        self.spectrometer = ''; self.ccd_model = ''; self.calib_coeffs = []
        self.num_pixels = 0; self.ccd_height = 0
        self.pixels = []; self.wavelengths = []; self.raman_shifts = []
        self._parse_header(); self._extract_data(); self._compute_axes()
    def _parse_header(self):
        h = self.data[:3000].decode('latin-1', errors='replace')
        m = re.search(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+e[+-]\d+)\s+(\d+)', h)
        if m: self.calib_coeffs = [float(m.group(i)) for i in range(1, 5)]
        m = re.search(r'(\d{3})\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*\d+\s*\n\s*Raman shift', h)
        if m: self.laser_wl = float(m.group(1))
        m = re.search(r'SR\d+[A-Za-z]*\d*', h)
        if m: self.spectrometer = m.group()
        m = re.search(r'DR\d+[A-Za-z]*[-\w]*', h)
        if m: self.ccd_model = m.group()
        m = re.search(r'\b(\d{3,4})\s+(\d{2,4})\s+(\d+)', h)
        if m: self.num_pixels = int(m.group(1)); self.ccd_height = int(m.group(2))
    def _extract_data(self):
        bo = 0; bc = 0
        for so in range(0, len(self.data) - 4, 4):
            vals = [struct.unpack('<f', self.data[so + i:so + i + 4])[0] for i in range(0, min(4000, len(self.data) - so), 4) if so + i + 4 <= len(self.data)]
            if len(vals) >= self.num_pixels: vals = vals[:self.num_pixels]
            else: continue
            nz = sum(1 for v in vals if v > 1.0)
            if nz < 50: continue
            avg = sum(vals) / len(vals)
            if 10 < avg < 100000 and nz > bc: bo = so; bc = nz
        if bo == 0 and self.num_pixels:
            xp = self.data.find(b'<?xml')
            if xp > 0: bo = xp - self.num_pixels * 4
        self.pixels = [struct.unpack('<f', self.data[bo + i * 4:bo + (i + 1) * 4])[0] for i in range(self.num_pixels)]
        if len(self.calib_coeffs) < 4: self.calib_coeffs = [0, 1, 0, 0]
    def _compute_axes(self):
        c0, c1, c2, c3 = self.calib_coeffs
        self.wavelengths = []; self.raman_shifts = []
        for p in range(len(self.pixels)):
            wl = c0 + c1 * p + c2 * p**2 + c3 * p**3
            self.wavelengths.append(wl)
            self.raman_shifts.append((1.0 / self.laser_wl - 1.0 / wl) * 1e7 if wl > 0 else 0.0)

# ============================================================
def arPLS(y, lam=1e5, max_iter=50, tol=1e-6):
    y = np.asarray(y, dtype=np.float64).flatten(); N = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format='csc', dtype=np.float64)
    DTD = D.T @ D; w = np.ones(N, dtype=np.float64)
    for _ in range(max_iter):
        W = diags(w, 0, shape=(N, N), format='csc'); A = W + lam * DTD; b = w * y; z = spsolve(A, b)
        d = y - z; idx = d < 0
        if np.any(idx): dn = d[idx]; m = np.mean(dn); s = np.std(dn); s = max(s, np.finfo(np.float64).eps)
        else: m, s = 0.0, 1.0
        arg = 2 * (d - (-m + 2 * s)) / s; arg = np.clip(arg, -60, 60)
        w_new = np.where(d >= 0, 1.0 / (1.0 + np.exp(arg)), 1.0)
        nw = np.linalg.norm(w); nw = max(nw, np.finfo(np.float64).eps)
        if np.linalg.norm(w_new - w) / nw < tol:
            w = w_new; W = diags(w, 0, shape=(N, N), format='csc'); z = spsolve(W + lam * DTD, w * y); break
        w = w_new
    return z

# ============================================================
DEFAULT_CONFIG = {
    "input_format": "sif", "strategy": "peak_area",
    "coeff_a": 1.0, "coeff_b": 1.0, "coeff_c": 1.0,
    "o2_center": 1555.0, "o2_half_width": 25.0,
    "n2_center": 2330.0, "n2_half_width": 20.0,
    "co2_center": 1388.0, "co2_half_width": 15.0,
    "alpha": 0.02, "window_size": 35, "w1": 0.15, "w2": 0.80, "w3": 0.05,
    "adapt_threshold": 3.0, "playback_delay_ms": 0,
}

_PARAMS_SMOOTH = {"alpha": 0.015, "window": 40, "w1": 0.15, "w2": 0.80, "w3": 0.05}
_PARAMS_TRACK  = {"alpha": 0.50,  "window": 3,  "w1": 0.10, "w2": 0.10, "w3": 0.80}
_PARAMS_SLOW_ALPHA = 0.05

# ============================================================
def parse_sif_to_spectrum(filepath):
    sif = SifFile(filepath)
    return np.array(sif.raman_shifts, dtype=np.float64), np.array(sif.pixels, dtype=np.float64)

def parse_txt_to_spectrum(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    rs = data[:, 1].astype(np.float64); corr = data[:, 3].astype(np.float64)
    return rs, np.maximum(corr, 0.0)

def baseline_correct(intensities, lam=1e5, max_iter=50, tol=1e-6):
    baseline = arPLS(intensities, lam=lam, max_iter=max_iter, tol=tol)
    return np.maximum(intensities - baseline, 0.0), baseline

def find_peak_indices(raman_shifts, center, half_width):
    lo, hi = center - half_width, center + half_width
    return np.where((raman_shifts >= lo) & (raman_shifts <= hi))[0]

def calc_peak_intensity_area(corrected, raman_shifts, indices):
    if len(indices) < 3: return 0.0
    x_win, y_win = raman_shifts[indices], corrected[indices]
    au = np.trapz(y_win, x_win)
    at = (corrected[indices[0]] + corrected[indices[-1]]) * (raman_shifts[indices[-1]] - raman_shifts[indices[0]]) / 2.0
    return max(au - at, 0.0)

def calc_peak_intensity_max(corrected, indices):
    if len(indices) < 3: return 0.0
    y_win = corrected[indices]; mv = np.max(y_win)
    ba = (corrected[indices[0]] + corrected[indices[-1]]) / 2.0
    return max(mv - ba, 0.0)

def calc_concentration(I_O2, I_N2, I_CO2, a, b, c):
    denom = I_O2 * a + I_N2 * b + I_CO2 * c
    if denom < 1e-15: return 0.0, 0.0, 0.0
    return (I_O2 * a) / denom, (I_N2 * b) / denom, (I_CO2 * c) / denom

def temporal_smooth(raw_values, alpha, window_size, w1, w2, w3, adapt_threshold=3.0):
    n = len(raw_values)
    if n == 0: return []
    if n == 1: return [float(raw_values[0])]
    if adapt_threshold <= 0:
        tw = w1 + w2 + w3
        if tw > 0: w1, w2, w3 = w1 / tw, w2 / tw, w3 / tw
        smoothed, ema = [], float(raw_values[0])
        for i in range(n):
            raw = float(raw_values[i])
            ema = raw if i == 0 else alpha * raw + (1.0 - alpha) * ema
            st = max(0, i - window_size + 1); ma = float(np.mean(raw_values[st:i + 1]))
            smoothed.append(w1 * ema + w2 * ma + w3 * raw)
        return smoothed
    smoothed, ema = [], float(raw_values[0]); ema_slow = float(raw_values[0]); k_prev = 0.0
    for i in range(n):
        raw = float(raw_values[i])
        if i == 0: ema_slow = raw
        else: ema_slow = _PARAMS_SLOW_ALPHA * raw + (1.0 - _PARAMS_SLOW_ALPHA) * ema_slow
        st_start = max(0, i - 30); rst = float(np.std(raw_values[st_start:i + 1], ddof=1)) if i - st_start >= 1 else 1e-8
        S = abs(raw - ema_slow) / max(rst, 1e-8)
        raw_k = 1.0 / (1.0 + np.exp(-3.0 * (S - adapt_threshold)))
        if raw_k > k_prev: k = 0.50 * raw_k + 0.50 * k_prev
        else: k = 0.05 * raw_k + 0.95 * k_prev
        k_prev = k
        av = _PARAMS_SMOOTH["alpha"] * (1 - k) + _PARAMS_TRACK["alpha"] * k
        wn = max(2, round(_PARAMS_SMOOTH["window"] * (1 - k) + _PARAMS_TRACK["window"] * k))
        w1v = _PARAMS_SMOOTH["w1"] * (1 - k) + _PARAMS_TRACK["w1"] * k
        w2v = _PARAMS_SMOOTH["w2"] * (1 - k) + _PARAMS_TRACK["w2"] * k
        w3v = _PARAMS_SMOOTH["w3"] * (1 - k) + _PARAMS_TRACK["w3"] * k
        tw = w1v + w2v + w3v
        if tw > 0: w1v, w2v, w3v = w1v / tw, w2v / tw, w3v / tw
        ema = raw if i == 0 else av * raw + (1.0 - av) * ema
        mst = max(0, i - wn + 1); ma = float(np.mean(raw_values[mst:i + 1]))
        smoothed.append(w1v * ema + w2v * ma + w3v * raw)
    return smoothed

class IncrementalSmoother:
    def __init__(self):
        self.ema = 0.0; self.ema_slow = 0.0; self.k_prev = 0.0
        self.history = []; self.smoothed = []; self.emas = []; self.mas = []; self.w_raws = []
    def add(self, raw, alpha, window_size, w1, w2, w3, adapt_threshold=3.0):
        self.history.append(raw); n = len(self.history)
        if n == 1:
            self.ema = raw; self.ema_slow = raw; self.smoothed.append(raw)
            self.emas.append(raw); self.mas.append(raw); self.w_raws.append(1.0)
            return raw
        self.ema_slow = _PARAMS_SLOW_ALPHA * raw + (1.0 - _PARAMS_SLOW_ALPHA) * self.ema_slow
        if adapt_threshold > 0:
            raw_k = self._compute_k(raw, adapt_threshold)
            if raw_k > self.k_prev: k = 0.50 * raw_k + 0.50 * self.k_prev
            else: k = 0.05 * raw_k + 0.95 * self.k_prev
            self.k_prev = k
            av = _PARAMS_SMOOTH["alpha"] * (1 - k) + _PARAMS_TRACK["alpha"] * k
            wn = max(2, round(_PARAMS_SMOOTH["window"] * (1 - k) + _PARAMS_TRACK["window"] * k))
            w1v = _PARAMS_SMOOTH["w1"] * (1 - k) + _PARAMS_TRACK["w1"] * k
            w2v = _PARAMS_SMOOTH["w2"] * (1 - k) + _PARAMS_TRACK["w2"] * k
            w3v = _PARAMS_SMOOTH["w3"] * (1 - k) + _PARAMS_TRACK["w3"] * k
        else:
            av, wn = alpha, max(2, window_size); w1v, w2v, w3v = w1, w2, w3
        tw = w1v + w2v + w3v
        if tw > 0: w1v, w2v, w3v = w1v / tw, w2v / tw, w3v / tw
        self.ema = av * raw + (1.0 - av) * self.ema
        mst = max(0, n - wn); ma = float(np.mean(self.history[mst:n]))
        val = w1v * self.ema + w2v * ma + w3v * raw
        self.smoothed.append(val); self.emas.append(self.ema); self.mas.append(ma); self.w_raws.append(w3v)
        return val
    def _compute_k(self, raw, adapt_threshold):
        rw = self.history[-30:]; rst = float(np.std(rw, ddof=1)) if len(rw) >= 2 else 1e-8
        S = abs(raw - self.ema_slow) / max(rst, 1e-8)
        return 1.0 / (1.0 + np.exp(-3.0 * (S - adapt_threshold)))

# ============================================================
def process_single_file(filepath, config):
    fmt = config.get("input_format", "sif")
    if fmt == "txt": raman_shifts, corrected = parse_txt_to_spectrum(filepath)
    else: raman_shifts, intensities = parse_sif_to_spectrum(filepath); corrected, _ = baseline_correct(intensities)
    idx_o2 = find_peak_indices(raman_shifts, config["o2_center"], config["o2_half_width"])
    idx_n2 = find_peak_indices(raman_shifts, config["n2_center"], config["n2_half_width"])
    idx_co2 = find_peak_indices(raman_shifts, config["co2_center"], config["co2_half_width"])
    if config["strategy"] == "peak_area":
        I_O2 = calc_peak_intensity_area(corrected, raman_shifts, idx_o2)
        I_N2 = calc_peak_intensity_area(corrected, raman_shifts, idx_n2)
        I_CO2 = calc_peak_intensity_area(corrected, raman_shifts, idx_co2)
    else:
        I_O2 = calc_peak_intensity_max(corrected, idx_o2)
        I_N2 = calc_peak_intensity_max(corrected, idx_n2)
        I_CO2 = calc_peak_intensity_max(corrected, idx_co2)
    return calc_concentration(I_O2, I_N2, I_CO2, config["coeff_a"], config["coeff_b"], config["coeff_c"])

# ============================================================
class ProcessingThread(threading.Thread):
    def __init__(self, file_list, config, result_queue, stop_event):
        super().__init__(daemon=True)
        self.file_list = file_list; self.config = config.copy()
        self.result_queue = result_queue; self.stop_event = stop_event
        self.progress = 0.0; self.status = ""
        self.delay_ms = config.get("playback_delay_ms", 0) / 1000.0
        self._lock = threading.Lock()
    def run(self):
        total = len(self.file_list); self._update_status("")
        for i, filepath in enumerate(self.file_list):
            if self.stop_event.is_set(): self.result_queue.put(("stopped", None)); return
            try:
                C_O2, C_N2, C_CO2 = process_single_file(filepath, self.config)
                self.result_queue.put(("result", (C_O2, C_N2, C_CO2, filepath)))
                if self.delay_ms > 0: time.sleep(self.delay_ms)
            except Exception as e:
                self.result_queue.put(("warning", f"{os.path.basename(filepath)}: {e}")); continue
            with self._lock: self.progress = (i + 1) / total * 100.0
        with self._lock: self.progress = 100.0
        self.result_queue.put(("done", None))
    def _update_status(self, msg):
        with self._lock: self.status = msg
    def get_progress(self):
        with self._lock: return self.progress, self.status

# ============================================================
class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, **kwargs)
        self._is_open = tk.BooleanVar(value=False)
        bf = ttk.Frame(self); bf.pack(fill=tk.X)
        self._btn = ttk.Button(bf, text=f"\u25b6 {text}", command=self._toggle)
        self._btn.pack(side=tk.LEFT, padx=2, pady=2)
        self._content = ttk.Frame(self, relief=tk.GROOVE, borderwidth=1)
    def _toggle(self):
        if self._is_open.get():
            self._content.pack_forget(); self._is_open.set(False)
            t = self._btn.cget("text"); self._btn.config(text=t.replace("\u25bc", "\u25b6"))
        else:
            self._content.pack(fill=tk.X, padx=5, pady=2); self._is_open.set(True)
            t = self._btn.cget("text"); self._btn.config(text=t.replace("\u25b6", "\u25bc"))
    @property
    def content(self): return self._content

# ============================================================
class RamanApp:
    def __init__(self, root):
        self.root = root; self.root.title("Raman Gas Analyzer v1.0")
        self.root.geometry("1200x720"); self.root.minsize(1000, 600)
        self.config = DEFAULT_CONFIG.copy()
        self.raw_o2_list = []; self.raw_n2_list = []; self.raw_co2_list = []
        self.o2_smoother = IncrementalSmoother(); self.n2_smoother = IncrementalSmoother(); self.co2_smoother = IncrementalSmoother()
        self.smoothed_o2_list = []; self.smoothed_n2_list = []; self.smoothed_co2_list = []
        self.file_labels = []; self.processed_count = 0
        self._stop_event = threading.Event(); self._worker = None; self._result_queue = queue.Queue()
        self._label_fmt = lambda val, _: self.file_labels[int(val)-1] if self.file_labels and 0 <= int(val)-1 < len(self.file_labels) else str(int(val))
        self._user_zoomed = False
        self._build_ui(); self._poll_queue()

    def _build_ui(self):
        mp = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL); mp.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        lo = ttk.Frame(mp, width=400); mp.add(lo, weight=0)
        lc = tk.Canvas(lo, width=390, highlightthickness=0)
        sv = ttk.Scrollbar(lo, orient=tk.VERTICAL, command=lc.yview)
        sh = ttk.Scrollbar(lo, orient=tk.HORIZONTAL, command=lc.xview)
        lsf = ttk.Frame(lc)
        def _oc(e): lc.configure(scrollregion=lc.bbox("all"))
        lsf.bind("<Configure>", _oc)
        cw = lc.create_window((0, 0), window=lsf, anchor="nw")
        lc.configure(yscrollcommand=sv.set, xscrollcommand=sh.set)
        lc.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sh.pack(side=tk.BOTTOM, fill=tk.X); sv.pack(side=tk.RIGHT, fill=tk.Y)
        def _mw(e): lc.yview_scroll(int(-1 * (e.delta / 120)), "units")
        lc.bind("<Enter>", lambda e: lc.bind_all("<MouseWheel>", _mw))
        lc.bind("<Leave>", lambda e: lc.unbind_all("<MouseWheel>"))
        self._build_control_panel(lsf)
        rf = ttk.Frame(mp); mp.add(rf, weight=1)
        self._build_chart_panel(rf)
        bf = ttk.Frame(self.root); bf.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.progress_bar = ttk.Progressbar(bf, mode="determinate"); self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.status_var = tk.StringVar(value="")
        ttk.Label(bf, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _build_control_panel(self, parent):
        po = {"padx": 5, "pady": 3}
        pf = ttk.LabelFrame(parent, text="", padding=5); pf.pack(fill=tk.X, **po)
        fr = ttk.Frame(pf); fr.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(fr, text="").pack(side=tk.LEFT)
        self.fmt_var = tk.StringVar(value="sif")
        ttk.Combobox(fr, textvariable=self.fmt_var, values=["sif", "txt"], state="readonly", width=4).pack(side=tk.LEFT, padx=5)
        ttk.Label(fr, text="  sif=raw  |  txt=corrected").pack(side=tk.LEFT)
        ttk.Label(pf, text="Input dir:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.input_path_var = tk.StringVar()
        ttk.Entry(pf, textvariable=self.input_path_var, width=35).grid(row=2, column=0, sticky=tk.EW, pady=1)
        ttk.Button(pf, text="Browse...", command=self._browse_input).grid(row=2, column=1, padx=2)
        ttk.Label(pf, text="Output dir:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.output_path_var = tk.StringVar()
        ttk.Entry(pf, textvariable=self.output_path_var, width=35).grid(row=4, column=0, sticky=tk.EW, pady=1)
        ttk.Button(pf, text="Browse...", command=self._browse_output).grid(row=4, column=1, padx=2)
        pf.columnconfigure(0, weight=1)

        bf = ttk.Frame(parent); bf.pack(fill=tk.X, **po)
        self.start_btn = ttk.Button(bf, text="Batch", command=self._start_batch); self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(bf, text="Stop", command=self._stop_processing, state=tk.DISABLED); self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.single_btn = ttk.Button(bf, text="File", command=self._open_single_file); self.single_btn.pack(side=tk.LEFT, padx=2)
        self.stats_btn = ttk.Button(bf, text="Stats", command=self._show_stats_popup); self.stats_btn.pack(side=tk.LEFT, padx=2)
        sf = ttk.Frame(parent); sf.pack(fill=tk.X, **po)
        self.save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text="Save results", variable=self.save_var).pack(anchor=tk.W)
        zf = ttk.LabelFrame(parent, text="", padding=3); zf.pack(fill=tk.X, **po)
        zr = ttk.Frame(zf); zr.pack(fill=tk.X)
        ttk.Button(zr, text="X+", width=4, command=lambda: self._zoom(0.95, 1.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(zr, text="X-", width=4, command=lambda: self._zoom(1.05, 1.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(zr, text="Y+", width=4, command=lambda: self._zoom(1.0, 0.95)).pack(side=tk.LEFT, padx=1)
        ttk.Button(zr, text="Y-", width=4, command=lambda: self._zoom(1.0, 1.05)).pack(side=tk.LEFT, padx=1)
        ff = ttk.LabelFrame(parent, text="", padding=3); ff.pack(fill=tk.X, **po)
        self.fluc_o2_var = tk.StringVar(value="O2: --")
        self.fluc_n2_var = tk.StringVar(value="N2: --")
        self.fluc_co2_var = tk.StringVar(value="CO2: --")
        ttk.Label(ff, textvariable=self.fluc_o2_var, foreground='red', font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        ttk.Label(ff, textvariable=self.fluc_n2_var, foreground='blue', font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        ttk.Label(ff, textvariable=self.fluc_co2_var, foreground='green', font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        self.param_frame = CollapsibleFrame(parent, text=""); self.param_frame.pack(fill=tk.X, **po)
        cf = self.param_frame.content
        rw = ttk.Frame(cf); rw.pack(fill=tk.X, pady=2)
        ttk.Label(rw, text="").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value=self.config["strategy"])
        self._sd = {"\u5cf0\u9762\u79ef": "peak_area", "\u6700\u9ad8\u5f3a\u5ea6": "peak_max"}
        self._sr = {"peak_area": "\u5cf0\u9762\u79ef", "peak_max": "\u6700\u9ad8\u5f3a\u5ea6"}
        sc = ttk.Combobox(rw, textvariable=self.strategy_var, values=list(self._sd.keys()), state="readonly", width=10)
        sc.set(self._sr.get(self.config["strategy"], "")); sc.pack(side=tk.LEFT, padx=5)
        rw2 = ttk.Frame(cf); rw2.pack(fill=tk.X, pady=2)
        ttk.Label(rw2, text="").pack(side=tk.LEFT)
        self.coeff_a_var = tk.StringVar(value=str(self.config["coeff_a"])); ttk.Entry(rw2, textvariable=self.coeff_a_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(rw2, text="  ").pack(side=tk.LEFT)
        self.coeff_b_var = tk.StringVar(value=str(self.config["coeff_b"])); ttk.Entry(rw2, textvariable=self.coeff_b_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(rw2, text="  ").pack(side=tk.LEFT)
        self.coeff_c_var = tk.StringVar(value=str(self.config["coeff_c"])); ttk.Entry(rw2, textvariable=self.coeff_c_var, width=8).pack(side=tk.LEFT, padx=2)
        pkf = ttk.LabelFrame(cf, text="", padding=3); pkf.pack(fill=tk.X, pady=3)
        gd = ttk.Frame(pkf); gd.pack(fill=tk.X)
        ttk.Label(gd, text="").grid(row=0, column=0); ttk.Label(gd, text="center(1/cm)").grid(row=0, column=1, padx=5); ttk.Label(gd, text="hw(1/cm)").grid(row=0, column=2, padx=5)
        ttk.Label(gd, text="O2:").grid(row=1, column=0, sticky=tk.W)
        self.o2_center_var = tk.StringVar(value=str(self.config["o2_center"])); ttk.Entry(gd, textvariable=self.o2_center_var, width=10).grid(row=1, column=1, padx=5)
        self.o2_hw_var = tk.StringVar(value=str(self.config["o2_half_width"])); ttk.Entry(gd, textvariable=self.o2_hw_var, width=10).grid(row=1, column=2, padx=5)
        ttk.Label(gd, text="N2:").grid(row=2, column=0, sticky=tk.W)
        self.n2_center_var = tk.StringVar(value=str(self.config["n2_center"])); ttk.Entry(gd, textvariable=self.n2_center_var, width=10).grid(row=2, column=1, padx=5)
        self.n2_hw_var = tk.StringVar(value=str(self.config["n2_half_width"])); ttk.Entry(gd, textvariable=self.n2_hw_var, width=10).grid(row=2, column=2, padx=5)
        ttk.Label(gd, text="CO2:").grid(row=3, column=0, sticky=tk.W)
        self.co2_center_var = tk.StringVar(value=str(self.config["co2_center"])); ttk.Entry(gd, textvariable=self.co2_center_var, width=10).grid(row=3, column=1, padx=5)
        self.co2_hw_var = tk.StringVar(value=str(self.config["co2_half_width"])); ttk.Entry(gd, textvariable=self.co2_hw_var, width=10).grid(row=3, column=2, padx=5)
        smf = ttk.LabelFrame(cf, text="", padding=3); smf.pack(fill=tk.X, pady=3)
        s0 = ttk.Frame(smf); s0.pack(fill=tk.X, pady=1)
        ttk.Label(s0, text="Alpha(EMA):").pack(side=tk.LEFT)
        self.alpha_var = tk.StringVar(value=str(self.config["alpha"])); ttk.Entry(s0, textvariable=self.alpha_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(s0, text="  N:").pack(side=tk.LEFT)
        self.window_var = tk.StringVar(value=str(self.config["window_size"])); ttk.Entry(s0, textvariable=self.window_var, width=8).pack(side=tk.LEFT, padx=2)
        s1 = ttk.Frame(smf); s1.pack(fill=tk.X, pady=1)
        ttk.Label(s1, text="w1(EMA):").pack(side=tk.LEFT)
        self.w1_var = tk.StringVar(value=str(self.config["w1"])); ttk.Entry(s1, textvariable=self.w1_var, width=6).pack(side=tk.LEFT)
        ttk.Label(s1, text="  w2(MA):").pack(side=tk.LEFT, padx=(8,0))
        self.w2_var = tk.StringVar(value=str(self.config["w2"])); ttk.Entry(s1, textvariable=self.w2_var, width=6).pack(side=tk.LEFT)
        ttk.Label(s1, text="  w3(Raw):").pack(side=tk.LEFT, padx=(8,0))
        self.w3_var = tk.StringVar(value=str(self.config["w3"])); ttk.Entry(s1, textvariable=self.w3_var, width=6).pack(side=tk.LEFT)
        s2 = ttk.Frame(smf); s2.pack(fill=tk.X, pady=1)
        ttk.Label(s2, text="Thr(0=off):").pack(side=tk.LEFT)
        self.adapt_thr_var = tk.StringVar(value=str(self.config["adapt_threshold"])); ttk.Entry(s2, textvariable=self.adapt_thr_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(s2, text="").pack(side=tk.LEFT, padx=5)
        s3 = ttk.Frame(smf); s3.pack(fill=tk.X, pady=1)
        ttk.Label(s3, text="Delay(ms):").pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value=str(self.config["playback_delay_ms"])); ttk.Entry(s3, textvariable=self.delay_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(s3, text="0=full").pack(side=tk.LEFT, padx=5)
        cbf = ttk.Frame(cf); cbf.pack(fill=tk.X, pady=5)
        ttk.Button(cbf, text="Save Cfg", command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(cbf, text="Load Cfg", command=self._load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(cbf, text="Reset", command=self._reset_config).pack(side=tk.LEFT, padx=2)

    def _build_chart_panel(self, parent):
        self.notebook = ttk.Notebook(parent); self.notebook.pack(fill=tk.BOTH, expand=True)
        self._charts = {}
        ta = ttk.Frame(self.notebook); self.notebook.add(ta, text="")
        ch = self._make_chart_tab(ta, "", gases="all"); self._charts["all"] = ch; ta._chart_key = "all"
        for gas, c, lbl, ref in [("O2","#d62728","O2",0.21),("N2","#1f77b4","N2",0.78),("CO2","#2ca02c","CO2",None)]:
            tb = ttk.Frame(self.notebook); self.notebook.add(tb, text=gas)
            ch = self._make_chart_tab(tb, f"{lbl} Raw vs Smooth", gases=gas, color=c, ref=ref); self._charts[gas] = ch; tb._chart_key = gas
        self.fig = self._charts["all"]["fig"]; self.ax = self._charts["all"]["ax"]
        self.line_o2 = self._charts["all"]["line_o2"]; self.line_n2 = self._charts["all"]["line_n2"]; self.line_co2 = self._charts["all"]["line_co2"]
        self.raw_o2_scatter = self._charts["all"]["raw_scatters"]["O2"]; self.raw_n2_scatter = self._charts["all"]["raw_scatters"]["N2"]
        self.canvas = self._charts["all"]["canvas"]

    def _make_chart_tab(self, parent, title, gases, color=None, ref=None):
        fig = Figure(figsize=(5, 3.5), dpi=100); ax = fig.add_subplot(111)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_ylim(-0.02, 1.02); ax.set_title(title); ax.grid(True, alpha=0.3)
        if ref is not None: ax.axhline(y=ref, color='gray', linestyle='--', alpha=0.5, label=f'{gases} ref({ref})')
        if gases == "all":
            lo = ax.plot([],[],'r-o',markersize=4,linewidth=1.2,label='O2 smooth')[0]
            ln = ax.plot([],[],'b-s',markersize=4,linewidth=1.2,label='N2 smooth')[0]
            lc_ = ax.plot([],[],'g-^',markersize=4,linewidth=1.2,label='CO2 smooth')[0]
            rs = {"O2":ax.plot([],[],'.',color='lightcoral',markersize=2,alpha=0.5,label='O2 raw')[0],
                  "N2":ax.plot([],[],'.',color='lightblue',markersize=2,alpha=0.5,label='N2 raw')[0]}
        else:
            lo = ln = lc_ = None; rs = {}
            sk = {"O2":("r-o","red","O2 smooth"),"N2":("b-s","blue","N2 smooth"),"CO2":("g-^","green","CO2 smooth")}
            fmt, c, lbl = sk[gases]
            lo = ax.plot([],[],fmt,color=c,markersize=4,linewidth=1.2,label=lbl)[0]
            rc = {"O2":"salmon","N2":"steelblue","CO2":"mediumseagreen"}
            rs[gases] = ax.plot([],[],'.',color=rc[gases],markersize=4,alpha=0.8,label=f'{gases} raw')[0]
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=50, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(self._label_fmt))
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent); toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return {"fig":fig,"ax":ax,"canvas":canvas,"toolbar":toolbar,"line_o2":lo,"line_n2":ln,"line_co2":lc_,"raw_scatters":rs,"gases":gases}

    def _browse_input(self):
        p = filedialog.askdirectory(title="")
        if p: self.input_path_var.set(p)
    def _browse_output(self):
        p = filedialog.askdirectory(title="")
        if p: self.output_path_var.set(p)
    @staticmethod
    def _float_or(v, d):
        try: return float(v)
        except (ValueError, TypeError): return d

    def _collect_config(self):
        rs = self.strategy_var.get(); st = self._sd.get(rs, rs)
        return {"input_format":self.fmt_var.get(),"strategy":st,
            "coeff_a":self._float_or(self.coeff_a_var.get(),1.0),"coeff_b":self._float_or(self.coeff_b_var.get(),1.0),"coeff_c":self._float_or(self.coeff_c_var.get(),1.0),
            "o2_center":self._float_or(self.o2_center_var.get(),1555.0),"o2_half_width":self._float_or(self.o2_hw_var.get(),25.0),
            "n2_center":self._float_or(self.n2_center_var.get(),2330.0),"n2_half_width":self._float_or(self.n2_hw_var.get(),20.0),
            "co2_center":self._float_or(self.co2_center_var.get(),1388.0),"co2_half_width":self._float_or(self.co2_hw_var.get(),15.0),
            "alpha":self._float_or(self.alpha_var.get(),0.02),"window_size":max(1,int(self._float_or(self.window_var.get(),35))),
            "w1":self._float_or(self.w1_var.get(),0.15),"w2":self._float_or(self.w2_var.get(),0.80),"w3":self._float_or(self.w3_var.get(),0.05),
            "adapt_threshold":self._float_or(self.adapt_thr_var.get(),3.0),
            "playback_delay_ms":max(0,int(self._float_or(self.delay_var.get(),0)))}

    def _apply_config(self, config):
        self.fmt_var.set(config.get("input_format","sif"))
        s = config.get("strategy","peak_area"); self.strategy_var.set(self._sr.get(s,s))
        self.coeff_a_var.set(str(config.get("coeff_a",1.0))); self.coeff_b_var.set(str(config.get("coeff_b",1.0))); self.coeff_c_var.set(str(config.get("coeff_c",1.0)))
        self.o2_center_var.set(str(config.get("o2_center",1555.0))); self.o2_hw_var.set(str(config.get("o2_half_width",25.0)))
        self.n2_center_var.set(str(config.get("n2_center",2330.0))); self.n2_hw_var.set(str(config.get("n2_half_width",20.0)))
        self.co2_center_var.set(str(config.get("co2_center",1388.0))); self.co2_hw_var.set(str(config.get("co2_half_width",15.0)))
        self.alpha_var.set(str(config.get("alpha",0.02))); self.window_var.set(str(config.get("window_size",35)))
        self.w1_var.set(str(config.get("w1",0.15))); self.w2_var.set(str(config.get("w2",0.80))); self.w3_var.set(str(config.get("w3",0.05)))
        self.adapt_thr_var.set(str(config.get("adapt_threshold",3.0))); self.delay_var.set(str(config.get("playback_delay_ms",0)))

    def _save_config(self):
        p = filedialog.asksaveasfilename(title="", defaultextension=".json", filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p,"w",encoding="utf-8") as f: json.dump(self._collect_config(),f,indent=2,ensure_ascii=False)
            self._set_status(f"")
        except Exception as e: messagebox.showerror("",str(e))
    def _load_config(self):
        p = filedialog.askopenfilename(title="", filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p,"r",encoding="utf-8") as f: self._apply_config(json.load(f)); self._set_status("")
        except Exception as e: messagebox.showerror("",str(e))
    def _reset_config(self):
        self._apply_config(DEFAULT_CONFIG); self._set_status("")

    def _set_status(self, msg): self.status_var.set(msg)
    def _set_buttons_state(self, r):
        if r: self.start_btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL); self.single_btn.config(state=tk.DISABLED)
        else: self.start_btn.config(state=tk.NORMAL); self.stop_btn.config(state=tk.DISABLED); self.single_btn.config(state=tk.NORMAL)

    def _start_batch(self):
        idir = self.input_path_var.get().strip()
        if not idir: messagebox.showwarning("",""); return
        if not os.path.isdir(idir): messagebox.showerror("",""); return
        try: config = self._collect_config()
        except Exception as e: messagebox.showerror("",str(e)); return
        ext = "." + config.get("input_format","sif")
        fl = sorted([os.path.join(idir,f) for f in os.listdir(idir) if f.lower().endswith(ext)])
        if not fl: messagebox.showwarning("",""); return
        self.config = config; self.raw_o2_list.clear(); self.raw_n2_list.clear(); self.raw_co2_list.clear()
        self.o2_smoother=IncrementalSmoother(); self.n2_smoother=IncrementalSmoother(); self.co2_smoother=IncrementalSmoother()
        self.smoothed_o2_list.clear(); self.smoothed_n2_list.clear(); self.smoothed_co2_list.clear()
        self.file_labels.clear(); self.processed_count = 0
        self._stop_event.clear(); self._result_queue = queue.Queue(); self._user_zoomed = False
        self._set_buttons_state(True); self.progress_bar["value"] = 0
        self._set_status("")
        self._worker = ProcessingThread(fl, config, self._result_queue, self._stop_event); self._worker.start()

    def _stop_processing(self): self._stop_event.set(); self._set_status("")

    def _poll_queue(self):
        try: self._poll_queue_inner()
        except Exception: self._set_status(f"") ; self._on_processing_done()
        finally: self.root.after(100, self._poll_queue)

    def _poll_queue_inner(self):
        try:
            batch = 0
            while True:
                mt, pl = self._result_queue.get_nowait()
                if mt == "result":
                    C_O2, C_N2, C_CO2, fp = pl
                    self.raw_o2_list.append(max(0.0,min(1.0,C_O2))); self.raw_n2_list.append(max(0.0,min(1.0,C_N2))); self.raw_co2_list.append(max(0.0,min(1.0,C_CO2)))
                    self.processed_count += 1
                    self.file_labels.append(self._parse_time_label(fp, self.processed_count))
                    batch += 1
                    if batch >= 8: break
                elif mt == "error": self._set_status(f""); self._on_processing_done(); break
                elif mt == "stopped": self._update_smoothing_and_plot(); self._set_status(""); self._on_processing_done(); self._maybe_save_results(); break
                elif mt == "done": self._update_smoothing_and_plot(); self._set_status(""); self._on_processing_done(); self._maybe_save_results(); break
                elif mt == "warning": self._set_status(f"")
        except queue.Empty: pass
        if batch > 0: self._update_smoothing_and_plot()
        if self._worker and self._worker.is_alive():
            pg, st = self._worker.get_progress(); self.progress_bar["value"] = pg
            if st: self._set_status(st)

    def _on_processing_done(self):
        self._set_buttons_state(False); self.progress_bar["value"] = 100; self._worker = None
        if self.processed_count > 0: self._update_fluctuation_status()

    @staticmethod
    def _compute_stats(data):
        if len(data) < 2: return {"std":0.0,"cv":0.0,"range":0.0,"mean":float(data[0]) if data else 0.0}
        arr = np.array(data, dtype=np.float64); m = float(np.mean(arr)); s = float(np.std(arr, ddof=1))
        return {"std":s,"cv":s/m if m>1e-12 else 0.0,"range":float(np.max(arr)-np.min(arr)),"mean":m}

    def _compute_fluctuation(self):
        ro2=self._compute_stats(self.raw_o2_list); rn2=self._compute_stats(self.raw_n2_list)
        so2=self._compute_stats(self.smoothed_o2_list); sn2=self._compute_stats(self.smoothed_n2_list)
        rc = self._compute_stats(self.raw_co2_list); sc = self._compute_stats(self.smoothed_co2_list)
        def _r(rs,ss): return 0.0 if rs<1e-15 else (rs-ss)/rs*100.0
        return {"raw_o2":ro2,"raw_n2":rn2,"smo_o2":so2,"smo_n2":sn2,
                "o2_std_red":_r(ro2["std"],so2["std"]),"n2_std_red":_r(rn2["std"],sn2["std"]),
                "raw_co2":rc,"smo_co2":sc,"co2_std_red":_r(rc["std"],sc["std"])}

    def _update_fluctuation_status(self):
        f = self._compute_fluctuation()
        self._set_status(f"O2 std {f['raw_o2']['std']:.4f}->{f['smo_o2']['std']:.4f} ({f['o2_std_red']:+.1f}%) | N2 std {f['raw_n2']['std']:.4f}->{f['smo_n2']['std']:.4f} ({f['n2_std_red']:+.1f}%) | CO2 std {f['raw_co2']['std']:.4f}->{f['smo_co2']['std']:.4f} ({f['co2_std_red']:+.1f}%)")

    def _show_stats_popup(self):
        if self.processed_count < 2: messagebox.showinfo("",""); return
        f = self._compute_fluctuation()
        lines = ["","","            raw                 smooth           reduce","           mean   std  CV     mean   std  CV"]
        for gas, raw, smo, red in [("O2",f["raw_o2"],f["smo_o2"],f["o2_std_red"]),("N2",f["raw_n2"],f["smo_n2"],f["n2_std_red"]),("CO2",f["raw_co2"],f["smo_co2"],f["co2_std_red"])]:
            lines.append(f"  {gas}    {raw['mean']:.4f}  {raw['std']:.4f}  {raw['cv']:.3f}    {smo['mean']:.4f}  {smo['std']:.4f}  {smo['cv']:.3f}    {red:+.1f}%")
        lines += [f"  O2 range: raw [{f['raw_o2']['range']:.4f}] ({min(self.raw_o2_list):.4f}~{max(self.raw_o2_list):.4f})  smooth [{f['smo_o2']['range']:.4f}] ({min(self.smoothed_o2_list):.4f}~{max(self.smoothed_o2_list):.4f})",
                  f"  N2 range: raw [{f['raw_n2']['range']:.4f}] ({min(self.raw_n2_list):.4f}~{max(self.raw_n2_list):.4f})  smooth [{f['smo_n2']['range']:.4f}] ({min(self.smoothed_n2_list):.4f}~{max(self.smoothed_n2_list):.4f})",
                  f"  CO2 range: raw [{f['raw_co2']['range']:.4f}] ({min(self.raw_co2_list):.4f}~{max(self.raw_co2_list):.4f})  smooth [{f['smo_co2']['range']:.4f}] ({min(self.smoothed_co2_list):.4f}~{max(self.smoothed_co2_list):.4f})",
                  f"", f"  points: {self.processed_count}"]
        messagebox.showinfo("", "\n".join(lines))

    def _zoom(self, sx, sy):
        idx = self.notebook.index("current"); tf = self.notebook.nametowidget(self.notebook.tabs()[idx])
        key = getattr(tf, "_chart_key", None); ch = self._charts.get(key)
        if ch is None: return
        self._user_zoomed = True; ax = ch["ax"]
        xl, xu = ax.get_xlim(); yl, yu = ax.get_ylim()
        cx, cy = (xl+xu)/2, (yl+yu)/2
        hw, hh = (xu-xl)/2*sx, (yu-yl)/2*sy
        ax.set_xlim(cx-hw, cx+hw); ax.set_ylim(cy-hh, cy+hh)
        ch["canvas"].draw_idle()

    def _update_smoothing_and_plot(self):
        cfg = self.config
        ns = len(self.o2_smoother.smoothed)
        for i in range(ns, len(self.raw_o2_list)):
            self.o2_smoother.add(self.raw_o2_list[i],cfg["alpha"],cfg["window_size"],cfg["w1"],cfg["w2"],cfg["w3"],cfg.get("adapt_threshold",3.0))
            self.n2_smoother.add(self.raw_n2_list[i],cfg["alpha"],cfg["window_size"],cfg["w1"],cfg["w2"],cfg["w3"],cfg.get("adapt_threshold",3.0))
            self.co2_smoother.add(self.raw_co2_list[i],cfg["alpha"],cfg["window_size"],cfg["w1"],cfg["w2"],cfg["w3"],cfg.get("adapt_threshold",3.0))
        self.smoothed_o2_list=self.o2_smoother.smoothed; self.smoothed_n2_list=self.n2_smoother.smoothed; self.smoothed_co2_list=self.co2_smoother.smoothed
        x = list(range(1, len(self.smoothed_o2_list)+1))
        self._update_all_charts(x); self._update_fluctuation_labels()

    def _update_all_charts(self, x):
        smo={"O2":self.smoothed_o2_list,"N2":self.smoothed_n2_list,"CO2":self.smoothed_co2_list}
        raw={"O2":self.raw_o2_list,"N2":self.raw_n2_list,"CO2":self.raw_co2_list}
        for key, ch in self._charts.items():
            ax = ch["ax"]
            if ch["gases"] == "all":
                ch["line_o2"].set_data(x,smo["O2"]); ch["line_n2"].set_data(x,smo["N2"]); ch["line_co2"].set_data(x,smo["CO2"])
                ch["raw_scatters"]["O2"].set_data(x,raw["O2"]); ch["raw_scatters"]["N2"].set_data(x,raw["N2"])
            else:
                gas = ch["gases"]; ch["line_o2"].set_data(x,smo[gas]); ch["raw_scatters"][gas].set_data(x,raw[gas])
            n = len(x)
            if not self._user_zoomed:
                if n <= 50: ax.set_xlim(0.5, n+0.5)
                else: ax.set_xlim(n-50+0.5, n+0.5)
            if ch["gases"]=="all" and not self._user_zoomed: ax.set_ylim(-0.02,1.02)
            elif ch["gases"]!="all" and not self._user_zoomed:
                vals = raw[ch["gases"]]
                if len(vals)>=2:
                    lo, hi = float(np.min(vals)), float(np.max(vals))
                    pad = max((hi-lo)*0.2, 0.001); ax.set_ylim(lo-pad, hi+pad)
            ax.relim(); ax.autoscale_view(scalex=False, scaley=False); ch["canvas"].draw_idle()

    def _update_fluctuation_labels(self):
        if self.processed_count < 2: return
        f = self._compute_fluctuation()
        self.fluc_o2_var.set(f"O2: raw std={f['raw_o2']['std']:.4f} CV={f['raw_o2']['cv']:.3f} -> smooth std={f['smo_o2']['std']:.4f} CV={f['smo_o2']['cv']:.3f} ({f['o2_std_red']:+.1f}%)")
        self.fluc_n2_var.set(f"N2: raw std={f['raw_n2']['std']:.4f} CV={f['raw_n2']['cv']:.3f} -> smooth std={f['smo_n2']['std']:.4f} CV={f['smo_n2']['cv']:.3f} ({f['n2_std_red']:+.1f}%)")
        self.fluc_co2_var.set(f"CO2: raw std={f['raw_co2']['std']:.4f} CV={f['raw_co2']['cv']:.3f} -> smooth std={f['smo_co2']['std']:.4f} CV={f['smo_co2']['cv']:.3f} ({f['co2_std_red']:+.1f}%)")
        so = self.o2_smoother
        if so.smoothed: self.status_var.set(f"EMA={so.emas[-1]:.4f} | MA={so.mas[-1]:.4f} | Raw={self.raw_o2_list[-1]:.4f} -> O2={so.smoothed[-1]:.4f}  w_raw={so.w_raws[-1]:.2f}")

    def _maybe_save_results(self):
        if not self.save_var.get(): return
        od = self.output_path_var.get().strip()
        if not od: return
        if not os.path.isdir(od):
            try: os.makedirs(od, exist_ok=True)
            except Exception as e: self._set_status(f""); return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cp = os.path.join(od, f"concentration_{ts}.csv")
        try:
            with open(cp,"w",newline="",encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["idx","file","O2_raw","O2EMA","O2MA","O2RawW","O2_smooth","N2_raw","N2EMA","N2MA","N2RawW","N2_smooth","CO2_raw","CO2_smooth"])
                so,sn,sc = self.o2_smoother,self.n2_smoother,self.co2_smoother
                for i in range(len(self.raw_o2_list)):
                    w.writerow([i+1,self.file_labels[i] if i<len(self.file_labels) else f"file_{i+1}",
                        f"{self.raw_o2_list[i]:.6f}",f"{so.emas[i]:.6f}" if i<len(so.emas) else "",f"{so.mas[i]:.6f}" if i<len(so.mas) else "",f"{so.w_raws[i]:.4f}" if i<len(so.w_raws) else "",f"{self.smoothed_o2_list[i]:.6f}" if i<len(self.smoothed_o2_list) else "",
                        f"{self.raw_n2_list[i]:.6f}",f"{sn.emas[i]:.6f}" if i<len(sn.emas) else "",f"{sn.mas[i]:.6f}" if i<len(sn.mas) else "",f"{sn.w_raws[i]:.4f}" if i<len(sn.w_raws) else "",f"{self.smoothed_n2_list[i]:.6f}" if i<len(self.smoothed_n2_list) else "",
                        f"{self.raw_co2_list[i]:.6f}",f"{self.smoothed_co2_list[i]:.6f}" if i<len(self.smoothed_co2_list) else ""])
                w.writerow([]); fluc = self._compute_fluctuation()
                w.writerow(["","","O2_raw","O2_smooth","N2_raw","N2_smooth","CO2_raw","CO2_smooth"])
                for lb,rk,sk in [("std","std","std"),("mean","mean","mean"),("range","range","range"),("CV","cv","cv")]:
                    w.writerow(["",lb,f"{fluc['raw_o2'][rk]:.6f}",f"{fluc['smo_o2'][sk]:.6f}",f"{fluc['raw_n2'][rk]:.6f}",f"{fluc['smo_n2'][sk]:.6f}",f"{fluc['raw_co2'][rk]:.6f}",f"{fluc['smo_co2'][sk]:.6f}"])
                w.writerow(["","std_red(%)",f"{fluc['o2_std_red']:.2f}%","",f"{fluc['n2_std_red']:.2f}%","",f"{fluc['co2_std_red']:.2f}%",""])
            self._set_status(f"")
        except Exception as e: self._set_status(f"")
        pp = os.path.join(od, f"concentration_{ts}.png")
        try: self.fig.savefig(pp, dpi=200, bbox_inches="tight"); self._set_status(f"")
        except Exception as e: self._set_status(f"")

    @staticmethod
    def _parse_time_label(fp, fi):
        bn = os.path.splitext(os.path.basename(fp))[0]
        for pat in [r'(\d{8}[_\-]?\d{6})', r'(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})', r'(\d{2}[-_]\d{2}[-_]\d{2})']:
            m = re.search(pat, bn)
            if m: return m.group(1)
        return str(fi)

    def _open_single_file(self):
        fmt = self.fmt_var.get()
        ft = [("TXT","*.txt")] if fmt=="txt" else [("SIF","*.sif")]
        p = filedialog.askopenfilename(title="", filetypes=ft+[("All","*.*")])
        if not p: return
        try: config = self._collect_config()
        except Exception as e: messagebox.showerror("",str(e)); return
        self.config = config
        try: C_O2, C_N2, C_CO2 = process_single_file(p, config)
        except Exception as e: messagebox.showerror("",f"{e}\n\n{traceback.format_exc()}"); return
        C_O2=max(0.0,min(1.0,C_O2)); C_N2=max(0.0,min(1.0,C_N2)); C_CO2=max(0.0,min(1.0,C_CO2))
        self.raw_o2_list.append(C_O2); self.raw_n2_list.append(C_N2); self.raw_co2_list.append(C_CO2)
        self.processed_count += 1
        self.file_labels.append(self._parse_time_label(p, self.processed_count))
        self._update_smoothing_and_plot()
        msg = f"O2={C_O2:.6f}\nN2={C_N2:.6f}\nCO2={C_CO2:.6f}"
        if self.smoothed_o2_list: msg += f"\n\nO2 smooth={self.smoothed_o2_list[-1]:.6f}\nN2 smooth={self.smoothed_n2_list[-1]:.6f}\nCO2 smooth={self.smoothed_co2_list[-1]:.6f}"
        messagebox.showinfo("", msg)

    def on_close(self): self._stop_event.set(); self.root.destroy()

def main():
    root = tk.Tk(); app = RamanApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close); root.mainloop()

if __name__ == "__main__": main()

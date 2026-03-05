"""
3D LUT Generator — Simple RBF
Based on: https://github.com/marccolemont/camera-match-LUT

No delta mode. No augmentation. No polynomial tricks.
Just: fit RBF(src_rgb) -> tgt_rgb, evaluate over lattice.
Smoothing is the only tuning knob that matters.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


class LutGenerator:

    @staticmethod
    def load_csv(path: str) -> np.ndarray:
        df = pd.read_csv(path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        arr = df.values[:, :3].astype(np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def generate(self, src_path, tgt_path, out_path,
                 grid_size, smoothing, kernel, log_cb, progress_cb):
        try:
            log_cb("📂  Loading CSVs …")
            src = self.load_csv(src_path)
            tgt = self.load_csv(tgt_path)
            log_cb(f"    {len(src)} source patches, {len(tgt)} target patches.")

            if len(src) != len(tgt):
                raise ValueError(
                    f"Row mismatch: {len(src)} source vs {len(tgt)} target.")

            progress_cb(0.05)

            # Fit RBF: src RGB -> tgt R, src RGB -> tgt G, src RGB -> tgt B
            # No delta, no polynomial, no anchors. Direct mapping.
            log_cb(f"⚙️   Fitting RBF → Red   [{kernel}, smoothing={smoothing}] …")
            rbf_r = RBFInterpolator(src, tgt[:, 0],
                                    kernel=kernel, smoothing=smoothing)
            progress_cb(0.30)

            log_cb(f"⚙️   Fitting RBF → Green …")
            rbf_g = RBFInterpolator(src, tgt[:, 1],
                                    kernel=kernel, smoothing=smoothing)
            progress_cb(0.52)

            log_cb(f"⚙️   Fitting RBF → Blue  …")
            rbf_b = RBFInterpolator(src, tgt[:, 2],
                                    kernel=kernel, smoothing=smoothing)
            progress_cb(0.65)

            # Fit quality
            err_r = np.abs(rbf_r(src) - tgt[:, 0]).mean()
            err_g = np.abs(rbf_g(src) - tgt[:, 1]).mean()
            err_b = np.abs(rbf_b(src) - tgt[:, 2]).mean()
            log_cb(f"    Fit error  R:{err_r:.5f}  G:{err_g:.5f}  B:{err_b:.5f}")

            progress_cb(0.70)

            # Build lattice — Adobe .cube: R fast axis, B slow axis
            log_cb(f"🔲  Building {grid_size}³ lattice …")
            axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
            B, G, R = np.meshgrid(axis, axis, axis, indexing="ij")
            query = np.column_stack([R.ravel(), G.ravel(), B.ravel()])
            progress_cb(0.75)

            log_cb("🧮  Evaluating …")
            out_r = np.clip(rbf_r(query), 0.0, 1.0)
            out_g = np.clip(rbf_g(query), 0.0, 1.0)
            out_b = np.clip(rbf_b(query), 0.0, 1.0)
            progress_cb(0.92)

            log_cb(f"💾  Writing → {os.path.basename(out_path)} …")
            with open(out_path, "w") as f:
                f.write("# 3D LUT — RBF\n")
                f.write(f"LUT_3D_SIZE {grid_size}\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
                for rv, gv, bv in zip(out_r, out_g, out_b):
                    f.write(f"{rv:.6f} {gv:.6f} {bv:.6f}\n")

            progress_cb(1.0)
            log_cb("✅  Done!")

        except ValueError as exc:
            log_cb(f"❌  {exc}"); raise
        except Exception as exc:
            log_cb(f"❌  {exc}"); raise


# ── UI ────────────────────────────────────────────────────────────────────────

BG      = "#1a1a2e"
PANEL   = "#16213e"
ACCENT  = "#e94560"
ACCENT2 = "#0f3460"
TEXT    = "#eaeaea"
DIM     = "#7a7a9a"
SUCCESS = "#4caf7d"
WARN    = "#f0a500"
INFO    = "#56b4d3"
MONO    = ("Courier New", 9)
SANS    = ("Segoe UI", 9)
SM      = ("Segoe UI", 7)
BTN_F   = ("Segoe UI", 9, "bold")

KERNELS = [
    "thin_plate_spline",
    "linear",
    "cubic",
    "multiquadric",
    "quintic",
    "inverse_multiquadric",
    "gaussian",
]


class LutApp:

    def __init__(self, root):
        self.root      = root
        self.generator = LutGenerator()
        self.q: queue.Queue = queue.Queue()

        self.src_var    = tk.StringVar()
        self.tgt_var    = tk.StringVar()
        self.out_var    = tk.StringVar()
        self.size_var   = tk.StringVar(value="33")
        self.smooth_var = tk.StringVar(value="0.1")
        self.kernel_var = tk.StringVar(value="thin_plate_spline")

        self._build_ui()
        self._poll()

    def _build_ui(self):
        r = self.root
        r.title("3D LUT Generator")
        r.configure(bg=BG)
        r.resizable(False, False)

        tk.Frame(r, bg=ACCENT, height=4).pack(fill="x")

        tf = tk.Frame(r, bg=BG, pady=12); tf.pack(fill="x", padx=24)
        tk.Label(tf, text="3D LUT GENERATOR",
                 font=("Segoe UI", 15, "bold"), bg=BG, fg=TEXT).pack(side="left")
        tk.Label(tf, text="RBF Interpolation",
                 font=("Segoe UI", 9), bg=BG, fg=DIM).pack(
                     side="left", padx=(10, 0), pady=(5, 0))

        outer = tk.Frame(r, bg=BG)
        outer.pack(fill="both", padx=24, pady=(0, 16))

        left = tk.Frame(outer, bg=PANEL, padx=18, pady=14)
        left.pack(side="left", fill="y")

        self._section(left, "INPUT FILES")
        self._file_row(left, "Source CSV",   self.src_var, self._pick_src)
        self._file_row(left, "Target CSV",   self.tgt_var, self._pick_tgt)

        self._section(left, "OUTPUT")
        self._file_row(left, "Output .cube", self.out_var, self._pick_out)

        self._section(left, "SETTINGS")

        # Kernel
        row = tk.Frame(left, bg=PANEL, pady=3); row.pack(fill="x")
        tk.Label(row, text="Kernel", font=SANS, bg=PANEL,
                 fg=TEXT, width=14, anchor="w").pack(side="left")
        ttk.Combobox(row, textvariable=self.kernel_var,
                     values=KERNELS, state="readonly",
                     width=22, font=MONO).pack(side="right")

        # Smoothing
        row2 = tk.Frame(left, bg=PANEL, pady=3); row2.pack(fill="x")
        tk.Label(row2, text="Smoothing", font=SANS, bg=PANEL,
                 fg=TEXT, width=14, anchor="w").pack(side="left")
        tk.Entry(row2, textvariable=self.smooth_var, width=10,
                 bg=ACCENT2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=MONO, justify="center").pack(side="right")

        # Smoothing guide card
        card = tk.Frame(left, bg="#0a1520", padx=9, pady=8)
        card.pack(fill="x", pady=(6, 4))
        tk.Label(card,
            text=(
                "Smoothing is the main tuning knob:\n"
                "  0        exact fit, will oscillate\n"
                "  0.01     tight fit, some oscillation\n"
                "  0.1      good starting point  ★\n"
                "  0.5      smoother, less accurate\n"
                "  1.0+     very smooth, blurry match\n\n"
                "If exposure ramps show colour shifts:\n"
                "  → increase smoothing until clean"
            ),
            font=SM, bg="#0a1520", fg=INFO, justify="left").pack(anchor="w")

        # Grid size
        row3 = tk.Frame(left, bg=PANEL, pady=3); row3.pack(fill="x")
        tk.Label(row3, text="Grid Size N³", font=SANS, bg=PANEL,
                 fg=TEXT, width=14, anchor="w").pack(side="left")
        tk.Entry(row3, textvariable=self.size_var, width=6,
                 bg=ACCENT2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=MONO, justify="center").pack(side="right")
        tk.Label(row3, text="33 std · 65 high", font=SM,
                 bg=PANEL, fg=DIM).pack(side="left", padx=(4, 0))

        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", pady=10)

        self.gen_btn = tk.Button(
            left, text="▶  GENERATE LUT",
            font=BTN_F, bg=ACCENT, fg="white",
            activebackground="#c0314a", activeforeground="white",
            relief="flat", padx=14, pady=9, cursor="hand2",
            command=self._on_generate)
        self.gen_btn.pack(fill="x")

        s = ttk.Style(); s.theme_use("clam")
        s.configure("P.Horizontal.TProgressbar",
                    troughcolor=BG, background=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, borderwidth=0)
        self.progress = ttk.Progressbar(
            left, orient="horizontal", mode="determinate",
            style="P.Horizontal.TProgressbar", length=280)
        self.progress.pack(fill="x", pady=(10, 0))

        # Console
        right = tk.Frame(outer, bg=BG, padx=14)
        right.pack(side="left", fill="both", expand=True)

        lh = tk.Frame(right, bg=BG); lh.pack(fill="x")
        tk.Label(lh, text="CONSOLE LOG",
                 font=("Segoe UI", 8, "bold"), bg=BG, fg=DIM).pack(side="left")
        tk.Button(lh, text="Clear", font=SM,
                  bg=PANEL, fg=DIM, relief="flat", cursor="hand2",
                  command=self._clear_log, padx=6).pack(side="right")

        self.log_box = tk.Text(
            right, width=52, height=28, bg="#0d0d1a", fg=TEXT,
            insertbackground=TEXT, font=MONO,
            relief="flat", padx=8, pady=6, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))
        for tag, col in [("info", TEXT), ("success", SUCCESS),
                         ("error", ACCENT), ("warn", WARN),
                         ("info2", INFO), ("dim", DIM)]:
            self.log_box.tag_configure(tag, foreground=col)

        sb = tk.Scrollbar(right, command=self.log_box.yview,
                          bg=PANEL, troughcolor=BG, relief="flat")
        sb.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=sb.set)

        tk.Frame(r, bg=ACCENT, height=3).pack(fill="x", side="bottom")

        self._log("Ready. Start with smoothing=0.1 and adjust.", "dim")

    def _section(self, p, text):
        f = tk.Frame(p, bg=PANEL); f.pack(fill="x", pady=(10, 4))
        tk.Label(f, text=text, font=("Segoe UI", 7, "bold"),
                 bg=PANEL, fg=DIM).pack(side="left")
        tk.Frame(f, bg=ACCENT2, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=4)

    def _file_row(self, p, label, var, cmd):
        row = tk.Frame(p, bg=PANEL, pady=3); row.pack(fill="x")
        tk.Label(row, text=label, font=SANS, bg=PANEL,
                 fg=TEXT, width=12, anchor="w").pack(side="left")
        tk.Button(row, text="Browse", font=("Segoe UI", 8),
                  bg=ACCENT2, fg=TEXT, activebackground="#1a4a80",
                  activeforeground=TEXT, relief="flat", padx=8, pady=2,
                  cursor="hand2", command=cmd).pack(side="right")
        tk.Label(row, textvariable=var, font=("Segoe UI", 8),
                 bg=PANEL, fg=DIM, anchor="w", width=26).pack(
                     side="left", padx=(6, 0))

    def _pick_src(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p: self.src_var.set(os.path.basename(p)); self._src = p

    def _pick_tgt(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p: self.tgt_var.set(os.path.basename(p)); self._tgt = p

    def _pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".cube",
            filetypes=[(".cube", "*.cube"), ("All", "*.*")])
        if p: self.out_var.set(os.path.basename(p)); self._out = p

    def _on_generate(self):
        src = getattr(self, "_src", None)
        tgt = getattr(self, "_tgt", None)
        out = getattr(self, "_out", None)
        if not src: messagebox.showerror("Missing", "Select Source CSV."); return
        if not tgt: messagebox.showerror("Missing", "Select Target CSV."); return
        if not out: messagebox.showerror("Missing", "Select output path."); return

        try:
            grid = int(self.size_var.get())
            assert 2 <= grid <= 256
        except:
            messagebox.showerror("Bad value", "Grid size must be 2–256."); return

        try:
            smooth = float(self.smooth_var.get())
            assert smooth >= 0
        except:
            messagebox.showerror("Bad value", "Smoothing must be ≥ 0."); return

        kernel = self.kernel_var.get()

        self.gen_btn.config(state="disabled", text="⏳  Generating …")
        self.progress["value"] = 0
        self._log("─" * 50, "dim")
        self._log(f"kernel={kernel}  smoothing={smooth}  grid={grid}³", "info2")

        threading.Thread(
            target=self._run,
            args=(src, tgt, out, grid, smooth, kernel),
            daemon=True).start()

    def _run(self, src, tgt, out, grid, smooth, kernel):
        try:
            self.generator.generate(src, tgt, out, grid, smooth, kernel,
                                     self._qlog, self._qprog)
            self.q.put(("done", True))
        except ValueError as e:
            self.q.put(("err", str(e))); self.q.put(("done", False))
        except Exception:
            self.q.put(("done", False))

    def _qlog(self, m):  self.q.put(("log", m))
    def _qprog(self, v): self.q.put(("prog", v))

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    tag = ("success" if val.startswith("✅") else
                           "error"   if val.startswith("❌") else
                           "warn"    if val.startswith("⚠") else
                           "info2"   if val.startswith("    ") else "info")
                    self._log(val, tag)
                elif kind == "prog":
                    self.progress["value"] = val * 100
                elif kind == "err":
                    messagebox.showerror("Error", val)
                elif kind == "done":
                    self.gen_btn.config(state="normal", text="▶  GENERATE LUT")
                    if val: self.progress["value"] = 100
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _log(self, msg, tag="info"):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")


def main():
    root = tk.Tk()
    root.configure(bg=BG)
    LutApp(root)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()
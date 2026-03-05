"""
Color Chart Patch Extractor
============================
Reads X-Rite ColorChecker Classic (24-patch, 4×6) and/or
X-Rite ColorChecker SG (140-patch, 10×14) from a single image,
and/or any number of manually-placed square readings.

Workflow
--------
1. Load Image
2. (Optional) Click 4 corners of the 24-patch chart  → TL, TR, BR, BL
3. (Optional) Click 4 corners of the 140-patch chart → TL, TR, BR, BL
4. Set the sample-square size (pixels) per chart
5. (Optional) Set manual-patch size, click "Place Patches", then click
   anywhere on the image.  Right-click removes the last placed patch.
6. Click "Extract & Save CSV"

CSV layout  →  R,G,B  (one row per patch)
            SG 140-patch rows come first, then 24-patch rows,
            then manually-placed patches in the order they were placed.

Dependencies
------------
    pip install opencv-python Pillow numpy
"""

import csv
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Natural (human-friendly) sort key
# ---------------------------------------------------------------------------

def _natural_sort_key(path: str):
    """Sort key that handles embedded numbers numerically.

    e.g. 'img2.tif' < 'img10.tif' instead of the default lexicographic
    ordering where '10' < '2'.
    """
    base = os.path.basename(path).lower()
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', base)]


# ---------------------------------------------------------------------------
# Chart definitions
# ---------------------------------------------------------------------------
CHARTS = {
    "24-patch Classic (4×6)":  {"rows": 4,  "cols": 6},
    "140-patch SG (10×14)":    {"rows": 10, "cols": 14},
}

CHART_COLORS = {
    "24-patch Classic (4×6)":  "#00AAFF",   # blue markers
    "140-patch SG (10×14)":    "#FF6600",   # orange markers
}

MANUAL_PATCH_COLOR = "#00DD55"   # green markers for manual patches

# CSV output order: SG 140 first, then 24-patch, then manual patches
CHART_CSV_ORDER = [
    "140-patch SG (10×14)",
    "24-patch Classic (4×6)",
]

# ---------------------------------------------------------------------------
# Perspective warp helper
# ---------------------------------------------------------------------------

def get_normalize_factor(dtype: np.dtype) -> float:
    """Get normalization factor based on image dtype."""
    if dtype == np.uint8:
        return 255.0
    elif dtype == np.uint16:
        return 65535.0
    elif dtype in (np.float32, np.float64):
        return 1.0
    else:
        return 255.0


def get_bit_depth_info(dtype: np.dtype) -> str:
    """Get human-readable bit depth string for dtype."""
    if dtype == np.uint8:
        return "8-bit"
    elif dtype == np.uint16:
        return "16-bit"
    elif dtype in (np.float32, np.float64):
        return "32-bit float"
    else:
        return "unknown"


def warp_chart(img_bgr: np.ndarray, corners: list[list[int]],
               rows: int, cols: int) -> np.ndarray:
    """
    Perspective-warp the quadrilateral defined by *corners*
    (TL, TR, BR, BL) to a rectangular image sized cols*100 × rows*100.
    """
    cell = 100          # pixels per cell in output
    W = cols * cell
    H = rows * cell
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H),
                                 flags=cv2.INTER_LANCZOS4)
    return warped, cell


def extract_patches(warped: np.ndarray, cell: int,
                    rows: int, cols: int,
                    sample_w: int, sample_h: int) -> list[tuple[float, float, float]]:
    """
    Sample from a rectangular region (sample_w × sample_h) centered on each patch.
    Reads raw encoded values and normalizes to 0.0-1.0 based on bit depth.
    Returns a list of (R, G, B) tuples as floats.
    """
    normalize_factor = get_normalize_factor(warped.dtype)
    
    half_w = max(1, sample_w // 2)
    half_h = max(1, sample_h // 2)
    results = []
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)
            x1 = max(0, cx - half_w)
            x2 = min(warped.shape[1], cx + half_w + 1)
            y1 = max(0, cy - half_h)
            y2 = min(warped.shape[0], cy + half_h + 1)
            patch = warped[y1:y2, x1:x2]
            if patch.size == 0:
                results.append((0.0, 0.0, 0.0))
                continue
            b, g, r_ch = cv2.split(patch)
            # Convert to float and normalize
            R = float(np.mean(r_ch)) / normalize_factor
            G = float(np.mean(g)) / normalize_factor
            B = float(np.mean(b)) / normalize_factor
            results.append((R, G, B))
    return results


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ColorChartApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Color Chart Patch Extractor")
        self.resizable(True, True)

        # --- State ---
        self.img_path:   str | None = None
        self.img_bgr:    np.ndarray | None = None   # original full-res
        self.tk_image:   ImageTk.PhotoImage | None = None
        self.scale:      float = 1.0                # canvas ↔ image ratio
        
        # Image caching for performance
        self.cached_pil_img: Image.Image | None = None  # cached scaled PIL image
        self.cached_scale: float = -1.0            # scale at which image was cached
        self.zoom_pending: bool = False            # flag for deferred high-quality zoom

        # Multi-image support
        self.image_dir:  str | None = None          # directory of images
        self.image_files: list[str] = []            # sorted list of image paths
        self.current_img_idx: int = -1              # index in image_files

        # corners[chart_key] = list of up to 4 [x, y] in ORIGINAL image coords
        self.corners: dict[str, list] = {}
        self.active_chart: str | None = None        # which chart we're clicking

        # Manual patch placement
        # Each entry: {"cx": int, "cy": int, "size": int}  — image coords
        self.manual_patches: list[dict] = []
        self.manual_mode: bool = False
        self.manual_size_var: tk.IntVar = tk.IntVar(value=20)

        # Per-chart sample size vars (width and height in pixels)
        self.sample_w_vars: dict[str, tk.IntVar] = {}  # sample width
        self.sample_h_vars: dict[str, tk.IntVar] = {}  # sample height

        # Save directory
        self.save_dir = tk.StringVar(value=os.path.expanduser("~"))

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # ── Top toolbar ─────────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="�  Import Images",
                   command=self._import_images).pack(side=tk.LEFT, padx=4)

        # Image info label
        self.img_info_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.img_info_var,
                  width=30, anchor=tk.W).pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        # Chart selection buttons + sample-size entries
        for key, cfg in CHARTS.items():
            color = CHART_COLORS[key]
            frm = ttk.LabelFrame(toolbar, text=key, padding=4)
            frm.pack(side=tk.LEFT, padx=4)

            btn = tk.Button(
                frm, text="📍 Select Corners",
                bg=color, fg="white", activebackground=color,
                relief=tk.RAISED, bd=2,
                command=lambda k=key: self._start_corner_selection(k))
            btn.pack(side=tk.LEFT, padx=2)

            # Sample width
            ttk.Label(frm, text="W:").pack(side=tk.LEFT, padx=(8, 2))
            var_w = tk.IntVar(value=50)
            self.sample_w_vars[key] = var_w
            spn_w = ttk.Spinbox(frm, from_=2, to=200, textvariable=var_w,
                        width=4, command=self._redraw)
            spn_w.pack(side=tk.LEFT, padx=1)

            # Sample height
            ttk.Label(frm, text="H:").pack(side=tk.LEFT, padx=(4, 2))
            var_h = tk.IntVar(value=50)
            self.sample_h_vars[key] = var_h
            spn_h = ttk.Spinbox(frm, from_=2, to=200, textvariable=var_h,
                        width=4, command=self._redraw)
            spn_h.pack(side=tk.LEFT, padx=1)

            clear_btn = ttk.Button(
                frm, text="✖ Clear",
                command=lambda k=key: self._clear_chart(k))
            clear_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        # ── Manual patch placement ───────────────────────────────────────────
        frm_manual = ttk.LabelFrame(toolbar, text="Manual Patches", padding=4)
        frm_manual.pack(side=tk.LEFT, padx=4)

        self.manual_btn = tk.Button(
            frm_manual, text="🖱 Place Patches",
            bg=MANUAL_PATCH_COLOR, fg="white",
            activebackground=MANUAL_PATCH_COLOR,
            relief=tk.RAISED, bd=2,
            command=self._toggle_manual_mode)
        self.manual_btn.pack(side=tk.LEFT, padx=2)

        ttk.Label(frm_manual, text="Size:").pack(side=tk.LEFT, padx=(8, 2))
        spn_manual = ttk.Spinbox(
            frm_manual, from_=2, to=500,
            textvariable=self.manual_size_var,
            width=5, command=self._redraw)
        spn_manual.pack(side=tk.LEFT, padx=1)

        self.manual_count_var = tk.StringVar(value="0 pts")
        ttk.Label(frm_manual, textvariable=self.manual_count_var,
                  width=6).pack(side=tk.LEFT, padx=4)

        ttk.Button(frm_manual, text="↩ Undo",
                   command=self._undo_manual_patch).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm_manual, text="✖ Clear",
                   command=self._clear_manual_patches).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(toolbar, text="✅  Extract & Save CSV",
                   command=self._extract_and_save).pack(side=tk.LEFT, padx=4)

        ttk.Button(toolbar, text="📂 Export separate CSVs for exposure bracket",
                   command=self._export_separate_csvs).pack(side=tk.LEFT, padx=4)

        # ── Status bar ──────────────────────────────────────────────────────
        self.status_var = tk.StringVar(
            value="Load an image to begin.")
        status_bar = ttk.Label(self, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Canvas with scrollbars ───────────────────────────────────────────
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b",
                                cursor="crosshair")
        hsb = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL,
                             command=self.canvas.xview)
        vsb = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                             command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hsb.set,
                              yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_click)
        self.canvas.bind("<Configure>",     self._on_canvas_resize)
        # Mouse-wheel zoom
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)       # Windows
        self.canvas.bind("<Control-Button-4>",   self._on_zoom_up)    # Linux
        self.canvas.bind("<Control-Button-5>",   self._on_zoom_down)  # Linux

    # -----------------------------------------------------------------------
    # Image loading & display
    # -----------------------------------------------------------------------

    def _import_images(self):
        """Import one or more images (files or directory)."""
        # Try multi-file selection first
        paths = filedialog.askopenfilenames(
            title="Select Image(s)",
            filetypes=[("Image files",
                        "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp *.exr"),
                       ("All files", "*.*")])
        
        if paths:
            # User selected one or more files
            if len(paths) == 1:
                # Single file - load directly
                self._load_single_image(paths[0])
            else:
                # Multiple files - treat as batch
                self._load_image_batch(list(paths))
            return
        
        # If no files selected, offer directory import as alternative
        if not messagebox.askyesno(
            "Import from Folder?",
            "No files selected.\n\nWould you like to import all images from a folder instead?"):
            return
        
        directory = filedialog.askdirectory(title="Select directory with images")
        if not directory:
            return
        
        self._load_directory(directory)

    def _load_single_image(self, path: str):
        """Load a single image file."""
        self.img_path = path
        # Use IMREAD_UNCHANGED to preserve original bit depth and encoding
        self.img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.img_bgr is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return
        # Reset corners and manual patches whenever a new image is loaded
        self.corners.clear()
        self.manual_patches.clear()
        self.manual_mode = False
        self._update_manual_btn_state()
        self.active_chart = None
        self.image_files = []
        self.image_dir = None
        self.current_img_idx = -1
        self.cached_pil_img = None  # Clear cache
        self._fit_image_to_canvas()
        
        bit_info = get_bit_depth_info(self.img_bgr.dtype)
        self.img_info_var.set(f"1 image loaded")
        self._status(f"Loaded: {os.path.basename(path)}  "
                     f"({self.img_bgr.shape[1]}×{self.img_bgr.shape[0]} px, {bit_info})")

    def _load_image_batch(self, paths: list[str]):
        """Load multiple image files for batch processing."""
        # Sort by filename (natural / numeric order)
        paths.sort(key=_natural_sort_key)
        
        self.image_files = paths
        self.current_img_idx = 0
        self.image_dir = os.path.dirname(paths[0])
        self.img_info_var.set(f"{len(paths)} images ready")
        
        # Reset corners and manual patches for new set of images
        self.corners.clear()
        self.manual_patches.clear()
        self.manual_mode = False
        self._update_manual_btn_state()
        self.active_chart = None
        
        # Load first image for preview
        self._load_current_image()

    def _load_directory(self, directory: str):
        """Load all images from a directory for batch processing."""
        # Supported image extensions
        img_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".exr")
        
        # Find all image files and sort by filename
        image_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(img_extensions):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path):
                    image_files.append(full_path)
        
        if not image_files:
            messagebox.showwarning("No Images", f"No image files found in:\n{directory}")
            return
        
        # Sort by filename (natural / numeric order)
        image_files.sort(key=_natural_sort_key)
        
        self.image_dir = directory
        self.image_files = image_files
        self.current_img_idx = 0
        self.img_info_var.set(f"{len(image_files)} images ready")
        
        # Reset corners and manual patches for new set of images
        self.corners.clear()
        self.manual_patches.clear()
        self.manual_mode = False
        self._update_manual_btn_state()
        self.active_chart = None
        
        # Load first image for preview
        self._load_current_image()

    def _load_current_image(self):
        """Load the image at current_img_idx."""
        if self.current_img_idx < 0 or self.current_img_idx >= len(self.image_files):
            return
        
        path = self.image_files[self.current_img_idx]
        self.img_path = path
        self.img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.img_bgr is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return
        
        self.cached_pil_img = None  # Clear cache
        self._fit_image_to_canvas()
        
        bit_info = get_bit_depth_info(self.img_bgr.dtype)
        img_num = self.current_img_idx + 1 if len(self.image_files) > 1 else ""
        status = f"[{img_num}/{len(self.image_files)}] " if len(self.image_files) > 1 else ""
        self._status(f"{status}Loaded: {os.path.basename(path)}  "
                     f"({self.img_bgr.shape[1]}×{self.img_bgr.shape[0]} px, {bit_info})")

    def _fit_image_to_canvas(self):
        """Scale the image to fit inside the current canvas, then redraw."""
        if self.img_bgr is None:
            return
        H, W = self.img_bgr.shape[:2]
        cw = max(self.canvas.winfo_width(),  1)
        ch = max(self.canvas.winfo_height(), 1)
        self.scale = min(cw / W, ch / H, 1.0)   # never upscale on initial load
        self.cached_pil_img = None  # Clear cache for new scale
        self._redraw()

    def _redraw(self):
        """Re-render the image and all annotations onto the canvas."""
        if self.img_bgr is None:
            return
        H, W = self.img_bgr.shape[:2]
        nw, nh = max(1, int(W * self.scale)), max(1, int(H * self.scale))

        # Check cache and use fast resampling if zooming
        if (self.cached_pil_img is not None and 
            abs(self.cached_scale - self.scale) / max(self.cached_scale, 0.001) < 0.25):
            # Cache is close enough, use cached version
            pil_img = self.cached_pil_img.resize((nw, nh), Image.BILINEAR)
            use_hq = False
        else:
            # Convert image to uint8 for display (handles different bit depths)
            display_img = self._convert_to_display_uint8(self.img_bgr)
            
            # Convert BGR → RGB for PIL
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img_full = Image.fromarray(img_rgb)
            
            # Use fast resampling for zoom responsiveness
            use_hq = False
            resample_method = Image.BILINEAR
            pil_img = pil_img_full.resize((nw, nh), resample_method)
            
            # Cache at a reasonable size for faster repeated zooming
            cached_scale_size = max(200, min(800, max(nw, nh)))
            if max(nw, nh) > 100:
                cache_w = int(W * cached_scale_size / max(nw, nh))
                cache_h = int(H * cached_scale_size / max(nw, nh))
                self.cached_pil_img = pil_img_full.resize((cache_w, cache_h), Image.LANCZOS)
                self.cached_scale = cached_scale_size / max(nw, nh) * self.scale
        
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Draw all confirmed corners & outlines
        for key, pts in self.corners.items():
            color = CHART_COLORS[key]
            scaled = [[int(p[0] * self.scale),
                       int(p[1] * self.scale)] for p in pts]
            for i, (sx, sy) in enumerate(scaled):
                r = 6
                self.canvas.create_oval(
                    sx - r, sy - r, sx + r, sy + r,
                    fill=color, outline="white", width=1, tags="annotation")
                self.canvas.create_text(
                    sx + 8, sy - 8, text=["TL","TR","BR","BL"][i],
                    fill=color, font=("Arial", 9, "bold"), tags="annotation")
            if len(scaled) == 4:
                pts_flat = [coord for p in scaled for coord in p]
                self.canvas.create_polygon(
                    *pts_flat,
                    outline=color, fill="", width=2,
                    dash=(6, 3), tags="annotation")
                # Draw patch grid overlay with sampling rectangles
                sample_w = self.sample_w_vars[key].get()
                sample_h = self.sample_h_vars[key].get()
                self._draw_grid(scaled, CHARTS[key]["rows"],
                                CHARTS[key]["cols"], color, sample_w, sample_h)

        # Highlight "next click" corner label when active
        if self.active_chart:
            pts = self.corners.get(self.active_chart, [])
            labels = ["TL", "TR", "BR", "BL"]
            if len(pts) < 4:
                nxt = labels[len(pts)]
                color = CHART_COLORS[self.active_chart]
                self.canvas.create_text(
                    10, 10, anchor=tk.NW,
                    text=f"↖ Click  {nxt}  corner of  {self.active_chart}",
                    fill=color, font=("Arial", 12, "bold"),
                    tags="annotation")

        # Draw manually placed patches
        self._draw_manual_patches()

        # Show manual mode hint
        if self.manual_mode:
            self.canvas.create_text(
                10, 10, anchor=tk.NW,
                text="🖱 Click to place patch  |  Right-click to undo",
                fill=MANUAL_PATCH_COLOR, font=("Arial", 12, "bold"),
                tags="annotation")

    def _draw_grid(self, scaled_corners, rows, cols, color, sample_w, sample_h):
        """Draw patch-grid overlay with sampling rectangles on the canvas."""
        # Use OpenCV's perspective mapping in canvas coordinates
        src = np.array(scaled_corners, dtype=np.float32)
        cell = 50   # just for overlay; doesn't need to match warp
        W, H = cols * cell, rows * cell
        dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst, src)   # inverse: canvas→quad

        # Draw column dividers
        for c in range(cols + 1):
            pts_in = np.array([[[c * cell, 0]],
                                [[c * cell, H]]], dtype=np.float32)
            pts_out = cv2.perspectiveTransform(pts_in, M)
            x1, y1 = pts_out[0][0]
            x2, y2 = pts_out[1][0]
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill=color, width=1,
                                    dash=(2, 4), tags="annotation")
        # Draw row dividers
        for r in range(rows + 1):
            pts_in = np.array([[[0,       r * cell]],
                                [[W,       r * cell]]], dtype=np.float32)
            pts_out = cv2.perspectiveTransform(pts_in, M)
            x1, y1 = pts_out[0][0]
            x2, y2 = pts_out[1][0]
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill=color, width=1,
                                    dash=(2, 4), tags="annotation")
        
        # Draw sampling rectangles (scaled down for visibility)
        scale_factor = cell / 100.0  # since extraction uses 100px cells
        scaled_sample_w = sample_w * scale_factor
        scaled_sample_h = sample_h * scale_factor
        half_w = scaled_sample_w / 2
        half_h = scaled_sample_h / 2
        
        for r in range(rows):
            for c in range(cols):
                # Center of this patch in overlay coords
                cx = (c + 0.5) * cell
                cy = (r + 0.5) * cell
                # Rectangle corners in overlay coords
                rect_pts = np.array([
                    [[cx - half_w, cy - half_h]],
                    [[cx + half_w, cy - half_h]],
                    [[cx + half_w, cy + half_h]],
                    [[cx - half_w, cy + half_h]]
                ], dtype=np.float32)
                # Transform to canvas coords
                rect_canvas = cv2.perspectiveTransform(rect_pts, M)
                # Draw rectangle
                for i in range(4):
                    p1 = rect_canvas[i][0]
                    p2 = rect_canvas[(i + 1) % 4][0]
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                          fill=color, width=1,
                                          tags="annotation")

    # -----------------------------------------------------------------------
    # Zoom
    # -----------------------------------------------------------------------

    def _on_zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self._apply_zoom(factor)

    def _on_zoom_up(self, event):
        self._apply_zoom(1.1)

    def _on_zoom_down(self, event):
        self._apply_zoom(0.9)

    def _apply_zoom(self, factor):
        if self.img_bgr is None:
            return
        H, W = self.img_bgr.shape[:2]
        new_scale = max(0.05, min(self.scale * factor, 8.0))
        self.scale = new_scale
        self._redraw()

    def _on_canvas_resize(self, event):
        if self.img_bgr is not None and not self.corners:
            self._fit_image_to_canvas()

    # -----------------------------------------------------------------------
    # Corner selection
    # -----------------------------------------------------------------------

    def _start_corner_selection(self, chart_key: str):
        if self.img_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        # Deactivate manual mode if it was on
        if self.manual_mode:
            self.manual_mode = False
            self._update_manual_btn_state()
        self.corners[chart_key] = []
        self.active_chart = chart_key
        self._redraw()
        self._status(f"Click TL corner of  {chart_key}  …")

    def _clear_chart(self, chart_key: str):
        self.corners.pop(chart_key, None)
        if self.active_chart == chart_key:
            self.active_chart = None
        self._redraw()
        self._status(f"Cleared corners for {chart_key}.")

    def _on_canvas_click(self, event):
        if self.img_bgr is None:
            return
        # Convert canvas (scrolled) coords → image coords
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = int(cx / self.scale)
        iy = int(cy / self.scale)

        # Clamp to image bounds
        H, W = self.img_bgr.shape[:2]
        ix = max(0, min(ix, W - 1))
        iy = max(0, min(iy, H - 1))

        if self.manual_mode:
            size = max(2, self.manual_size_var.get())
            self.manual_patches.append({"cx": ix, "cy": iy, "size": size})
            self._update_manual_count()
            self._redraw()
            self._status(
                f"Manual patch #{len(self.manual_patches)} placed at "
                f"({ix}, {iy})  size={size}px")
            return

        if self.active_chart is None:
            return

        pts = self.corners.setdefault(self.active_chart, [])
        if len(pts) >= 4:
            return
        pts.append([ix, iy])
        self._redraw()

        labels = ["TL", "TR", "BR", "BL"]
        if len(pts) < 4:
            self._status(
                f"Click  {labels[len(pts)]}  corner of  {self.active_chart}  …")
        else:
            self.active_chart = None
            self._status(
                f"✔ {list(self.corners.keys())[-1]}  corners set. "
                "Select another chart or click Extract & Save CSV.")

    def _on_canvas_right_click(self, event):
        """Right-click: undo last manual patch (only in manual mode)."""
        if self.manual_mode and self.manual_patches:
            self.manual_patches.pop()
            self._update_manual_count()
            self._redraw()
            self._status(
                f"Removed last manual patch. "
                f"{len(self.manual_patches)} remaining.")

    # -----------------------------------------------------------------------
    # Manual patch management
    # -----------------------------------------------------------------------

    def _toggle_manual_mode(self):
        if self.img_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        self.manual_mode = not self.manual_mode
        # Deactivate chart corner mode when entering manual mode
        if self.manual_mode:
            self.active_chart = None
        self._update_manual_btn_state()
        self._redraw()
        if self.manual_mode:
            self._status(
                "Manual mode ON — click to place patches, right-click to undo.")
        else:
            self._status("Manual mode OFF.")

    def _update_manual_btn_state(self):
        if self.manual_mode:
            self.manual_btn.config(relief=tk.SUNKEN, text="🖱 Placing…")
        else:
            self.manual_btn.config(relief=tk.RAISED, text="🖱 Place Patches")

    def _update_manual_count(self):
        n = len(self.manual_patches)
        self.manual_count_var.set(f"{n} pt{'s' if n != 1 else ''}")

    def _undo_manual_patch(self):
        if self.manual_patches:
            self.manual_patches.pop()
            self._update_manual_count()
            self._redraw()
            self._status(
                f"Removed last manual patch. "
                f"{len(self.manual_patches)} remaining.")

    def _clear_manual_patches(self):
        self.manual_patches.clear()
        self.manual_mode = False
        self._update_manual_btn_state()
        self._update_manual_count()
        self._redraw()
        self._status("Cleared all manual patches.")

    def _draw_manual_patches(self):
        """Draw all manually placed patch squares on the canvas."""
        for i, mp in enumerate(self.manual_patches):
            cx_img, cy_img, size = mp["cx"], mp["cy"], mp["size"]
            # Convert image coords → canvas coords
            cx_c = cx_img * self.scale
            cy_c = cy_img * self.scale
            half = (size * self.scale) / 2

            # Filled translucent-ish rectangle (drawn as outline)
            self.canvas.create_rectangle(
                cx_c - half, cy_c - half,
                cx_c + half, cy_c + half,
                outline=MANUAL_PATCH_COLOR, fill="", width=2,
                tags="annotation")
            # Center dot
            r = 4
            self.canvas.create_oval(
                cx_c - r, cy_c - r, cx_c + r, cy_c + r,
                fill=MANUAL_PATCH_COLOR, outline="white", width=1,
                tags="annotation")
            # Index label
            self.canvas.create_text(
                cx_c + half + 4, cy_c - half,
                anchor=tk.NW,
                text=str(i + 1),
                fill=MANUAL_PATCH_COLOR,
                font=("Arial", 8, "bold"),
                tags="annotation")

    # -----------------------------------------------------------------------
    # Shared extraction logic
    # -----------------------------------------------------------------------

    def _extract_patches_for_image(
        self,
        img_bgr: np.ndarray,
        ordered_keys: list[str],
    ) -> list[tuple[float, float, float]]:
        """Extract all patch data (grid charts + manual) from a single image.

        This is the single source of truth for patch extraction, used by
        both the regular export and the separate-CSV export so that they
        always produce identical readings.

        Returns a list of (R, G, B) tuples normalised to 0.0–1.0.
        """
        rows_data: list[tuple[float, float, float]] = []

        # ── Grid charts (SG 140 → 24-patch) ────────────────────────
        for key in ordered_keys:
            cfg = CHARTS[key]
            r_count, c_count = cfg["rows"], cfg["cols"]
            corners = self.corners[key]
            sample_w = self.sample_w_vars[key].get()
            sample_h = self.sample_h_vars[key].get()

            warped, cell = warp_chart(img_bgr, corners, r_count, c_count)
            patches = extract_patches(warped, cell, r_count, c_count,
                                      sample_w, sample_h)
            rows_data.extend(patches)

        # ── Manual patches (in placement order) ─────────────────────
        if self.manual_patches:
            img_H, img_W = img_bgr.shape[:2]
            norm = get_normalize_factor(img_bgr.dtype)

            for mp in self.manual_patches:
                cx, cy, size = mp["cx"], mp["cy"], mp["size"]
                half = size // 2
                x1 = max(0, cx - half)
                x2 = min(img_W, cx + half + 1)
                y1 = max(0, cy - half)
                y2 = min(img_H, cy + half + 1)
                patch = img_bgr[y1:y2, x1:x2]
                if patch.size == 0:
                    rows_data.append((0.0, 0.0, 0.0))
                else:
                    b, g, r_ch = cv2.split(patch)
                    R = float(np.mean(r_ch)) / norm
                    G = float(np.mean(g))   / norm
                    B = float(np.mean(b))   / norm
                    rows_data.append((R, G, B))

        return rows_data

    # -----------------------------------------------------------------------
    # Extraction & CSV export
    # -----------------------------------------------------------------------

    def _extract_and_save(self):
        if self.img_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if not self.corners and not self.manual_patches:
            messagebox.showwarning(
                "Nothing to Extract",
                "Please select corners for at least one chart, "
                "or place manual patches.")
            return

        # Validate all selected charts have 4 corners
        for key, pts in self.corners.items():
            if len(pts) != 4:
                messagebox.showwarning(
                    "Incomplete Selection",
                    f"Chart '{key}' only has {len(pts)} corner(s) set.\n"
                    "Please complete the corner selection.")
                return

        # Process charts in canonical CSV order: SG 140 first, then 24-patch
        ordered_keys = [k for k in CHART_CSV_ORDER if k in self.corners]

        # Determine how many images to process
        if self.image_files and len(self.image_files) > 0:
            images_to_process = self.image_files
        else:
            images_to_process = [self.img_path] if self.img_path else []

        if not images_to_process:
            messagebox.showwarning("No Images", "No images to process.")
            return

        # Total steps: grid charts + manual patches, per image
        total_steps = len(images_to_process) * (len(ordered_keys) + (1 if self.manual_patches else 0))

        progress_win = tk.Toplevel(self)
        progress_win.title("Extracting…")
        progress_win.resizable(False, False)
        progress_win.grab_set()
        ttk.Label(progress_win, text="Extracting patch data…",
                  padding=10).pack()
        progress = ttk.Progressbar(progress_win, length=300,
                                   mode="determinate",
                                   maximum=max(total_steps, 1))
        progress.pack(padx=20, pady=(0, 10))
        progress_label = ttk.Label(progress_win, text="")
        progress_label.pack()
        self.update_idletasks()

        all_rows: list[tuple[float, float, float]] = []

        # Process each image
        for img_idx, img_path in enumerate(images_to_process):
            img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                messagebox.showerror("Error", f"Cannot read image:\n{img_path}")
                progress_win.destroy()
                return

            status_text = f"{img_idx + 1}/{len(images_to_process)} - {os.path.basename(img_path)}"
            progress_label.config(text=status_text)
            self.update_idletasks()

            image_patches = self._extract_patches_for_image(
                img_bgr, ordered_keys)
            all_rows.extend(image_patches)
            progress.step(len(ordered_keys) + (1 if self.manual_patches else 0))
            self.update_idletasks()

        progress_win.destroy()

        # Build output filename
        if self.image_files:
            # Use directory name for batch processing
            dir_name = os.path.basename(os.path.normpath(self.image_dir))
            out_filename = f"{dir_name}_patches.csv"
        else:
            # Use single image name
            base = os.path.splitext(os.path.basename(self.img_path))[0]
            out_filename = f"{base}_patches.csv"

        # Prompt user for save location
        out_path = filedialog.asksaveasfilename(
            title="Save patches as CSV",
            initialfile=out_filename,
            initialdir=self.save_dir.get(),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        
        if not out_path:
            # User cancelled
            return
        
        # Update save_dir for next time
        self.save_dir.set(os.path.dirname(out_path))

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Format each normalized float value with full precision, no header
                for r, g, b in all_rows:
                    # Use repr() for ~17 significant digits of float precision
                    writer.writerow([repr(r), repr(g), repr(b)])
        except OSError as e:
            messagebox.showerror("Save Error", str(e))
            return

        num_images = len(images_to_process)
        img_text = f"{num_images} image{'s' if num_images != 1 else ''}"
        chart_count = sum(CHARTS[k]["rows"] * CHARTS[k]["cols"]
                          for k in ordered_keys) * num_images
        manual_count = len(self.manual_patches) * num_images
        detail_parts = []
        if chart_count:
            detail_parts.append(f"{chart_count} grid patch{'es' if chart_count != 1 else ''}")
        if manual_count:
            detail_parts.append(f"{manual_count} manual patch{'es' if manual_count != 1 else ''}")
        detail = "  |  ".join(detail_parts)
        messagebox.showinfo(
            "Done",
            f"Extracted {len(all_rows)} patches from {img_text}.\n"
            f"({detail})\n\nSaved to:\n{out_path}")
        self._status(
            f"✔ Saved {len(all_rows)} patches from {img_text} → {os.path.basename(out_path)}")

    # -----------------------------------------------------------------------
    # Export separate CSVs for exposure bracket
    # -----------------------------------------------------------------------

    def _export_separate_csvs(self):
        """Export one CSV per image, preserving photo order.

        Each CSV is named  <NNN>_<original_filename>_patches.csv  where NNN
        is a zero-padded index that preserves the sorted order of the photos.
        The user picks a directory; all files are written there.
        """
        if self.img_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if not self.corners and not self.manual_patches:
            messagebox.showwarning(
                "Nothing to Extract",
                "Please select corners for at least one chart, "
                "or place manual patches.")
            return

        # Validate all selected charts have 4 corners
        for key, pts in self.corners.items():
            if len(pts) != 4:
                messagebox.showwarning(
                    "Incomplete Selection",
                    f"Chart '{key}' only has {len(pts)} corner(s) set.\n"
                    "Please complete the corner selection.")
                return

        # Determine images to process
        if self.image_files and len(self.image_files) > 0:
            images_to_process = self.image_files
        else:
            images_to_process = [self.img_path] if self.img_path else []

        if not images_to_process:
            messagebox.showwarning("No Images", "No images to process.")
            return

        # Ask user for output directory
        out_dir = filedialog.askdirectory(
            title="Select folder for separate CSV files",
            initialdir=self.save_dir.get())
        if not out_dir:
            return
        self.save_dir.set(out_dir)

        ordered_keys = [k for k in CHART_CSV_ORDER if k in self.corners]

        total_steps = len(images_to_process)
        progress_win = tk.Toplevel(self)
        progress_win.title("Exporting separate CSVs…")
        progress_win.resizable(False, False)
        progress_win.grab_set()
        ttk.Label(progress_win, text="Exporting one CSV per image…",
                  padding=10).pack()
        progress = ttk.Progressbar(progress_win, length=300,
                                   mode="determinate",
                                   maximum=max(total_steps, 1))
        progress.pack(padx=20, pady=(0, 10))
        progress_label = ttk.Label(progress_win, text="")
        progress_label.pack()
        self.update_idletasks()

        # Zero-pad width for ordering prefix
        pad_width = len(str(len(images_to_process)))
        saved_paths: list[str] = []
        errors: list[str] = []

        for img_idx, img_path in enumerate(images_to_process):
            img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                errors.append(os.path.basename(img_path))
                progress.step(1)
                self.update_idletasks()
                continue

            progress_label.config(
                text=f"{img_idx + 1}/{len(images_to_process)} – "
                     f"{os.path.basename(img_path)}")
            self.update_idletasks()

            rows_data = self._extract_patches_for_image(
                img_bgr, ordered_keys)

            # ── Write CSV for this image ────────────────────────────────────
            base = os.path.splitext(os.path.basename(img_path))[0]
            csv_name = f"{str(img_idx + 1).zfill(pad_width)}_{base}_patches.csv"
            csv_path = os.path.join(out_dir, csv_name)

            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r, g, b in rows_data:
                        writer.writerow([repr(r), repr(g), repr(b)])
                saved_paths.append(csv_path)
            except OSError as e:
                errors.append(f"{os.path.basename(img_path)}: {e}")

            progress.step(1)
            self.update_idletasks()

        progress_win.destroy()

        # Report results
        msg_parts = [f"Exported {len(saved_paths)} CSV file(s) to:\n{out_dir}"]
        if errors:
            msg_parts.append(f"\n\nFailed for {len(errors)} image(s):\n"
                             + "\n".join(errors))
        messagebox.showinfo("Export Complete", "\n".join(msg_parts))
        self._status(
            f"✔ Exported {len(saved_paths)} separate CSVs → {out_dir}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _convert_to_display_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert image of any bit depth to uint8 for display purposes."""
        dtype = img.dtype
        
        if dtype == np.uint8:
            return img
        elif dtype == np.uint16:
            # Scale from [0, 65535] to [0, 255]
            return (img // 256).astype(np.uint8)
        elif dtype == np.float32 or dtype == np.float64:
            # Clamp to [0, 1] and scale to [0, 255]
            img_clipped = np.clip(img, 0, 1)
            return (img_clipped * 255).astype(np.uint8)
        else:
            # Fallback: try to normalize to [0, 255]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
                return (img_norm * 255).astype(np.uint8)
            else:
                return img.astype(np.uint8)

    def _status(self, msg: str):
        self.status_var.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ColorChartApp()
    app.geometry("1200x760")
    app.mainloop()
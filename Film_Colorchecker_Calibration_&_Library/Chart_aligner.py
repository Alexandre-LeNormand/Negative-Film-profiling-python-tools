#!/usr/bin/env python3
"""
Film Density / Log Exposure Bending Tool
=========================================
Takes real-world Cineon 10-bit float scan values of a ColorChecker 24
(6 neutral patches) and finds the best-fit position of those patches
on a manufacturer characteristic curve.

Pipeline:
  1. Convert scan Cineon floats → Density Above DMin
  2. Use the GREEN channel across all 6 patches to find X0 —
     the single log exposure ANCHOR that best aligns the measured
     densities to the emulsion curve (least-squares fit).
     The fit uses the same exposure-shifted spacing as the ideal-output
     lookup (EXPOSURE_PATCH_SPACINGS[stop]), so the exposure offset is
     encoded only once.  Green is used because it best represents
     luminance and is least affected by colour imbalances in the scan.
  3. From X0, derive the ideal log exposure for every patch:
         patch_logexp[i] = X0 + EXPOSURE_PATCH_SPACINGS[stop][i]
     For a bracketed shot the whole ladder shifts: e.g. at +1 stop the
     six patches occupy curve positions one stop higher than normal.
     X0 is the virtual anchor of whichever shifted ladder is active —
     not the absolute log exposure of middle grey (except when stop=0).
  4. Look up ideal density on each channel's emulsion curve.
  5. Convert ideal density → Cineon float output.
  6. Report: scan IN values vs ideal OUT values for DaVinci Resolve.

Cineon conversions (10-bit float, 0-1 range):
  cineon -> density :  (1023 x CV - 95) / 500
  density -> cineon :  (95 + 500 x density) / 1023

X-axis units
  Manufacturer H&D curve data comes in three different x-axis conventions.
  Each emulsion in the library stores a single "x_unit" tag, and all raw
  source values are kept exactly as digitised.  The script converts to
  log10 internally before any calculation.

  "log10"       Relative log₁₀ exposure.  1 stop = log10(2) ≈ 0.301.
                Typical range: ≈ −2.4 to +2.4 (8-stop span) or ±1.8 (6-stop).
                This is the script's internal working unit.  No conversion.
                Example: Kodak Vision3 5219 data as distributed.

  "stops"       EV / photographic stops.  1 stop = 1.0 unit.
                Typical range: −6 to +6 or −8 to +8.
                Conversion: x_log10 = x_stops × log10(2) = x_stops × 0.30103.
                Example: Fuji Eterna/Reala published H&D charts.

  "log_lux_sec" Absolute log₁₀(lux·seconds).  Same scale as "log10" but with
                an absolute photometric origin rather than a relative one.
                Typical range: −3 to +3 or −4 to +2 (varies by ISO/speed).
                No scale conversion needed — find_x0 absorbs the offset.
                Example: KODAK datasheets that specify exposure in lux·s.
"""

import json
import os
import re
import sys
import numpy as np
from scipy.optimize import minimize_scalar

# ─────────────────────────────────────────────
#  DEBUG MODE
# ─────────────────────────────────────────────
DEBUG_MODE = False

DEBUG_CINEON_VALUES = [
    (0.28453, 0.22799, 0.23154),
    (0.38708, 0.32153, 0.32505),
    (0.45538, 0.39425, 0.38327),
    (0.50653, 0.45554, 0.42925),
    (0.54069, 0.49651, 0.46772),
    (0.57175, 0.54323, 0.50510),
]

# ─────────────────────────────────────────────
#  FILE PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LIBRARY_FILE = os.path.join(SCRIPT_DIR, "emulsion_library.json")
LUT_OUTPUT_PATH = SCRIPT_DIR  # Default: script directory. Use set_lut_output_path() to change

# ─────────────────────────────────────────────
#  X-AXIS UNIT DEFINITIONS
# ─────────────────────────────────────────────
# Each emulsion stores "x_unit" alongside its curve data.
# The table below drives all user-facing menus and conversion logic.
# Keys are the canonical strings stored in the JSON library.
X_UNITS = {
    "log10": {
        "label":       "Log₁₀ exposure (relative)",
        "description": "Relative log₁₀ units.  1 stop = 0.301.  Typical range ≈ ±1.8 (6-stop) "
                       "or ±2.4 (8-stop).  This is the script's internal working unit.",
        "example":     "Kodak Vision3 datasheets (as distributed)",
    },
    "stops": {
        "label":       "Stops / EV",
        "description": "Photographic stops.  1 stop = 1.0 unit.  Typical range −6 to +6 "
                       "or −8 to +8.  Converted to log₁₀ by × log₁₀(2) = × 0.30103.",
        "example":     "Fuji Eterna / Fuji Reala published H&D charts",
    },
    "log_lux_sec": {
        "label":       "Log₁₀ lux-seconds (absolute)",
        "description": "Absolute log₁₀(lux·s).  Same scale as log₁₀ but origin is an "
                       "absolute photometric level.  Typical range −4 to +2 (varies with ISO). "
                       "No scale conversion; find_x0 absorbs the absolute offset automatically.",
        "example":     "Kodak datasheets that express exposure in lux·seconds",
    },
}

# Ordered key list for menus (stable insertion order in Python 3.7+).
X_UNIT_KEYS = list(X_UNITS.keys())   # ["log10", "stops", "log_lux_sec"]

# ─────────────────────────────────────────────
#  DEFAULT EMULSION DATA  (Kodak 5219)
# ─────────────────────────────────────────────
KODAK_5219 = {
    "name":   "Kodak Vision3 5219 500T",
    "x_unit": "log10",
    "red": {
        "x": [-4.00000, -3.68790, -3.37580, -3.07010, -2.75800, -2.43950,
               -2.12740, -1.81530, -1.50320, -1.18470, -0.86624, -0.55414,
               -0.25478,  0.05733,  0.36943,  0.70064,  0.99363],
        "y": [ 0.00000,  0.00766,  0.03065,  0.08812,  0.16475,  0.29119,
                0.42912,  0.57088,  0.70881,  0.85826,  1.00386,  1.14946,
                1.28356,  1.40616,  1.51726,  1.61306,  1.66666]
    },
    "green": {
        "x": [-4.00000, -3.68790, -3.38220, -3.07010, -2.75160, -2.44590,
               -2.13380, -1.81530, -1.50320, -1.19750, -0.87261, -0.55414,
               -0.25478,  0.03822,  0.36943,  0.68153,  0.98726],
        "y": [ 0.00000,  0.00383,  0.03065,  0.08812,  0.18007,  0.32184,
                0.49039,  0.66279,  0.82759,  0.98849,  1.16089,  1.32949,
                1.49039,  1.63219,  1.77779,  1.88889,  1.97319]
    },
    "blue": {
        "x": [-4.00000, -3.68150, -3.37580, -3.06370, -2.75160, -2.43950,
               -2.13380, -1.81530, -1.50320, -1.19110, -0.87898, -0.56051,
               -0.26752,  0.06369,  0.36943,  0.68153,  0.99363],
        "y": [ 0.00000,  0.00766,  0.02682,  0.08046,  0.16092,  0.31032,
                0.47892,  0.63982,  0.79692,  0.96172,  1.12642,  1.29502,
                1.44442,  1.60152,  1.73182,  1.82762,  1.90802]
    }
}

# Known relative log exposure spacing of the 6 ColorChecker neutral patches.
# Patch 3 (Neutral 5, middle grey) is the anchor at 0.0.
# These offsets are fixed by the chart manufacturer.
PATCH_SPACING = [-0.80, -0.30, 0.00, 0.30, 0.50, 0.70]

PATCH_LABELS = [
    "Black 2 (1.5D)",
    "Neutral 3.5 (1.05D)",
    "Neutral 5 (0.70D)",
    "Neutral 6.5 (0.44D)",
    "Neutral 8 (0.23D)",
    "White 9.5 (0.05D)"
]

# ─────────────────────────────────────────────
#  EXPOSURE OFFSET PATCH SPACING REFERENCE TABLE
# ─────────────────────────────────────────────
# Each entry shifts the base PATCH_SPACING by (stop × 0.30) log units.
# This moves the reference ladder up or down the emulsion curve to match
# the declared exposure of the chart at the time of shooting.
# Range: -5 stops (severe underexposure) to +5 stops (severe overexposure).
# Key = integer stop value, value = 6-element spacing list.
EXPOSURE_PATCH_SPACINGS = {
    -5: [-2.30, -1.80, -1.50, -1.20, -1.00, -0.80],
    -4: [-2.00, -1.50, -1.20, -0.90, -0.70, -0.50],
    -3: [-1.70, -1.20, -0.90, -0.60, -0.40, -0.20],
    -2: [-1.40, -0.90, -0.60, -0.30, -0.10,  0.10],
    -1: [-1.10, -0.60, -0.30,  0.00,  0.20,  0.40],
     0: [-0.80, -0.30,  0.00,  0.30,  0.50,  0.70],
    +1: [-0.50,  0.00,  0.30,  0.60,  0.80,  1.00],
    +2: [-0.20,  0.30,  0.60,  0.90,  1.10,  1.30],
    +3: [ 0.10,  0.60,  0.90,  1.20,  1.40,  1.60],
    +4: [ 0.40,  0.90,  1.20,  1.50,  1.70,  1.90],
    +5: [ 0.70,  1.20,  1.50,  1.80,  2.00,  2.20],
}

# ─────────────────────────────────────────────
#  LIBRARY MANAGEMENT
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  X-AXIS UNIT: CONVERSION + INTERACTIVE HELPERS
# ─────────────────────────────────────────────

import math as _math
_LOG10_2 = _math.log10(2)   # ≈ 0.30103


def x_unit_label(x_unit: str) -> str:
    """Short display label for a unit key, e.g. 'Stops / EV'."""
    return X_UNITS.get(x_unit, {}).get("label", x_unit)


def curve_to_log10(curve: dict, x_unit: str) -> dict:
    """
    Return a copy of *curve* with x values converted to log₁₀ relative units.

    "log10"       → no-op (already in internal format)
    "log_lux_sec" → no-op (same scale; absolute offset absorbed by find_x0)
    "stops"       → multiply each x by log10(2) = 0.30103
    """
    if x_unit in ("log10", "log_lux_sec"):
        return curve          # already correct scale; share the dict (read-only use)
    if x_unit == "stops":
        return {"x": [v * _LOG10_2 for v in curve["x"]], "y": curve["y"]}
    raise ValueError(f"Unknown x_unit '{x_unit}'. Expected one of {X_UNIT_KEYS}.")


def emulsion_curves_to_log10(emulsion: dict) -> dict:
    """
    Return a shallow copy of *emulsion* where all three channel curves have
    their x values converted to log₁₀.  The original library entry is not
    modified — raw source values are always preserved in storage.
    """
    x_unit = emulsion.get("x_unit", "log10")
    return {
        **emulsion,
        "red":   curve_to_log10(emulsion["red"],   x_unit),
        "green": curve_to_log10(emulsion["green"], x_unit),
        "blue":  curve_to_log10(emulsion["blue"],  x_unit),
    }


def ask_x_unit(current: str | None = None) -> str:
    """
    Interactive prompt: let the user choose which x-axis unit their source
    data uses.  Returns one of the canonical X_UNIT_KEYS strings.

    If *current* is provided it is highlighted as the active value so the
    user can press Enter to leave it unchanged.
    """
    print("\n  -- X-Axis Unit --")
    print("  Which unit does the manufacturer use for the x-axis of the H&D curve?\n")
    for i, key in enumerate(X_UNIT_KEYS, 1):
        info   = X_UNITS[key]
        marker = " ◀ current" if key == current else ""
        print(f"  {i}. {info['label']}{marker}")
        print(f"     {info['description']}")
        print(f"     Example: {info['example']}\n")
    default_idx = (X_UNIT_KEYS.index(current) + 1) if current in X_UNIT_KEYS else None
    prompt = f"  Select unit [1-{len(X_UNIT_KEYS)}]"
    if default_idx is not None:
        prompt += f" (Enter = keep '{x_unit_label(current)}')"
    prompt += ": "
    while True:
        raw = input(prompt).strip()
        if raw == "" and default_idx is not None:
            return current
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(X_UNIT_KEYS):
                chosen = X_UNIT_KEYS[idx]
                print(f"  ✓ x-axis unit set to: {x_unit_label(chosen)}")
                return chosen
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(X_UNIT_KEYS)}.")


def _detect_x_unit_heuristic(emulsion: dict) -> str:
    """
    Best-guess x_unit for a legacy emulsion that lacks the field.

    Decision rule — applied to the span (max − min) of all channel x values:

      span > 6.0  →  "stops"
        In log₁₀ units a span of 6.0 would equal 6 / log₁₀(2) ≈ 20 stops of
        measured range.  No calibrated film sensitometry extends that far,
        so any curve with a span that wide is almost certainly in stops/EV.
        Typical stops-based charts: span = 12 (−6 to +6) or 16 (−8 to +8).

      span ≤ 6.0  →  "log10"
        A 6-stop measurement in log₁₀ gives span ≈ 1.81; an 8-stop span ≈ 2.41.
        Even generous manufacturer data (Kodak 5219: −4.0 to +1.0, span = 5.0)
        stays well under 6.0 when the axis is already in log₁₀.
        "log_lux_sec" has identical scale to "log10" and cannot be
        distinguished heuristically — the user is prompted to confirm.
    """
    all_x = []
    for ch in ("red", "green", "blue"):
        all_x.extend(emulsion.get(ch, {}).get("x", []))
    if not all_x:
        return "log10"
    span = max(all_x) - min(all_x)
    return "stops" if span > 6.0 else "log10"


def _migrate_missing_xunits(lib: dict) -> bool:
    """
    Inspect every emulsion that lacks an "x_unit" field.
    Auto-detect the most likely unit, print a clear explanation, and ask
    the user to confirm or correct.  Updates lib in place.
    Returns True if any entries were migrated (caller should save).
    """
    needs_migration = [
        name for name, em in lib.items() if "x_unit" not in em
    ]
    if not needs_migration:
        return False

    print("\n  *** LIBRARY MIGRATION: X-Axis Units ***")
    print("  The following emulsions were saved before the x_unit field was")
    print("  introduced.  Please confirm the correct unit for each one.")
    print("  (Wrong unit → flat/incorrect LUTs.  You can change it later via Edit.)\n")

    for name in needs_migration:
        em     = lib[name]
        guess  = _detect_x_unit_heuristic(em)
        all_x  = []
        for ch in ("red", "green", "blue"):
            all_x.extend(em.get(ch, {}).get("x", []))
        x_min  = min(all_x); x_max = max(all_x)

        print(f"  Emulsion : {name}")
        print(f"  X range  : {x_min:.4f}  to  {x_max:.4f}")
        print(f"  Auto-detected as: {x_unit_label(guess)}")
        if guess == "stops":
            converted_range = max(abs(x_min), abs(x_max)) * _LOG10_2
            print(f"  (Converting would give log₁₀ range ≈ ±{converted_range:.3f}, "
                  f"= ±{converted_range/_LOG10_2:.1f} stops — looks correct)")

        chosen = ask_x_unit(current=guess)
        em["x_unit"] = chosen
        print()

    return True


# ─────────────────────────────────────────────
#  LIBRARY MANAGEMENT
# ─────────────────────────────────────────────

def load_library() -> dict:
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "r") as f:
            lib = json.load(f)
        if "Kodak Vision3 5219 500T" not in lib:
            lib["Kodak Vision3 5219 500T"] = KODAK_5219
            save_library(lib)
        # Migrate any legacy entries that predate the x_unit field
        if _migrate_missing_xunits(lib):
            save_library(lib)
        return lib
    else:
        lib = {"Kodak Vision3 5219 500T": KODAK_5219}
        save_library(lib)
        return lib


def save_library(lib: dict):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(lib, f, indent=2)
    print(f"  [Library saved → {LIBRARY_FILE}]")


def list_emulsions(lib: dict):
    print("\n  Emulsions in library:")
    for i, (name, em) in enumerate(lib.items(), 1):
        unit_tag = x_unit_label(em.get("x_unit", "?? (no unit)"))
        print(f"    {i}. {name}  [{unit_tag}]")


def pick_emulsion(lib: dict) -> tuple[str, dict]:
    list_emulsions(lib)
    names = list(lib.keys())
    while True:
        try:
            idx = int(input("\n  Select emulsion number: ")) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                return name, lib[name]
        except ValueError:
            pass
        print("  Invalid selection, try again.")


def add_emulsion(lib: dict):
    print("\n-- Add New Emulsion --")
    name = input("  Emulsion name: ").strip()
    if not name:
        print("  Aborted (empty name).")
        return
    if name in lib:
        print(f"  '{name}' already exists. Edit it instead.")
        return

    # Ask for the x-axis unit ONCE, before entering any channel data.
    # All three channels of one emulsion share the same x-axis convention.
    x_unit = ask_x_unit()

    data = {"name": name, "x_unit": x_unit}
    unit_note = x_unit_label(x_unit)
    for channel in ("red", "green", "blue"):
        print(f"\n  === {channel.upper()} channel ===")
        print(f"  You can enter values one per line, or paste all 17 at once.\n")
        print(f"  Enter 17 X values for {channel.upper()}  ({unit_note}):")
        xs = _read_17_floats()
        print(f"  Enter 17 Y (Density Above DMin) values for {channel.upper()}:")
        ys = _read_17_floats()
        data[channel] = {"x": xs, "y": ys}
    lib[name] = data
    save_library(lib)
    print(f"  '{name}' added.")


def edit_emulsion(lib: dict):
    print("\n-- Edit Emulsion --")
    name, emulsion = pick_emulsion(lib)
    current_unit = emulsion.get("x_unit", "log10")
    print(f"\n  Editing '{name}'  [x_unit: {x_unit_label(current_unit)}]")
    print("  What would you like to edit?")
    print("  1=Red channel   2=Green channel   3=Blue channel")
    print("  4=Rename        5=Change x_unit   0=Cancel")
    choice = input("  > ").strip()
    ch_map = {"1": "red", "2": "green", "3": "blue"}

    if choice in ch_map:
        ch = ch_map[choice]
        unit_note = x_unit_label(current_unit)
        print(f"\n  You can enter values one per line, or paste all 17 at once.")
        print(f"  (If the unit has changed, use option 5 first to update x_unit.)\n")
        print(f"  Enter 17 NEW X values for {ch.upper()}  ({unit_note}):")
        xs = _read_17_floats()
        print(f"  Enter 17 NEW Y values for {ch.upper()}:")
        ys = _read_17_floats()
        lib[name][ch] = {"x": xs, "y": ys}
        save_library(lib)
        print(f"  {ch.upper()} channel updated.")

    elif choice == "4":
        new_name = input("  New name: ").strip()
        if new_name and new_name not in lib:
            lib[new_name] = lib.pop(name)
            save_library(lib)
            print(f"  Renamed to '{new_name}'.")
        else:
            print("  Invalid name or already exists.")

    elif choice == "5":
        print(f"\n  Current x_unit: {x_unit_label(current_unit)}")
        new_unit = ask_x_unit(current=current_unit)
        if new_unit != current_unit:
            lib[name]["x_unit"] = new_unit
            save_library(lib)
            print(f"  x_unit updated to: {x_unit_label(new_unit)}")
            print(f"  Note: stored x values are NOT rescaled — they remain as entered.")
            print(f"  The new unit tag tells the script how to interpret them at runtime.")
        else:
            print("  Unit unchanged.")

    else:
        print("  Cancelled.")


def delete_emulsion(lib: dict):
    print("\n-- Delete Emulsion --")
    name, _ = pick_emulsion(lib)
    if name == "Kodak Vision3 5219 500T":
        print("  Cannot delete the default emulsion.")
        return
    confirm = input(f"  Delete '{name}'? (yes/no): ").strip().lower()
    if confirm == "yes":
        del lib[name]
        save_library(lib)
        print(f"  '{name}' deleted.")
    else:
        print("  Cancelled.")


def _read_17_floats() -> list:
    """Read exactly 17 float values.

    Accepts input in two ways:
      1. One value per line (classic mode).
      2. All 17 values pasted at once, separated by newlines, spaces,
         tabs, commas (as separators), or semicolons.
    """
    values: list[float] = []
    while len(values) < 17:
        prompt = f"    [{len(values)+1}/17]: "
        raw = input(prompt).strip()
        if not raw:
            continue
        # Split on newlines, tabs, spaces, or semicolons
        tokens = re.split(r'[\n\r\t;\s]+', raw)
        parsed_count = 0
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            # Replace comma with dot for decimal parsing
            tok = tok.replace(",", ".")
            try:
                value = float(tok)
                values.append(value)
                parsed_count += 1
            except ValueError:
                # Skip invalid tokens silently
                pass
            if len(values) >= 17:
                break
        remaining = 17 - len(values)
        if remaining > 0 and parsed_count > 1:
            # User pasted multiple values
            print(f"    Got {len(values)} so far, {remaining} more needed.")
    if len(values) > 17:
        print(f"    (Received {len(values)} values, using first 17.)")
        values = values[:17]
    return values


def _read_single_float(prompt: str) -> float:
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("    Not a number, try again.")

# ─────────────────────────────────────────────
#  CINEON <-> DENSITY CONVERSION
# ─────────────────────────────────────────────

def cineon_to_density(cv: float) -> float:
    """10-bit float (0-1) -> Density Above DMin.
    Cineon integer: CV_int = 95 + 500 x density
    Solving for density with CV_int = CV_float x 1023:
        density = (1023 x CV_float - 95) / 500
    """
    return (1023.0 * cv - 95.0) / 500.0


def density_to_cineon(density: float) -> float:
    """Density Above DMin -> 10-bit float (0-1).
        CV_float = (95 + 500 x density) / 1023
    """
    return (95.0 + 500.0 * density) / 1023.0

# ─────────────────────────────────────────────
#  CORE INTERPOLATION
# ─────────────────────────────────────────────

def log_exposure_to_density(log_exp: float, curve: dict) -> float:
    """Forward: Log Exposure -> Density on the emulsion curve."""
    return float(np.interp(log_exp, curve["x"], curve["y"]))

# ─────────────────────────────────────────────
#  BEST-FIT ANCHOR FINDER
# ─────────────────────────────────────────────

def find_x0(green_densities: list, green_curve: dict,
            patch_spacing: list | None = None) -> float:
    """
    Find X0 — the virtual log-exposure anchor of the reference ladder —
    by minimising the sum of squared differences between the 6 measured
    green densities and the emulsion curve predictions at X0 + spacing[i].

    patch_spacing must be the SAME spacing array used for the ideal-output
    lookup in run_calculation.  Keeping both steps on the same ladder is
    critical: if find_x0 used base PATCH_SPACING while run_calculation used
    an exposure-shifted version, the stop offset would be double-counted,
    pushing every ideal output by an extra stop in the wrong direction.

    When patch_spacing is None the base PATCH_SPACING is used (0-stop case).

    Uses only the green channel as it best represents luminance
    and is least affected by colour imbalances in the scan.
    """
    if patch_spacing is None:
        patch_spacing = PATCH_SPACING
    spacings = np.array(patch_spacing)
    measured = np.array(green_densities)

    def residuals(x0):
        predicted = np.array([
            log_exposure_to_density(x0 + s, green_curve)
            for s in spacings
        ])
        return float(np.sum((predicted - measured) ** 2))

    result = minimize_scalar(residuals, bounds=(-4.0, 2.0), method="bounded")
    return float(result.x)

# ─────────────────────────────────────────────
#  EXPOSURE STOP SELECTION
# ─────────────────────────────────────────────

def ask_exposure_stop() -> int:
    """Ask the user how the chart was exposed relative to normal.
    Returns an integer in the range [-5, +5]. Defaults to 0 (normal).
    """
    print("\n-- Chart Exposure --")
    print("  How was the chart exposed when shot?")
    print("  Enter an integer stop value from -5 (underexposed) to +5 (overexposed).")
    print("  Press Enter for normal exposure (0).")
    while True:
        raw = input("  Exposure offset in stops [0]: ").strip()
        if raw == "":
            return 0
        try:
            val = int(raw)
            if -5 <= val <= 5:
                return val
        except ValueError:
            pass
        print("  Please enter a whole number between -5 and +5.")


def set_lut_output_path():
    """Allow the user to define or change the default LUT output directory."""
    global LUT_OUTPUT_PATH
    print("\n-- Set LUT Output Path --")
    print(f"  Current output path: {LUT_OUTPUT_PATH}")
    print("  Enter a new directory path (or press Enter to keep current):")
    raw = input("  > ").strip().strip("'\"")
    if raw == "":
        print("  Path unchanged.")
        return
    if os.path.isdir(raw):
        LUT_OUTPUT_PATH = raw
        print(f"  LUT output path set to: {LUT_OUTPUT_PATH}")
    else:
        print(f"  Invalid: '{raw}' is not a valid directory.")


def get_channel_dmax(curve: dict) -> float:
    """Return the maximum density (DMax) defined in the emulsion curve."""
    return float(max(curve["y"]))


# ─────────────────────────────────────────────
#  MAIN CALCULATION
# ─────────────────────────────────────────────

def run_calculation(patches_cineon: list, emulsion: dict, exposure_stop: int = 0) -> tuple[list, float]:
    """
    1. Convert emulsion curves to log₁₀ x-axis (respecting x_unit tag).
    2. Convert scan Cineon floats -> density for all patches/channels.
    3. Fit X0 using green channel least-squares.
    4. For each patch: ideal log exposure = X0 + spacing[i]  (spacing chosen
       from EXPOSURE_PATCH_SPACINGS for the declared exposure_stop).
    5. Look up ideal density on each channel's emulsion curve.
       If the reference exceeds DMax, clamp to DMax and flag as clipped.
    6. Convert ideal density -> Cineon float output.
    Returns (results list, fitted X0).
    """
    patch_spacing = EXPOSURE_PATCH_SPACINGS[exposure_stop]

    # Step 0 — convert stored x values to log₁₀ (no-op for "log10" / "log_lux_sec")
    em = emulsion_curves_to_log10(emulsion)

    # DMax per channel — ceiling for clamping
    r_dmax = get_channel_dmax(em["red"])
    g_dmax = get_channel_dmax(em["green"])
    b_dmax = get_channel_dmax(em["blue"])

    # Step 1 — all densities
    densities = [
        (cineon_to_density(r), cineon_to_density(g), cineon_to_density(b))
        for r, g, b in patches_cineon
    ]

    # Step 2 — find X0 from green channel.
    # IMPORTANT: pass the same patch_spacing that will be used for ideal-output
    # lookup below.  Both steps must use the same ladder so the exposure-stop
    # offset is not double-counted (once by the shifted fit, once by the shifted
    # lookup), which was the source of the one-stop output error.
    green_densities = [d[1] for d in densities]
    x0 = find_x0(green_densities, em["green"], patch_spacing)

    # Steps 3-5 — per patch
    results = []
    for i, ((r_cin, g_cin, b_cin), (r_den, g_den, b_den)) in enumerate(
        zip(patches_cineon, densities)
    ):
        patch_logexp = x0 + patch_spacing[i]

        r_raw = log_exposure_to_density(patch_logexp, em["red"])
        g_raw = log_exposure_to_density(patch_logexp, em["green"])
        b_raw = log_exposure_to_density(patch_logexp, em["blue"])

        r_clipped = r_raw >= r_dmax
        g_clipped = g_raw >= g_dmax
        b_clipped = b_raw >= b_dmax

        r_ideal_den = min(r_raw, r_dmax)
        g_ideal_den = min(g_raw, g_dmax)
        b_ideal_den = min(b_raw, b_dmax)

        results.append({
            "patch":        i + 1,
            "label":        PATCH_LABELS[i],
            "patch_logexp": patch_logexp,
            "cin_in":       (r_cin,       g_cin,       b_cin),
            "den_in":       (r_den,       g_den,       b_den),
            "den_ideal":    (r_ideal_den, g_ideal_den, b_ideal_den),
            "cin_ideal":    (density_to_cineon(r_ideal_den),
                             density_to_cineon(g_ideal_den),
                             density_to_cineon(b_ideal_den)),
            "clipped":      (r_clipped,   g_clipped,   b_clipped),
        })

    return results, x0

# ─────────────────────────────────────────────
#  1D LUT EXPORT  (.cube)
# ─────────────────────────────────────────────

LUT_SIZE = 4096   # standard for 1-D correction LUTs; change here if needed

def _build_channel_interpolator(control_points: list[tuple[float, float]]):
    """
    Build a PCHIP (monotone cubic) interpolator for one channel from a list
    of (cin_in, cin_ideal) control points.

    Two hard anchors are added automatically:
      • (0.0, 0.0) — below the darkest measured patch → identity
      • (1.0, 1.0) — above the brightest measured patch → identity

    This keeps the LUT as identity outside the range where we have actual
    measurements, which is the safest assumption for values never seen on
    the chart.  PCHIP guarantees monotonicity and passes exactly through
    every control point, mirroring the behaviour of Resolve's curve tool.
    """
    from scipy.interpolate import PchipInterpolator

    # Sort by input value (should already be ascending patch 1→6)
    pts = sorted(control_points, key=lambda p: p[0])

    # Deduplicate identical x values (e.g. two channels clamped to the same
    # Cineon value at DMax) by keeping the first occurrence
    seen = set()
    deduped = []
    for x, y in pts:
        if x not in seen:
            deduped.append((x, y))
            seen.add(x)

    xs = [0.0] + [p[0] for p in deduped] + [1.0]
    ys = [0.0] + [p[1] for p in deduped] + [1.0]
    return PchipInterpolator(xs, ys, extrapolate=True)


def export_1d_lut(
    results:       list,
    emulsion_name: str,
    exposure_stop: int,
    output_path:   str,
    lut_size:      int = LUT_SIZE,
) -> bool:
    """
    Derive a 3-channel 1D correction LUT from the bending results and write
    it as a .cube file.

    Curve construction
    ------------------
    For each channel (R, G, B) independently, the 6 (cin_in, cin_ideal)
    pairs become control points for a PCHIP interpolator with identity
    anchors at 0 and 1.  The interpolator is sampled at lut_size evenly
    spaced input values covering [0, 1].

    The resulting LUT replicates exactly what you would do in Resolve's
    colour curves tool: one node per patch, IN = scan value, OUT = ideal
    emulsion value, with a smooth monotone spline between nodes.

    .cube format
    ------------
    1D cube: LUT_1D_SIZE lines, each "R G B", where entry i maps
    input level i/(N-1) independently on each channel.

    Returns True on success, False on write error.
    """
    stop_label = (f"+{exposure_stop}" if exposure_stop > 0 else str(exposure_stop))
    exposure_desc = (
        "Normal exposure" if exposure_stop == 0
        else f"{stop_label} stop{'s' if abs(exposure_stop) > 1 else ''} "
             f"{'over' if exposure_stop > 0 else 'under'}exposed"
    )

    # Build per-channel control-point lists
    ctrl_r = [(r["cin_in"][0], r["cin_ideal"][0]) for r in results]
    ctrl_g = [(r["cin_in"][1], r["cin_ideal"][1]) for r in results]
    ctrl_b = [(r["cin_in"][2], r["cin_ideal"][2]) for r in results]

    interp_r = _build_channel_interpolator(ctrl_r)
    interp_g = _build_channel_interpolator(ctrl_g)
    interp_b = _build_channel_interpolator(ctrl_b)

    t = np.linspace(0.0, 1.0, lut_size)
    lut_r = np.clip(interp_r(t), 0.0, 1.0)
    lut_g = np.clip(interp_g(t), 0.0, 1.0)
    lut_b = np.clip(interp_b(t), 0.0, 1.0)

    # Sanitise emulsion name for the TITLE field (no quotes)
    title_str = f"{emulsion_name} | {exposure_desc} | 1D Correction"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # ── Header ────────────────────────────────────────────────
            f.write(f'TITLE "{title_str}"\n')
            f.write(f"# Generated by Chart Aligner\n")
            f.write(f"# Emulsion : {emulsion_name}\n")
            f.write(f"# Exposure : {exposure_desc}\n")
            f.write(f"# Control points per channel (cin_in -> cin_ideal):\n")
            for ch_label, ctrl in [("R", ctrl_r), ("G", ctrl_g), ("B", ctrl_b)]:
                pts_str = "  ".join(f"{xi:.5f}->{yi:.5f}" for xi, yi in ctrl)
                f.write(f"#   {ch_label}: {pts_str}\n")
            f.write(f"# Identity anchors added at 0.0->0.0 and 1.0->1.0\n")
            f.write(f"# Interpolation: PCHIP monotone cubic\n")
            f.write(f"\n")
            f.write(f"LUT_1D_SIZE {lut_size}\n")
            f.write(f"\n")
            # ── LUT data ───────────────────────────────────────────────
            for r_val, g_val, b_val in zip(lut_r, lut_g, lut_b):
                f.write(f"{r_val:.6f} {g_val:.6f} {b_val:.6f}\n")
    except OSError as exc:
        print(f"  [LUT] Write error: {exc}")
        return False

    return True


def ask_lut_export(
    results:       list,
    emulsion_name: str,
    exposure_stop: int,
):
    """
    Prompt the user for LUT export after a calculation run.
    Suggests a default filename derived from the emulsion name and stop value.
    Uses LUT_OUTPUT_PATH as the default directory.
    """
    print("\n-- Export 1D LUT (.cube) --")
    print("  Export a 1D correction LUT from these results? (y/n) [y]: ", end="")
    ans = input().strip().lower()
    if ans in ("n", "no"):
        return

    # Build a safe default filename
    stop_label  = (f"+{exposure_stop}" if exposure_stop > 0 else str(exposure_stop))
    safe_name   = emulsion_name.replace(" ", "_").replace("/", "-")
    default_name = f"{safe_name}_{stop_label}EV_1D.cube"

    print(f"\n  Output path (press Enter for '{default_name}' in {LUT_OUTPUT_PATH}):")
    raw = input("  > ").strip().strip("'\"")
    if raw == "":
        output_path = os.path.join(LUT_OUTPUT_PATH, default_name)
    elif os.path.isdir(raw):
        output_path = os.path.join(raw, default_name)
    else:
        # If user gave a name without extension, append .cube
        if not raw.lower().endswith(".cube"):
            raw += ".cube"
        output_path = raw

    lut_size = LUT_SIZE
    print(f"\n  LUT size [{lut_size}] (press Enter to keep, or type a power-of-2 value): ", end="")
    size_raw = input().strip()
    if size_raw:
        try:
            candidate = int(size_raw)
            if candidate >= 2:
                lut_size = candidate
            else:
                print("  Value too small, keeping default.")
        except ValueError:
            print("  Not a number, keeping default.")

    print(f"\n  Writing {lut_size}-entry 1D LUT …")
    ok = export_1d_lut(results, emulsion_name, exposure_stop, output_path, lut_size)
    if ok:
        print(f"  [LUT saved -> {output_path}]")
    else:
        print("  LUT export failed.")


# ─────────────────────────────────────────────
#  OUTPUT / DISPLAY
# ─────────────────────────────────────────────

SEP  = "-" * 110
SEP2 = "=" * 110

def _fmt3(t, dp=4):
    fmt = f"{{:9.{dp}f}}"
    return f"{fmt.format(t[0])}  {fmt.format(t[1])}  {fmt.format(t[2])}"


def print_results(results: list, emulsion_name: str, x0: float,
                  exposure_stop: int = 0, x_unit: str = "log10"):
    stop_label = (f"+{exposure_stop}" if exposure_stop > 0 else str(exposure_stop))
    exposure_desc = "Normal" if exposure_stop == 0 else f"{stop_label} stop{'s' if abs(exposure_stop) > 1 else ''} {'over' if exposure_stop > 0 else 'under'}exposed"

    print(f"\n{SEP2}")
    print(f"  RESULTS  |  Emulsion: {emulsion_name}  |  Chart Exposure: {exposure_desc}  |  x_unit: {x_unit_label(x_unit)}")
    print(SEP2)

    print(f"\n  Best-fit anchor X0 = {x0:+.5f}")
    if exposure_stop == 0:
        print(f"  (fitted log exposure of the Neutral 5 / middle-grey patch on this scan)")
    else:
        anchor_offset = EXPOSURE_PATCH_SPACINGS[exposure_stop][2]
        midgrey_logexp = x0 + anchor_offset
        print(f"  (virtual ladder anchor; middle-grey patch sits at X0 {anchor_offset:+.2f} = {midgrey_logexp:+.5f} log exp)")
        print(f"  Reference ladder shifted {stop_label} stop{'s' if abs(exposure_stop) > 1 else ''}: "
              f"find_x0 and ideal lookup both use the same shifted spacings {EXPOSURE_PATCH_SPACINGS[exposure_stop]}")
    print()

    # Table 1: Scan input
    print(f"  TABLE 1 — Scan Input: Cineon float values & Density Above DMin")
    print(f"  {SEP}")
    print(f"  {'Patch':<26} {'Cineon In  (R / G / B)':^35}  {'Density In  (R / G / B)':^35}")
    print(f"  {SEP}")
    for r in results:
        print(f"  {r['label']:<26} {_fmt3(r['cin_in'], dp=5)}    {_fmt3(r['den_in'])}")

    # Table 2: Ideal output
    print(f"\n\n  TABLE 2 — Ideal Output: emulsion curve at best-fit log exposures")
    print(f"  (* = clamped to DMax — no film information beyond this point)")
    print(f"  {SEP}")
    print(f"  {'Patch':<26} {'LogExp':>9}  {'Ideal Density  (R / G / B)':^35}  {'Ideal Cineon  (R / G / B)':^35}")
    print(f"  {SEP}")
    for r in results:
        clip_flag = "*" if any(r["clipped"]) else " "
        print(f"  {r['label']:<26} {r['patch_logexp']:>+9.4f}  "
              f"{_fmt3(r['den_ideal'])}    {_fmt3(r['cin_ideal'], dp=5)} {clip_flag}")

    # Bending Summary
    print(f"\n\n  BENDING SUMMARY — IN (scan) / OUT (ideal) for DaVinci Resolve")
    print(f"  Positive delta = scan reads too dark vs. ideal curve.")
    print(f"  (* = channel clamped to DMax)")
    print(f"  {SEP}")
    print(f"  {'Patch':<26}  {'Scan R':>9}  {'Ideal R':>9}  {'DR':>8}   "
          f"{'Scan G':>9}  {'Ideal G':>9}  {'DG':>8}   "
          f"{'Scan B':>9}  {'Ideal B':>9}  {'DB':>8}")
    print(f"  {SEP}")
    for r in results:
        dr = r["cin_ideal"][0] - r["cin_in"][0]
        dg = r["cin_ideal"][1] - r["cin_in"][1]
        db = r["cin_ideal"][2] - r["cin_in"][2]
        rc = "*" if r["clipped"][0] else " "
        gc = "*" if r["clipped"][1] else " "
        bc = "*" if r["clipped"][2] else " "
        print(
            f"  {r['label']:<26} "
            f"{r['cin_in'][0]:>9.5f}  {r['cin_ideal'][0]:>9.5f}{rc} {dr:>+8.5f}   "
            f"{r['cin_in'][1]:>9.5f}  {r['cin_ideal'][1]:>9.5f}{gc} {dg:>+8.5f}   "
            f"{r['cin_in'][2]:>9.5f}  {r['cin_ideal'][2]:>9.5f}{bc} {db:>+8.5f}"
        )

    print(f"\n{SEP2}\n")

# ─────────────────────────────────────────────
#  USER INPUT
# ─────────────────────────────────────────────

def input_cineon_values() -> list[tuple[float, float, float]]:
    print("\n-- Real-World Scan Data --")
    print("  Enter Cineon 10-bit float values for the 6 neutral patches")
    print("  of the ColorChecker 24 (darkest -> lightest).\n")
    patches = []
    for i, label in enumerate(PATCH_LABELS, 1):
        print(f"  Patch {i}: {label}")
        r = _read_single_float("    R")
        g = _read_single_float("    G")
        b = _read_single_float("    B")
        patches.append((r, g, b))
        print()
    return patches


def get_debug_cineon_values() -> list[tuple[float, float, float]]:
    print("\n-- Using Predefined Debug Values --")
    for i, (r, g, b) in enumerate(DEBUG_CINEON_VALUES, 1):
        print(f"  Patch {i}: {PATCH_LABELS[i-1]}  R={r}  G={g}  B={b}")
    print()
    return DEBUG_CINEON_VALUES


def import_csv(path: str) -> list[tuple[float, float, float]] | None:
    """
    Load patch data from a headerless CSV file.

    Expected layout
    ---------------
    - 6 rows, one per patch, ordered BRIGHTEST first -> DARKEST last
      (matches the natural export order of most measurement tools).
    - 3 comma-separated float columns per row: R, G, B Cineon values.
    - Blank lines and lines starting with '#' are ignored.

    The rows are reversed internally so the script receives them in its
    required order: darkest (patch 1 / Black 2) -> lightest (patch 6 / White 9.5).

    Returns the 6-element list of (R, G, B) tuples, or None on any error.
    """
    import csv

    rows = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for lineno, row in enumerate(reader, 1):
                # Skip blank lines and comments
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if row[0].strip().startswith("#"):
                    continue
                if len(row) < 3:
                    print(f"  [CSV] Line {lineno}: expected 3 columns, got {len(row)} — skipping.")
                    continue
                try:
                    r = float(row[0].strip().replace(",", "."))
                    g = float(row[1].strip().replace(",", "."))
                    b = float(row[2].strip().replace(",", "."))
                except ValueError as exc:
                    print(f"  [CSV] Line {lineno}: cannot parse float — {exc}")
                    return None
                rows.append((r, g, b))
    except FileNotFoundError:
        print(f"  [CSV] File not found: {path}")
        return None
    except OSError as exc:
        print(f"  [CSV] Cannot open file: {exc}")
        return None

    if len(rows) != 6:
        print(f"  [CSV] Expected exactly 6 data rows, found {len(rows)}.")
        return None

    # CSV is brightest->darkest; script expects darkest->brightest
    patches = list(reversed(rows))

    print(f"\n  Loaded from: {path}")
    print(f"  {'Patch':<26} {'R':>11}  {'G':>11}  {'B':>11}")
    print(f"  {'-' * 62}")
    for i, (r, g, b) in enumerate(patches):
        print(f"  {PATCH_LABELS[i]:<26} {r:>11.7f}  {g:>11.7f}  {b:>11.7f}")
    print()

    return patches


def ask_data_source() -> list[tuple[float, float, float]]:
    """
    Ask the user whether to enter patch values manually or import a CSV.
    Loops until valid data is obtained.
    """
    print("\n-- Patch Data Source --")
    print("  1  Enter values manually")
    print("  2  Import from CSV file")
    while True:
        choice = input("\n  > ").strip()
        if choice == "1":
            return input_cineon_values()
        if choice == "2":
            while True:
                raw = input("\n  CSV file path: ").strip()
                # Strip surrounding quotes that some shells or drag-and-drop add
                path = raw.strip("'\"")
                patches = import_csv(path)
                if patches is not None:
                    return patches
                retry = input("  Try a different path? (y/n) [y]: ").strip().lower()
                if retry in ("n", "no"):
                    print("  Falling back to manual entry.")
                    return input_cineon_values()
        print("  Please enter 1 or 2.")

# ─────────────────────────────────────────────
#  BATCH BRACKET LUT EXPORT
# ─────────────────────────────────────────────

def batch_bracket_luts(lib: dict,
                       emulsion_name: str | None = None,
                       emulsion: dict | None = None):
    """
    Batch-generate 1D correction LUTs for an exposure bracket set.

    The user provides a folder containing CSV files (one per exposure stop).
    Files are sorted alphabetically/numerically:
      • file with lowest sort order  →  most under-exposed stop
      • file with highest sort order →  most over-exposed stop

    The stop range is -5 to +5 (11 stops max).  The user can confirm or
    override the starting stop so the bracket can be placed anywhere in
    the -5 … +5 window.

    If emulsion_name / emulsion are already provided (called from the
    Run calculation flow), the emulsion picker is skipped.
    """
    print("\n-- Batch Bracket LUTs --")
    if emulsion is None:
        print("  Select emulsion for all bracket LUTs:")
        emulsion_name, emulsion = pick_emulsion(lib)

    # ── Ask for CSV folder ───────────────────────────────────────────────
    print("\n  Enter path to folder containing bracket CSV files:")
    raw = input("  > ").strip().strip("'\"")
    if not os.path.isdir(raw):
        print(f"  Not a valid directory: '{raw}'")
        return
    csv_dir = raw

    # ── Collect and sort CSV files ───────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv")
    )
    if not all_files:
        print(f"  No .csv files found in: {csv_dir}")
        return

    n = len(all_files)
    print(f"\n  Found {n} CSV file(s):")
    for i, fn in enumerate(all_files):
        print(f"    {i+1:>2}.  {fn}")

    # ── Determine stop mapping ───────────────────────────────────────────
    STOP_MIN, STOP_MAX = -5, 5
    max_count = STOP_MAX - STOP_MIN + 1  # 11

    if n > max_count:
        print(f"\n  Warning: more than {max_count} files found. "
              f"Only the first {max_count} will be processed.")
        all_files = all_files[:max_count]
        n = max_count

    # Default: centre symmetrically, bias toward underexposure on even counts
    default_start = max(STOP_MIN, -(n // 2))
    if default_start + n - 1 > STOP_MAX:
        default_start = STOP_MAX - n + 1

    default_end = default_start + n - 1
    def _stop_label(s):
        return f"+{s}" if s > 0 else str(s)

    print(f"\n  Default stop assignment:")
    print(f"    file[1] = {_stop_label(default_start)} EV  …  "
          f"file[{n}] = {_stop_label(default_end)} EV")
    print(f"  Press Enter to accept, or type a different starting stop "
          f"[{STOP_MIN} … {STOP_MAX - n + 1}]:")
    while True:
        raw_stop = input("  > ").strip()
        if raw_stop == "":
            start_stop = default_start
            break
        try:
            candidate = int(raw_stop)
            if STOP_MIN <= candidate <= STOP_MAX - n + 1:
                start_stop = candidate
                break
            print(f"  Must be between {STOP_MIN} and {STOP_MAX - n + 1}.")
        except ValueError:
            print("  Please enter a whole number.")

    stop_assignments = list(range(start_stop, start_stop + n))
    print(f"\n  Assignment confirmed:")
    for fn, stop in zip(all_files, stop_assignments):
        print(f"    {_stop_label(stop):>3} EV  ←  {fn}")

    # ── Ask for output directory ─────────────────────────────────────────
    print(f"\n  Output directory (press Enter for: {LUT_OUTPUT_PATH}):")
    raw_out = input("  > ").strip().strip("'\"")
    if raw_out == "":
        out_dir = LUT_OUTPUT_PATH
    elif os.path.isdir(raw_out):
        out_dir = raw_out
    else:
        print(f"  '{raw_out}' is not a valid directory — using default.")
        out_dir = LUT_OUTPUT_PATH

    # ── Ask for LUT size ─────────────────────────────────────────────────
    lut_size = LUT_SIZE
    print(f"\n  LUT size [{lut_size}] (press Enter to keep, or type a value): ", end="")
    size_raw = input().strip()
    if size_raw:
        try:
            candidate = int(size_raw)
            if candidate >= 2:
                lut_size = candidate
            else:
                print("  Value too small, keeping default.")
        except ValueError:
            print("  Not a number, keeping default.")

    # ── Process each file ────────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    print(f"  Processing {n} file(s) …\n")
    safe_name = emulsion_name.replace(" ", "_").replace("/", "-")
    successes: list[tuple[str, str]] = []
    failures:  list[tuple[str, str, str]] = []

    for csv_file, stop in zip(all_files, stop_assignments):
        stop_lbl = _stop_label(stop)
        csv_path = os.path.join(csv_dir, csv_file)

        print(f"  [{stop_lbl} EV]  {csv_file}")
        patches = import_csv(csv_path)
        if patches is None:
            print(f"  ✗ Skipped (CSV load error)\n")
            failures.append((csv_file, stop_lbl, "CSV load error"))
            continue

        results, x0 = run_calculation(patches, emulsion, stop)
        lut_filename = f"{safe_name}_{stop_lbl}EV_1D.cube"
        lut_path     = os.path.join(out_dir, lut_filename)

        ok = export_1d_lut(results, emulsion_name, stop, lut_path, lut_size)
        if ok:
            successes.append((lut_filename, stop_lbl))
            print(f"  ✔ Saved → {lut_path}\n")
        else:
            failures.append((csv_file, stop_lbl, "LUT write error"))
            print(f"  ✗ LUT write failed\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"  {'=' * 60}")
    print(f"  Batch complete: {len(successes)} LUT(s) saved, "
          f"{len(failures)} failed.")
    if successes:
        print(f"\n  Saved LUTs:")
        for fn, stop_lbl in successes:
            print(f"    {stop_lbl:>3} EV  →  {fn}")
    if failures:
        print(f"\n  Failures:")
        for csv_f, stop_lbl, reason in failures:
            print(f"    {stop_lbl:>3} EV  ←  {csv_f}  ({reason})")
    print(f"\n  Output directory: {out_dir}")
    print(f"  {'=' * 60}")


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  FILM DENSITY / LOG EXPOSURE BENDING TOOL")
    print("  ColorChecker 24 | Cineon 10-bit Float")
    print("=" * 60)

    lib = load_library()

    while True:
        print("\n  -- Main Menu --")
        print("  1  Run calculation")
        print("  2  View emulsion library")
        print("  3  Add emulsion")
        print("  4  Edit emulsion")
        print("  5  Delete emulsion")
        print("  6  Set LUT output path")
        print("  7  Batch bracket LUTs from CSV folder")
        print("  0  Quit")
        choice = input("\n  > ").strip()

        if choice == "1":
            print("\n-- Select Emulsion --")
            emulsion_name, emulsion = pick_emulsion(lib)

            print("\n  -- Processing Mode --")
            print("  1  Single exposure")
            print("  2  Full bracket (batch, one CSV per stop)")
            while True:
                mode = input("\n  > ").strip()
                if mode in ("1", "2"):
                    break
                print("  Please enter 1 or 2.")

            if mode == "1":
                exposure_stop = ask_exposure_stop()
                patches = get_debug_cineon_values() if DEBUG_MODE else ask_data_source()
                results, x0 = run_calculation(patches, emulsion, exposure_stop)
                print_results(results, emulsion_name, x0, exposure_stop,
                              x_unit=emulsion.get("x_unit", "log10"))
                ask_lut_export(results, emulsion_name, exposure_stop)
            else:
                batch_bracket_luts(lib, emulsion_name=emulsion_name, emulsion=emulsion)

        elif choice == "2":
            list_emulsions(lib)

        elif choice == "3":
            add_emulsion(lib)

        elif choice == "4":
            edit_emulsion(lib)

        elif choice == "5":
            delete_emulsion(lib)

        elif choice == "6":
            set_lut_output_path()

        elif choice == "7":
            batch_bracket_luts(lib)

        elif choice == "0":
            print("  Goodbye.\n")
            sys.exit(0)

        else:
            print("  Unknown option.")


if __name__ == "__main__":
    main()
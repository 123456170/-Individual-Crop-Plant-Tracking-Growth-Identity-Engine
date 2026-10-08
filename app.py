import math
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="Crop Plant Identity Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SEED = 20261008

FIELD_W = 100.0
FIELD_H = 60.0
IMG_W = 1280
IMG_H = 720

ROWS = 8
COLS = 14
N_SURVEYS = 25
START_DATE = pd.Timestamp("2026-04-15")

ROW_Y = np.linspace(4.8, FIELD_H - 4.8, ROWS)
COL_X = np.linspace(4.2, FIELD_W - 4.2, COLS)

SCALE_X = IMG_W / FIELD_W
SCALE_Y = IMG_H / FIELD_H

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    try:
        v = float(value)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _is_none_or_nan(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _mean_or_zero(values):
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _rmse_or_zero(errors):
    try:
        arr = np.asarray(errors, dtype=float).reshape(-1)
    except Exception:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _safe_max(values, default=1.0):
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.max())


def nearest_row(y):
    return int(np.argmin(np.abs(ROW_Y - float(y))))


def nearest_col(x):
    return int(np.argmin(np.abs(COL_X - float(x))))


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - np.dot(a, b) / (na * nb))))


def stage_from_height(h):
    h = _safe_float(h, 0.0)
    if h <= 0.0:
        return "Unknown"
    if h < 0.12:
        return "Emergence"
    if h < 0.35:
        return "Vegetative"
    if h < 0.70:
        return "Rapid growth"
    if h < 0.95:
        return "Reproductive"
    return "Maturity"


def make_embedding(
    height,
    area,
    leaf_count,
    ndvi,
    hue,
    saturation,
    value,
    occlusion,
    row,
    col,
    density,
    rng=None,
):
    """
    Visual + geometric embedding used for re-identification.
    Absolute x/y are intentionally excluded to tolerate camera translation.
    """
    height = _safe_float(height, 0.2)
    area = max(0.001, _safe_float(area, 0.05))
    leaf_count = max(0.0, _safe_float(leaf_count, 3.0))
    ndvi = _safe_float(ndvi, 0.5)
    hue = _safe_float(hue, 95.0)
    saturation = _safe_float(saturation, 150.0)
    value = _safe_float(value, 130.0)
    occlusion = _safe_float(occlusion, 0.0)
    row = _safe_float(row, 0.0)
    col = _safe_float(col, 0.0)
    density = _safe_float(density, 0.0)

    v = np.array(
        [
            height / 1.8,
            np.log1p(area * 12.0),
            leaf_count / 22.0,
            ndvi,
            (hue - 85.0) / 50.0,
            saturation / 255.0,
            value / 255.0,
            1.0 - occlusion,
            row / max(1.0, float(ROWS - 1)),
            col / max(1.0, float(COLS - 1)),
            min(1.0, density / 8.0),
            area / (area + 0.25),
            height / (math.sqrt(area) + 0.15),
            0.5 * (saturation / 255.0) * (1.0 - occlusion),
        ],
        dtype=float,
    )

    if rng is not None:
        v = v + rng.normal(0.0, 0.018, size=v.shape)

    v = np.nan_to_num(v, nan=0.0, posinf=3.0, neginf=-3.0)
    v = np.clip(v, -3.0, 3.0)
    return normalize_vector(v)


# ------------------------------------------------------------------------------
# Synthetic ground truth and simulated instance-segmentation detections
# ------------------------------------------------------------------------------
def generate_ground_truth(rng):
    rows = []
    idx = 1

    for r in range(ROWS):
        for c in range(COLS):
            # Some stations never emerge: missing-plant / gap simulation.
            if rng.random() < 0.07:
                continue

            x = float(COL_X[c] + rng.normal(0.0, 0.22))
            y = float(ROW_Y[r] + rng.normal(0.0, 0.18))

            emergence_date = pd.Timestamp("2026-03-20") + pd.Timedelta(
                days=float(rng.normal(0.0, 5.0))
            )

            max_height = float(np.clip(rng.normal(1.28, 0.16), 0.72, 1.78))
            canopy_potential = float(np.clip(rng.normal(0.35, 0.075), 0.10, 0.65))
            health = float(np.clip(rng.normal(0.89, 0.08), 0.45, 1.0))
            variety = str(rng.choice(["Hybrid-A", "Hybrid-B", "Landrace-C", "Test-line-D"]))

            stress_start = pd.NaT
            if rng.random() < 0.19:
                stress_start = pd.Timestamp("2026-06-05") + pd.Timedelta(
                    days=float(rng.normal(8.0, 18.0))
                )

            rows.append(
                {
                    "gt_id": f"GT-{idx:04d}",
                    "row": int(r),
                    "col": int(c),
                    "x": x,
                    "y": y,
                    "emergence_date": emergence_date,
                    "max_height": max_height,
                    "canopy_potential": canopy_potential,
                    "health": health,
                    "variety": variety,
                    "stress_start": stress_start,
                }
            )
            idx += 1

    return pd.DataFrame(rows)


def compute_traits(pl, date):
    age = max(0, int((date - pl["emergence_date"]).days))
    frac = 1.0 / (1.0 + math.exp(-0.055 * (age - 42.0)))

    stress = 0.0
    stress_start = pl.get("stress_start")
    if pd.notna(stress_start) and date >= stress_start:
        stress_days = max(0, int((date - stress_start).days))
        stress = float(min(0.42, 0.016 * stress_days))

    health_eff = float(np.clip(pl["health"] - stress, 0.22, 1.0))

    height = float(
        max(
            0.02,
            pl["max_height"]
            * frac
            * (0.74 + 0.36 * health_eff)
            * (1.0 - 0.34 * stress),
        )
    )

    canopy = float(
        max(
            0.008,
            pl["canopy_potential"]
            * (frac ** 1.25)
            * (0.64 + 0.56 * health_eff)
            * (1.0 - 0.24 * stress),
        )
    )

    leaf_count = int(max(1, round(2 + 13 * frac + 3 * health_eff - 2.2 * stress)))
    ndvi = float(np.clip(health_eff * (0.32 + 0.62 * frac) - 0.34 * stress, 0.04, 0.98))

    hue = float(np.clip(106.0 - (1.0 - health_eff) * 48.0 - stress * 28.0, 38.0, 126.0))
    saturation = float(np.clip(175.0 + 55.0 * health_eff - 45.0 * stress, 55.0, 255.0))
    value = float(np.clip(105.0 + 95.0 * frac + 35.0 * health_eff - 25.0 * stress, 50.0, 245.0))

    return {
        "age_days": age,
        "fraction": float(frac),
        "stress": float(stress),
        "health_eff": health_eff,
        "height_true": height,
        "canopy_true": canopy,
        "leaf_true": leaf_count,
        "ndvi_true": ndvi,
        "hue": hue,
        "saturation": saturation,
        "value": value,
    }


def generate_detections(gt, date, survey_id, rng, densities):
    """
    Simulates an instance-segmentation model output:
    masks are represented by centroid, area, color statistics and embeddings.
    """
    detections = []
    truth_rows = []

    shift_x = float(rng.normal(0.0, 0.18))
    shift_y = float(rng.normal(0.0, 0.18))
    if rng.random() < 0.18:
        shift_x += float(rng.normal(0.0, 0.55))
        shift_y += float(rng.normal(0.0, 0.45))

    gt_records = gt.to_dict("records")
    visible_count = 0
    plant_detection_count = 0

    for i, pl in enumerate(gt_records):
        traits = compute_traits(pl, date)
        density = _safe_float(densities[i], 0.0)
        density_norm = min(1.0, density / 8.0)

        visible_prob = 0.97
        visible_prob -= 0.18 * max(0.0, traits["canopy_true"] - 0.34)
        visible_prob -= 0.12 * density_norm
        if traits["stress"] > 0.18:
            visible_prob -= 0.04
        if rng.random() < 0.07:
            visible_prob *= 0.72
        visible_prob = float(np.clip(visible_prob, 0.22, 0.995))

        visible = bool(rng.random() < visible_prob)

        truth_rows.append(
            {
                "survey_id": survey_id,
                "gt_id": pl["gt_id"],
                "x_true": float(pl["x"]),
                "y_true": float(pl["y"]),
                "visible": visible,
                **traits,
            }
        )

        if not visible:
            continue

        visible_count += 1

        occ_prob = 0.14 + 0.42 * max(0.0, traits["canopy_true"] - 0.18) + 0.18 * density_norm
        occ = 0.0
        if rng.random() < occ_prob:
            occ = float(np.clip(rng.beta(1.5, 4.0), 0.0, 0.92))

        x = float(
            np.clip(
                pl["x"] + shift_x + rng.normal(0.0, 0.10 + 0.55 * occ),
                0.2,
                FIELD_W - 0.2,
            )
        )
        y = float(
            np.clip(
                pl["y"] + shift_y + rng.normal(0.0, 0.10 + 0.55 * occ),
                0.2,
                FIELD_H - 0.2,
            )
        )

        area = float(
            max(
                0.008,
                traits["canopy_true"] * math.exp(rng.normal(0.0, 0.06 + 0.30 * occ)),
            )
        )
        height = float(
            max(
                0.02,
                traits["height_true"] * math.exp(rng.normal(0.0, 0.05 + 0.22 * occ)),
            )
        )
        leaf_count = int(max(1, round(traits["leaf_true"] + rng.normal(0.0, 0.6 + 1.8 * occ))))
        ndvi = float(np.clip(traits["ndvi_true"] + rng.normal(0.0, 0.02 + 0.09 * occ), 0.02, 0.99))
        hue = float(np.clip(traits["hue"] + rng.normal(0.0, 2.0 + 8.0 * occ), 35.0, 130.0))
        saturation = float(np.clip(traits["saturation"] + rng.normal(0.0, 5.0 + 20.0 * occ), 40.0, 255.0))
        value = float(np.clip(traits["value"] + rng.normal(0.0, 5.0 + 25.0 * occ), 40.0, 255.0))

        conf = float(np.clip((1.0 - 0.52 * occ) * rng.beta(6.0, 1.6), 0.05, 0.99))

        emb = make_embedding(
            height=height,
            area=area,
            leaf_count=leaf_count,
            ndvi=ndvi,
            hue=hue,
            saturation=saturation,
            value=value,
            occlusion=occ,
            row=nearest_row(y),
            col=nearest_col(x),
            density=density,
            rng=rng,
        )

        detections.append(
            {
                "survey_id": survey_id,
                "timestamp": date,
                "x": x,
                "y": y,
                "area": area,
                "height": height,
                "leaf_count": leaf_count,
                "ndvi": ndvi,
                "hue": hue,
                "saturation": saturation,
                "value": value,
                "occlusion": occ,
                "conf": conf,
                "embedding": emb,
                "gt_id": pl["gt_id"],
                "row": nearest_row(y),
                "col": nearest_col(x),
                "source": "crop_sim",
            }
        )
        plant_detection_count += 1

    # Weed / residue false positives.
    fp_count = int(rng.poisson(2.1))
    for _ in range(fp_count):
        x = float(rng.uniform(1.5, FIELD_W - 1.5))
        y = float(rng.uniform(1.5, FIELD_H - 1.5))
        area = float(np.clip(rng.gamma(1.3, 0.035), 0.004, 0.18))
        height = float(max(0.03, math.sqrt(area) * rng.uniform(0.35, 0.95)))
        leaf_count = int(rng.integers(1, 7))
        ndvi = float(np.clip(rng.normal(0.38, 0.12), 0.04, 0.82))
        hue = float(np.clip(rng.normal(88.0, 18.0), 40.0, 125.0))
        saturation = float(np.clip(rng.normal(150.0, 45.0), 40.0, 255.0))
        value = float(np.clip(rng.normal(130.0, 35.0), 40.0, 245.0))
        occ = float(np.clip(rng.beta(2.0, 5.0), 0.0, 0.72))
        conf = float(np.clip(rng.beta(2.0, 4.0), 0.04, 0.76))

        emb = make_embedding(
            height=height,
            area=area,
            leaf_count=leaf_count,
            ndvi=ndvi,
            hue=hue,
            saturation=saturation,
            value=value,
            occlusion=occ,
            row=nearest_row(y),
            col=nearest_col(x),
            density=float(rng.uniform(0.0, 0.22)),
            rng=rng,
        )

        detections.append(
            {
                "survey_id": survey_id,
                "timestamp": date,
                "x": x,
                "y": y,
                "area": area,
                "height": height,
                "leaf_count": leaf_count,
                "ndvi": ndvi,
                "hue": hue,
                "saturation": saturation,
                "value": value,
                "occlusion": occ,
                "conf": conf,
                "embedding": emb,
                "gt_id": None,
                "row": nearest_row(y),
                "col": nearest_col(x),
                "source": "weed_sim",
            }
        )

    meta = {
        "survey_id": survey_id,
        "timestamp": date,
        "visible_gt": visible_count,
        "plant_detections": plant_detection_count,
        "false_positives": fp_count,
        "generated_shift_x": shift_x,
        "generated_shift_y": shift_y,
    }

    return detections, truth_rows, meta


# ------------------------------------------------------------------------------
# Persistent plant identity tracker / re-identification engine
# ------------------------------------------------------------------------------
class PlantTracker:
    def __init__(self, params=None):
        self.params = params or {
            "gate": 3.5,
            "match_threshold": 0.62,
            "max_missed": 4,
            "miss_lost": 2,
            "new_min_conf": 0.26,
        }
        self.tracks = {}
        self.next_id = 1
        self.events = []

    def _event(self, etype, track_id, gt_id, timestamp, survey_id, message, severity=2):
        self.events.append(
            {
                "timestamp": timestamp,
                "survey_id": survey_id,
                "type": etype,
                "track_id": track_id,
                "gt_id": gt_id,
                "message": message,
                "severity": severity,
            }
        )

    def _align_detections(self, dets, active_tracks):
        """
        Robust camera-shift compensation using median nearest-track offset.
        """
        if not dets or not active_tracks:
            return (0.0, 0.0)

        track_pts = np.array([[float(t["x"]), float(t["y"])] for t in active_tracks], dtype=float)
        det_pts = np.array([[float(d["x"]), float(d["y"])] for d in dets], dtype=float)

        tree = cKDTree(track_pts)
        dist, idx = tree.query(det_pts, k=1)

        offsets = []
        for d, i, dd in zip(dets, idx, dist):
            if dd < self.params["gate"] * 1.8:
                offsets.append(
                    [
                        float(d["x"]) - float(active_tracks[i]["x"]),
                        float(d["y"]) - float(active_tracks[i]["y"]),
                    ]
                )

        if len(offsets) < 3:
            return (0.0, 0.0)

        off = np.median(np.array(offsets, dtype=float), axis=0)
        off = np.clip(off, -1.2, 1.2)

        for d in dets:
            d["x"] = float(np.clip(d["x"] - off[0], 0.0, FIELD_W))
            d["y"] = float(np.clip(d["y"] - off[1], 0.0, FIELD_H))
            d["row"] = nearest_row(d["y"])
            d["col"] = nearest_col(d["x"])

        return (float(off[0]), float(off[1]))

    def _create_track(self, det, survey_id, timestamp, source):
        tid = f"P-{self.next_id:04d}"
        self.next_id += 1

        det_conf = _safe_float(det.get("conf", det.get("confidence", 0.5)), 0.5)
        gt_id = det.get("gt_id")

        track = {
            "track_id": tid,
            "status": "active",
            "x": float(det["x"]),
            "y": float(det["y"]),
            "row": int(det.get("row", nearest_row(det["y"]))),
            "col": int(det.get("col", nearest_col(det["x"]))),
            "area": float(det["area"]),
            "height": float(det["height"]),
            "embedding": normalize_vector(det["embedding"]),
            "confidence": float(np.clip(0.22 + 0.62 * det_conf, 0.05, 0.92)),
            "hits": 1,
            "misses": 0,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "last_survey": survey_id,
            "last_gt_id": gt_id,
            "gt_counts": {},
            "id_switches": 0,
            "source": det.get("source", source),
        }

        if gt_id is not None and not _is_none_or_nan(gt_id):
            track["gt_counts"][gt_id] = 1

        self.tracks[tid] = track
        self._event(
            "NEW_TRACK",
            tid,
            gt_id,
            timestamp,
            survey_id,
            "New persistent plant identity created",
            severity=1,
        )

        return self._observation_row(track, det, survey_id, timestamp, match_cost=None)

    def _match_track(self, track, det, cost, survey_id, timestamp, source):
        old_status = track["status"]
        old_gt = track.get("last_gt_id")
        new_gt = det.get("gt_id")

        if new_gt is not None and not _is_none_or_nan(new_gt):
            if old_gt is not None and not _is_none_or_nan(old_gt) and new_gt != old_gt:
                track["id_switches"] += 1
                self._event(
                    "ID_SWITCH",
                    track["track_id"],
                    new_gt,
                    timestamp,
                    survey_id,
                    f"Identity changed from {old_gt} to {new_gt}",
                    severity=4,
                )
            track["gt_counts"][new_gt] = track["gt_counts"].get(new_gt, 0) + 1
            track["last_gt_id"] = new_gt

        if old_status == "lost":
            self._event(
                "TRACK_RECOVERED",
                track["track_id"],
                new_gt,
                timestamp,
                survey_id,
                "Lost track re-associated after occlusion / missed survey",
                severity=2,
            )

        det_conf = _safe_float(det.get("conf", det.get("confidence", 0.5)), 0.5)
        match_quality = 1.0 - min(float(cost), 1.0)

        track["confidence"] = float(
            np.clip(
                0.56 * track["confidence"] + 0.26 * match_quality + 0.18 * det_conf,
                0.03,
                0.99,
            )
        )

        track["x"] = float(0.50 * track["x"] + 0.50 * det["x"])
        track["y"] = float(0.50 * track["y"] + 0.50 * det["y"])
        track["area"] = float(0.55 * track["area"] + 0.45 * det["area"])
        track["height"] = float(0.55 * track["height"] + 0.45 * det["height"])
        track["embedding"] = normalize_vector(0.60 * track["embedding"] + 0.40 * det["embedding"])
        track["row"] = int(det.get("row", track["row"]))
        track["col"] = int(det.get("col", track["col"]))
        track["hits"] += 1
        track["misses"] = 0
        track["status"] = "active"
        track["last_seen"] = timestamp
        track["last_survey"] = survey_id
        track["source"] = det.get("source", source)

        return self._observation_row(track, det, survey_id, timestamp, match_cost=float(cost))

    def _miss_track(self, track, survey_id, timestamp):
        track["misses"] += 1
        track["confidence"] = float(max(0.02, track["confidence"] * 0.72))

        if track["misses"] == self.params["miss_lost"] and track["status"] == "active":
            track["status"] = "lost"
            self._event(
                "TRACK_LOST",
                track["track_id"],
                track.get("last_gt_id"),
                timestamp,
                survey_id,
                "Track temporarily lost due to occlusion / missed detection",
                severity=3,
            )
        elif track["misses"] > self.params["max_missed"] and track["status"] != "closed":
            track["status"] = "closed"
            self._event(
                "TRACK_CLOSED",
                track["track_id"],
                track.get("last_gt_id"),
                timestamp,
                survey_id,
                "Track closed after excessive missed surveys",
                severity=4,
            )

    def _observation_row(self, track, det, survey_id, timestamp, match_cost):
        return {
            "survey_id": survey_id,
            "timestamp": timestamp,
            "track_id": track["track_id"],
            "gt_id": det.get("gt_id"),
            "x": float(det["x"]),
            "y": float(det["y"]),
            "canopy_area": float(det["area"]),
            "height_est": float(det["height"]),
            "leaf_count": int(det["leaf_count"]),
            "ndvi": float(det["ndvi"]),
            "hue": float(det["hue"]),
            "saturation": float(det["saturation"]),
            "value": float(det["value"]),
            "confidence": float(track["confidence"]),
            "detection_conf": _safe_float(det.get("conf", det.get("confidence", 0.5)), 0.5),
            "match_cost": match_cost,
            "occlusion": _safe_float(det.get("occlusion", 0.0), 0.0),
            "row": int(det.get("row", track["row"])),
            "col": int(det.get("col", track["col"])),
            "source": det.get("source", "unknown"),
        }

    def update(self, detections, survey_id, timestamp, source="demo"):
        dets = [dict(d) for d in detections]
        active_tracks = [
            t
            for t in self.tracks.values()
            if t["status"] != "closed" and t["misses"] <= self.params["max_missed"]
        ]

        shift = self._align_detections(dets, active_tracks)

        obs_rows = []
        matched_track_idx = set()
        matched_det_idx = set()

        if active_tracks and dets:
            cost = np.full((len(active_tracks), len(dets)), 1e6, dtype=float)

            for i, tr in enumerate(active_tracks):
                for j, d in enumerate(dets):
                    dist = math.hypot(float(tr["x"]) - float(d["x"]), float(tr["y"]) - float(d["y"]))
                    if dist > self.params["gate"]:
                        continue

                    spatial = min(dist / self.params["gate"], 1.0)
                    appearance = cosine_distance(tr["embedding"], d["embedding"])
                    row_pen = min(abs(float(tr["row"]) - float(d.get("row", tr["row"]))) / 2.0, 1.0)
                    col_pen = min(abs(float(tr["col"]) - float(d.get("col", tr["col"]))) / 2.0, 1.0)
                    size_pen = min(
                        abs(math.log((float(tr["area"]) + 0.03) / (float(d["area"]) + 0.03))) / 0.9,
                        1.0,
                    )
                    det_pen = 1.0 - _safe_float(d.get("conf", d.get("confidence", 0.5)), 0.5)
                    miss_pen = min(float(tr["misses"]) / 4.0, 0.35)

                    c = (
                        0.31 * spatial
                        + 0.33 * appearance
                        + 0.05 * row_pen
                        + 0.04 * col_pen
                        + 0.13 * size_pen
                        + 0.09 * det_pen
                        + 0.05 * miss_pen
                    )
                    cost[i, j] = float(c)

            row_ind, col_ind = linear_sum_assignment(cost)

            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < self.params["match_threshold"]:
                    tr = active_tracks[i]
                    d = dets[j]
                    row = self._match_track(tr, d, cost[i, j], survey_id, timestamp, source)
                    obs_rows.append(row)
                    matched_track_idx.add(i)
                    matched_det_idx.add(j)

        for i, tr in enumerate(active_tracks):
            if i not in matched_track_idx:
                self._miss_track(tr, survey_id, timestamp)

        for j, d in enumerate(dets):
            if j not in matched_det_idx:
                det_conf = _safe_float(d.get("conf", d.get("confidence", 0.5)), 0.5)
                if det_conf >= self.params["new_min_conf"]:
                    row = self._create_track(d, survey_id, timestamp, source)
                    obs_rows.append(row)

        return obs_rows, shift


# ------------------------------------------------------------------------------
# Derived tables, analytics, evaluation
# ------------------------------------------------------------------------------
def events_to_df(events):
    cols = ["timestamp", "survey_id", "type", "track_id", "gt_id", "message", "severity"]
    if not events:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(events)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_tracks_df(tracker, obs_df):
    cols = [
        "track_id",
        "primary_gt_id",
        "status",
        "first_seen",
        "last_seen",
        "hits",
        "misses",
        "confidence",
        "x",
        "y",
        "row",
        "col",
        "canopy_latest",
        "height_latest",
        "id_switches",
        "source",
    ]

    rows = []
    for tid, t in tracker.tracks.items():
        primary_gt = None
        if t.get("gt_counts"):
            primary_gt = max(t["gt_counts"].items(), key=lambda kv: kv[1])[0]

        rows.append(
            {
                "track_id": tid,
                "primary_gt_id": primary_gt,
                "status": t["status"],
                "first_seen": t["first_seen"],
                "last_seen": t["last_seen"],
                "hits": t["hits"],
                "misses": t["misses"],
                "confidence": t["confidence"],
                "x": t["x"],
                "y": t["y"],
                "row": t["row"],
                "col": t["col"],
                "canopy_latest": t["area"],
                "height_latest": t["height"],
                "id_switches": t["id_switches"],
                "source": t.get("source", "unknown"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["last_seen"] = pd.to_datetime(df["last_seen"])
    return df


def add_growth_and_anomalies(obs_df, tracks_df, truth_df, surveys_df):
    obs = obs_df.copy()
    tracks = tracks_df.copy()

    base_track_cols = [
        "track_id",
        "primary_gt_id",
        "status",
        "first_seen",
        "last_seen",
        "hits",
        "misses",
        "confidence",
        "x",
        "y",
        "row",
        "col",
        "canopy_latest",
        "height_latest",
        "id_switches",
        "source",
        "obs_count",
        "mean_confidence",
        "latest_anomaly_score",
        "max_anomaly_score",
        "latest_growth_rate",
        "latest_ndvi",
        "latest_leaf",
        "canopy_slope",
        "abnormal",
        "suspected_missing",
        "cluster_id",
        "growth_stage",
    ]

    if obs.empty:
        if tracks.empty:
            tracks = pd.DataFrame(columns=base_track_cols)
        for c in base_track_cols:
            if c not in tracks.columns:
                tracks[c] = np.nan
        if not tracks.empty:
            tracks["abnormal"] = False
            tracks["suspected_missing"] = False
            tracks["cluster_id"] = -1
            tracks["growth_stage"] = tracks["height_latest"].apply(stage_from_height)
        return obs, tracks

    obs["timestamp"] = pd.to_datetime(obs["timestamp"])
    obs = obs.sort_values(["track_id", "timestamp"]).reset_index(drop=True)

    obs["track_first_time"] = obs.groupby("track_id")["timestamp"].transform("min")
    obs["days_since_first"] = (obs["timestamp"] - obs["track_first_time"]).dt.days

    obs["prev_canopy"] = obs.groupby("track_id")["canopy_area"].shift(1)
    obs["prev_days"] = obs.groupby("track_id")["days_since_first"].shift(1)
    dt = obs["days_since_first"] - obs["prev_days"]
    obs["instant_growth"] = np.where(dt > 0, (obs["canopy_area"] - obs["prev_canopy"]) / dt, np.nan)
    obs["instant_growth"] = (
        obs.groupby("track_id", sort=False)["instant_growth"]
        .transform(lambda s: s.bfill().ffill().fillna(0.0))
    )

    obs["local_canopy_z"] = 0.0
    obs["local_growth_z"] = 0.0
    obs["anomaly_score"] = 0.0
    obs["abnormal"] = False
    obs["cluster_id"] = -1

    for _, grp in obs.groupby("survey_id", sort=False):
        if len(grp) < 4:
            continue

        pts = grp[["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(pts)
        canopy = grp["canopy_area"].to_numpy(dtype=float)
        growth = grp["instant_growth"].to_numpy(dtype=float)
        neigh_lists = tree.query_ball_point(pts, r=9.0)

        for pos, idx_label in enumerate(grp.index):
            neigh = [n for n in neigh_lists[pos] if n != pos]
            z_c = 0.0
            z_g = 0.0

            if len(neigh) >= 3:
                c = canopy[neigh]
                med_c = float(np.median(c))
                mad_c = float(np.median(np.abs(c - med_c))) + 1e-3
                z_c = float((canopy[pos] - med_c) / (1.4826 * mad_c))

                g = growth[neigh]
                g = g[~np.isnan(g)]
                if len(g) >= 3:
                    med_g = float(np.median(g))
                    mad_g = float(np.median(np.abs(g - med_g))) + 1e-4
                    z_g = float((growth[pos] - med_g) / (1.4826 * mad_g))

            score = 0.0
            if z_c < 0:
                score += (-z_c) * 0.65
            if z_g < 0:
                score += (-z_g) * 0.45
            if growth[pos] < -0.0012:
                score += 0.75

            obs.at[idx_label, "local_canopy_z"] = z_c
            obs.at[idx_label, "local_growth_z"] = z_g
            obs.at[idx_label, "anomaly_score"] = float(np.clip(score, 0.0, 10.0))
            obs.at[idx_label, "abnormal"] = bool(score >= 2.0 or z_c <= -2.2 or z_g <= -2.0)

    agg = (
        obs.groupby("track_id")
        .agg(
            obs_count=("track_id", "size"),
            mean_confidence=("confidence", "mean"),
            latest_anomaly_score=("anomaly_score", "last"),
            max_anomaly_score=("anomaly_score", "max"),
            latest_growth_rate=("instant_growth", "last"),
            latest_ndvi=("ndvi", "last"),
            latest_leaf=("leaf_count", "last"),
        )
        .reset_index()
    )

    if tracks.empty:
        tracks = pd.DataFrame({"track_id": agg["track_id"].unique()})

    drop_cols = [c for c in agg.columns if c != "track_id" and c in tracks.columns]
    if drop_cols:
        tracks = tracks.drop(columns=drop_cols)

    tracks = tracks.merge(agg, on="track_id", how="left")

    slopes = []
    for tid, grp in obs.groupby("track_id"):
        if len(grp) >= 3 and float(np.ptp(grp["days_since_first"].to_numpy(dtype=float))) > 7:
            x = grp["days_since_first"].to_numpy(dtype=float)
            y = grp["canopy_area"].to_numpy(dtype=float)
            xm = float(np.mean(x))
            ym = float(np.mean(y))
            denom = float(np.sum((x - xm) ** 2))
            if denom > 1e-9:
                slope = float(np.sum((x - xm) * (y - ym)) / denom)
            else:
                slope = np.nan
        else:
            slope = np.nan
        slopes.append((tid, slope))

    slope_df = pd.DataFrame(slopes, columns=["track_id", "canopy_slope"])
    tracks = tracks.drop(columns=["canopy_slope"], errors="ignore").merge(slope_df, on="track_id", how="left")

    for c in [
        "obs_count",
        "mean_confidence",
        "latest_anomaly_score",
        "max_anomaly_score",
        "latest_growth_rate",
        "latest_ndvi",
        "latest_leaf",
        "canopy_slope",
    ]:
        if c not in tracks.columns:
            tracks[c] = np.nan
        tracks[c] = pd.to_numeric(tracks[c], errors="coerce")

    tracks["max_anomaly_score"] = tracks["max_anomaly_score"].fillna(0.0)
    tracks["abnormal"] = tracks["max_anomaly_score"] >= 2.0

    final_time = pd.Timestamp.now()
    if surveys_df is not None and not surveys_df.empty and "timestamp" in surveys_df.columns:
        final_time = pd.to_datetime(surveys_df["timestamp"]).max()

    tracks["last_seen"] = pd.to_datetime(tracks["last_seen"], errors="coerce")
    tracks["suspected_missing"] = (
        tracks["status"].isin(["lost", "closed"])
        | (
            (tracks["last_seen"] < final_time - pd.Timedelta(days=21))
            & (tracks["obs_count"].fillna(0) > 0)
        )
    )

    abnormal_obs = obs[obs["abnormal"].fillna(False).astype(bool)]
    if len(abnormal_obs) >= 3:
        try:
            db = DBSCAN(eps=8.0, min_samples=2).fit(abnormal_obs[["x", "y"]].to_numpy(dtype=float))
            obs.loc[abnormal_obs.index, "cluster_id"] = db.labels_

            cluster_map = (
                obs[obs["cluster_id"] >= 0]
                .groupby("track_id")["cluster_id"]
                .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) > 0 else -1)
            )
            tracks["cluster_id"] = tracks["track_id"].map(cluster_map).fillna(-1).astype(int)
        except Exception:
            tracks["cluster_id"] = -1
    else:
        tracks["cluster_id"] = -1

    tracks["growth_stage"] = tracks["height_latest"].apply(stage_from_height)

    for c in base_track_cols:
        if c not in tracks.columns:
            tracks[c] = np.nan

    return obs, tracks


def evaluate_engine(tracker, obs_df, truth_df, det_eval_df, tracks_df):
    metrics = {
        "detection_precision": 0.0,
        "detection_recall": 0.0,
        "detection_f1": 0.0,
        "tracked_recall": 0.0,
        "id_switch_count": 0,
        "id_switch_rate": 0.0,
        "track_persistence": 0.0,
        "mean_track_coverage": 0.0,
        "track_loss_events": 0,
        "height_mae": 0.0,
        "height_rmse": 0.0,
        "canopy_mae": 0.0,
        "canopy_rmse": 0.0,
        "ndvi_mae": 0.0,
        "active_tracks": 0,
        "total_tracks": 0,
        "avg_confidence": 0.0,
    }

    if det_eval_df is not None and not det_eval_df.empty:
        pairs = [
            ("precision", "detection_precision"),
            ("recall", "detection_recall"),
            ("f1", "detection_f1"),
            ("tracked_recall", "tracked_recall"),
        ]
        for col, key in pairs:
            if col in det_eval_df.columns:
                metrics[key] = _mean_or_zero(det_eval_df[col])

    matched = obs_df[obs_df["gt_id"].notna()].copy() if not obs_df.empty else pd.DataFrame()
    total_gt_obs = len(matched)

    id_switches = int(sum(t.get("id_switches", 0) for t in tracker.tracks.values()))
    metrics["id_switch_count"] = id_switches
    metrics["id_switch_rate"] = float(id_switches / max(1, total_gt_obs))

    if not matched.empty and truth_df is not None and not truth_df.empty:
        visible_counts = truth_df[truth_df["visible"].astype(bool)].groupby("gt_id").size()
        gt_tracks = matched.groupby("gt_id")["track_id"].nunique()
        obs_counts = matched.groupby("gt_id").size()

        persistence_frame = pd.concat(
            [visible_counts.rename("visible"), gt_tracks.rename("tracks")],
            axis=1,
            join="inner",
        )
        if not persistence_frame.empty:
            metrics["track_persistence"] = float((persistence_frame["tracks"] == 1).mean())

        coverage_frame = pd.concat(
            [visible_counts.rename("visible"), obs_counts.rename("obs")],
            axis=1,
            join="inner",
        )
        if not coverage_frame.empty:
            cov = coverage_frame["obs"] / coverage_frame["visible"].replace(0, np.nan)
            metrics["mean_track_coverage"] = _mean_or_zero(cov)

        merged = matched.merge(truth_df, on=["survey_id", "gt_id"], how="inner")
        if not merged.empty:
            dh = merged["height_est"] - merged["height_true"]
            dc = merged["canopy_area"] - merged["canopy_true"]
            dn = merged["ndvi"] - merged["ndvi_true"]

            metrics["height_mae"] = _mean_or_zero(np.abs(dh))
            metrics["height_rmse"] = _rmse_or_zero(dh)
            metrics["canopy_mae"] = _mean_or_zero(np.abs(dc))
            metrics["canopy_rmse"] = _rmse_or_zero(dc)
            metrics["ndvi_mae"] = _mean_or_zero(np.abs(dn))

    loss_events = [e for e in tracker.events if e.get("type") in ("TRACK_LOST", "TRACK_CLOSED")]
    metrics["track_loss_events"] = len(loss_events)

    metrics["active_tracks"] = int(sum(1 for t in tracker.tracks.values() if t["status"] == "active"))
    metrics["total_tracks"] = int(len(tracker.tracks))
    if tracker.tracks:
        metrics["avg_confidence"] = _mean_or_zero([t["confidence"] for t in tracker.tracks.values()])

    return metrics


def build_demo_data():
    rng = np.random.default_rng(SEED)

    gt = generate_ground_truth(rng)

    pts = gt[["x", "y"]].to_numpy(dtype=float)
    gt_tree = cKDTree(pts)
    densities = []
    for i in range(len(pts)):
        neigh = gt_tree.query_ball_point(pts[i], r=6.0)
        densities.append(len(neigh) - 1)
    gt["density"] = densities

    dates = [START_DATE + pd.Timedelta(days=7 * i) for i in range(N_SURVEYS)]

    tracker = PlantTracker()
    obs_rows = []
    truth_rows = []
    survey_rows = []
    det_eval_rows = []

    for i, date in enumerate(dates):
        sid = f"S-{i + 1:03d}"
        dets, truth, meta = generate_detections(gt, date, sid, rng, gt["density"].to_numpy())
        rows, shift = tracker.update(dets, sid, date, source="demo")

        obs_rows.extend(rows)
        truth_rows.extend(truth)

        survey_rows.append(
            {
                "survey_id": sid,
                "timestamp": date,
                "source": "demo_uav",
                "shift_x": shift[0],
                "shift_y": shift[1],
                "detection_count": len(dets),
                "assigned_observations": len(rows),
            }
        )

        tp = int(meta["plant_detections"])
        fp = int(meta["false_positives"])
        vis = int(meta["visible_gt"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / vis if vis > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        tracked_gt = len(
            {
                r["gt_id"]
                for r in rows
                if r.get("gt_id") is not None and not _is_none_or_nan(r.get("gt_id"))
            }
        )
        tracked_recall = tracked_gt / vis if vis > 0 else 1.0

        det_eval_rows.append(
            {
                "survey_id": sid,
                "timestamp": date,
                "visible_gt": vis,
                "plant_detections": tp,
                "false_positives": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tracked_recall": tracked_recall,
            }
        )

    obs = pd.DataFrame(obs_rows)
    truth = pd.DataFrame(truth_rows)
    surveys = pd.DataFrame(survey_rows)
    det_eval = pd.DataFrame(det_eval_rows)

    tracks = build_tracks_df(tracker, obs)
    obs, tracks = add_growth_and_anomalies(obs, tracks, truth, surveys)
    events = events_to_df(tracker.events)
    metrics = evaluate_engine(tracker, obs, truth, det_eval, tracks)

    return {
        "gt": gt,
        "truth": truth,
        "obs": obs,
        "tracks": tracks,
        "surveys": surveys,
        "det_eval": det_eval,
        "events": events,
        "metrics": metrics,
        "tracker": tracker,
        "frame_cache": {},
    }


def refresh_derived_tables(data):
    tracker = data["tracker"]
    obs = data["obs"]
    tracks = build_tracks_df(tracker, obs)
    obs, tracks = add_growth_and_anomalies(obs, tracks, data["truth"], data["surveys"])

    data["obs"] = obs
    data["tracks"] = tracks
    data["events"] = events_to_df(tracker.events)
    data["metrics"] = evaluate_engine(tracker, obs, data["truth"], data["det_eval"], tracks)

    st.session_state.image_cache = {}


# ------------------------------------------------------------------------------
# Visualization / synthetic field imagery
# ------------------------------------------------------------------------------
def synthesize_field_image(data, idx, sid, show_ids, show_masks, show_truth):
    rng = np.random.default_rng(SEED + idx * 97)

    img = np.full((IMG_H, IMG_W, 3), (54, 68, 92), dtype=np.uint8)

    for ry in ROW_Y:
        py = int(ry * SCALE_Y)
        y0 = max(0, py - 9)
        y1 = min(IMG_H, py + 9)
        cv2.rectangle(img, (0, y0), (IMG_W, y1), (64, 82, 108), -1)

    noise = rng.integers(0, 16, size=(IMG_H, IMG_W, 1), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random shadows / camera artifacts.
    for _ in range(5):
        overlay = img.copy()
        cx = int(rng.uniform(0, IMG_W))
        cy = int(rng.uniform(0, IMG_H))
        ax = int(rng.uniform(80, 260))
        ay = int(rng.uniform(40, 130))
        angle = int(rng.uniform(0, 180))
        cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, (28, 34, 42), -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0.0, img)

    obs = data["obs"]
    cur = obs[obs["survey_id"] == sid].copy()

    for _, o in cur.iterrows():
        px = int(np.clip(_safe_float(o["x"]) * SCALE_X, 0, IMG_W - 1))
        py = int(np.clip(_safe_float(o["y"]) * SCALE_Y, 0, IMG_H - 1))

        canopy = max(0.01, _safe_float(o.get("canopy_area", 0.05), 0.05))
        radius = max(5.0, math.sqrt(canopy) * SCALE_X * 0.62)
        axes = (int(radius), int(radius * 0.78))
        angle = int((hash(str(o["track_id"])) % 180))

        hue = _safe_float(o.get("hue", 95.0), 95.0)
        sat = _safe_float(o.get("saturation", 160.0), 160.0)
        val = _safe_float(o.get("value", 140.0), 140.0)

        hcv = int(np.clip(hue / 2.0, 0, 179))
        scv = int(np.clip(sat, 0, 255))
        vcv = int(np.clip(val, 0, 255))
        plant_color = cv2.cvtColor(np.uint8([[[hcv, scv, vcv]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

        is_abnormal = bool(o.get("abnormal", False)) or _safe_float(o.get("anomaly_score", 0.0), 0.0) >= 2.0
        outline_color = (0, 0, 220) if is_abnormal else tuple(int(c) for c in plant_color)

        if show_masks:
            overlay = img.copy()
            cv2.ellipse(overlay, (px, py), axes, angle, 0, 360, tuple(int(c) for c in plant_color), -1)
            cv2.addWeighted(overlay, 0.42, img, 0.58, 0.0, img)

        cv2.ellipse(img, (px, py), axes, angle, 0, 360, outline_color, 2)

        if show_ids:
            tx = int(np.clip(px - 18, 2, IMG_W - 75))
            ty = int(np.clip(py - radius - 6, 12, IMG_H - 4))
            cv2.putText(
                img,
                str(o["track_id"]),
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if show_truth:
        truth = data["truth"]
        tr = truth[truth["survey_id"] == sid]
        detected_gt = set(cur[cur["gt_id"].notna()]["gt_id"].astype(str))

        for _, t in tr.iterrows():
            px = int(np.clip(_safe_float(t["x_true"]) * SCALE_X, 0, IMG_W - 1))
            py = int(np.clip(_safe_float(t["y_true"]) * SCALE_Y, 0, IMG_H - 1))
            gt = str(t["gt_id"])

            if bool(t["visible"]):
                if gt not in detected_gt:
                    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                else:
                    cv2.circle(img, (px, py), 2, (255, 255, 0), -1)

    return img


def overlay_detections_on_frame(frame, obs_df, sid, show_ids, show_masks):
    img = cv2.resize(frame, (IMG_W, IMG_H))
    cur = obs_df[obs_df["survey_id"] == sid].copy()

    for _, o in cur.iterrows():
        px = int(np.clip(_safe_float(o["x"]) / FIELD_W * IMG_W, 0, IMG_W - 1))
        py = int(np.clip(_safe_float(o["y"]) / FIELD_H * IMG_H, 0, IMG_H - 1))

        canopy = max(0.01, _safe_float(o.get("canopy_area", 0.05), 0.05))
        radius = max(5.0, math.sqrt(canopy) * (IMG_W / FIELD_W) * 0.62)
        axes = (int(radius), int(radius * 0.78))
        angle = int((hash(str(o.get("track_id", ""))) % 180))

        hue = _safe_float(o.get("hue", 95.0), 95.0)
        sat = _safe_float(o.get("saturation", 160.0), 160.0)
        val = _safe_float(o.get("value", 140.0), 140.0)

        hcv = int(np.clip(hue / 2.0, 0, 179))
        scv = int(np.clip(sat, 0, 255))
        vcv = int(np.clip(val, 0, 255))
        color = cv2.cvtColor(np.uint8([[[hcv, scv, vcv]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

        if show_masks:
            overlay = img.copy()
            cv2.ellipse(overlay, (px, py), axes, angle, 0, 360, tuple(int(c) for c in color), -1)
            cv2.addWeighted(overlay, 0.40, img, 0.60, 0.0, img)

        cv2.ellipse(img, (px, py), axes, angle, 0, 360, tuple(int(c) for c in color), 2)

        if show_ids:
            tx = int(np.clip(px - 18, 2, IMG_W - 75))
            ty = int(np.clip(py - radius - 6, 12, IMG_H - 4))
            cv2.putText(
                img,
                str(o.get("track_id", "")),
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


def get_survey_image(data, idx, show_ids, show_masks, show_truth):
    surveys = data["surveys"]
    if surveys.empty:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    idx = int(np.clip(idx, 0, len(surveys) - 1))
    sid = surveys.iloc[idx]["survey_id"]

    if "image_cache" not in st.session_state:
        st.session_state.image_cache = {}

    key = (sid, bool(show_ids), bool(show_masks), bool(show_truth))
    cache = st.session_state.image_cache
    if key in cache:
        return cache[key]

    if sid in data.get("frame_cache", {}):
        bgr = overlay_detections_on_frame(data["frame_cache"][sid], data["obs"], sid, show_ids, show_masks)
    else:
        bgr = synthesize_field_image(data, idx, sid, show_ids, show_masks, show_truth)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cache[key] = rgb
    return rgb


def build_field_map(data, idx, show_ids, min_conf, anomaly_threshold):
    surveys = data["surveys"]
    idx = int(np.clip(idx, 0, max(0, len(surveys) - 1)))
    sid = surveys.iloc[idx]["survey_id"] if not surveys.empty else ""

    obs = data["obs"]
    tracks = data["tracks"]

    cur = obs[(obs["survey_id"] == sid) & (obs["confidence"] >= min_conf)].copy()
    if not cur.empty and not tracks.empty:
        cur = cur.merge(
            tracks[["track_id", "status", "suspected_missing"]],
            on="track_id",
            how="left",
        )

    fig = go.Figure()

    # Field boundary.
    fig.add_trace(
        go.Scatter(
            x=[0, FIELD_W, FIELD_W, 0, 0],
            y=[0, 0, FIELD_H, FIELD_H, 0],
            mode="lines",
            line=dict(color="#8a6d3b", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Row geometry.
    for y in ROW_Y:
        fig.add_trace(
            go.Scatter(
                x=[0, FIELD_W],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(130,100,70,0.22)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if not cur.empty:
        colors = []
        sizes = []
        for _, r in cur.iterrows():
            score = _safe_float(r.get("anomaly_score", 0.0), 0.0)
            status = str(r.get("status", "active"))
            missing = bool(r.get("suspected_missing", False))
            gt = r.get("gt_id")

            if score >= anomaly_threshold:
                colors.append("#d62728")
            elif missing or status in ("lost", "closed"):
                colors.append("#ff9900")
            elif gt is None or _is_none_or_nan(gt):
                colors.append("#7f7f7f")
            else:
                colors.append("#2ca02c")

            canopy = max(0.01, _safe_float(r.get("canopy_area", 0.05), 0.05))
            sizes.append(max(6.0, min(34.0, 8.0 + math.sqrt(canopy) * 24.0)))

        mode = "markers+text" if show_ids else "markers"
        text = cur["track_id"].astype(str).tolist() if show_ids else None

        customdata = list(
            zip(
                cur["track_id"].astype(str),
                cur["canopy_area"].astype(float),
                cur["height_est"].astype(float),
                cur["confidence"].astype(float),
                cur["gt_id"].fillna("none").astype(str),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=cur["x"],
                y=cur["y"],
                mode=mode,
                text=text,
                textposition="top center",
                marker=dict(size=sizes, color=colors, opacity=0.78, line=dict(width=1, color="white")),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "GT: %{customdata[4]}<br>"
                    "X: %{x:.2f} m<br>"
                    "Y: %{y:.2f} m<br>"
                    "Canopy: %{customdata[1]:.3f} m²<br>"
                    "Height: %{customdata[2]:.2f} m<br>"
                    "Confidence: %{customdata[3]:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    selected = st.session_state.get("selected_track", "All")
    if selected != "All" and not obs.empty:
        hist = obs[obs["track_id"] == selected].sort_values("timestamp")
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["x"],
                    y=hist["y"],
                    mode="lines+markers",
                    name=selected,
                    line=dict(dash="dot", width=2, color="#1f77b4"),
                    marker=dict(size=6),
                    customdata=list(zip(hist["track_id"].astype(str))),
                    hovertemplate="<b>%{customdata[0]}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        height=540,
        margin=dict(l=35, r=15, t=35, b=35),
        xaxis=dict(title="Field X (m)", range=[-2, FIELD_W + 2], zeroline=False),
        yaxis=dict(title="Field Y (m)", range=[FIELD_H + 2, -2], zeroline=False),
        plot_bgcolor="rgba(245,240,230,0.55)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def draw_preview(frame, dets):
    img = cv2.resize(frame, (IMG_W, IMG_H))
    for d in dets:
        px = int(np.clip(_safe_float(d["x"]) / FIELD_W * IMG_W, 0, IMG_W - 1))
        py = int(np.clip(_safe_float(d["y"]) / FIELD_H * IMG_H, 0, IMG_H - 1))
        area = max(0.01, _safe_float(d.get("area", 0.05), 0.05))
        radius = max(5.0, math.sqrt(area) * (IMG_W / FIELD_W) * 0.62)
        axes = (int(radius), int(radius * 0.78))
        cv2.ellipse(img, (px, py), axes, 0, 0, 360, (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{_safe_float(d.get('conf', 0.0), 0.0):.2f}",
            (px - 16, py - int(radius) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------------
# Upload / real media processing
# ------------------------------------------------------------------------------
def detect_from_frame(frame, survey_id, timestamp):
    if frame is None:
        return []

    h, w = frame.shape[:2]
    scale_x = FIELD_W / float(w)
    scale_y = FIELD_H / float(h)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    min_area = 35.0
    max_area = 0.18 * float(w) * float(h)

    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        if area_px < min_area or area_px > max_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        x = float(np.clip(cx * scale_x, 0.0, FIELD_W))
        y = float(np.clip(cy * scale_y, 0.0, FIELD_H))
        canopy = float(max(0.005, area_px * scale_x * scale_y))

        height = float(max(0.04, math.sqrt(canopy) * 0.75))
        leaf_count = int(max(1, round(2 + 10 * math.sqrt(min(1.0, canopy / 0.45)))))

        bx, by, bw, bh = cv2.boundingRect(cnt)
        extent = float(max(1, bw * bh))
        occ = float(np.clip(1.0 - area_px / extent, 0.0, 0.85))

        roi_mask = mask[by : by + bh, bx : bx + bw]
        roi_hsv = hsv[by : by + bh, bx : bx + bw]

        if roi_mask is not None and roi_mask.size > 0 and np.count_nonzero(roi_mask) > 0:
            mean_hsv = cv2.mean(roi_hsv, mask=roi_mask)
        else:
            mean_hsv = (45.0, 120.0, 120.0, 0.0)

        hue = float(np.clip(mean_hsv[0] * 2.0, 25.0, 145.0))
        saturation = float(np.clip(mean_hsv[1], 0.0, 255.0))
        value = float(np.clip(mean_hsv[2], 0.0, 255.0))

        ndvi = float(np.clip((saturation / 255.0) * 0.62 + (value / 255.0) * 0.28, 0.04, 0.96))
        conf = float(np.clip(0.32 + 0.38 * (area_px / max_area) + 0.20 * (saturation / 255.0), 0.08, 0.95))

        emb = make_embedding(
            height=height,
            area=canopy,
            leaf_count=leaf_count,
            ndvi=ndvi,
            hue=hue,
            saturation=saturation,
            value=value,
            occlusion=occ,
            row=nearest_row(y),
            col=nearest_col(x),
            density=0.12,
            rng=None,
        )

        dets.append(
            {
                "survey_id": survey_id,
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "area": canopy,
                "height": height,
                "leaf_count": leaf_count,
                "ndvi": ndvi,
                "hue": hue,
                "saturation": saturation,
                "value": value,
                "occlusion": occ,
                "conf": conf,
                "embedding": emb,
                "gt_id": None,
                "row": nearest_row(y),
                "col": nearest_col(x),
                "source": "upload",
            }
        )

    return dets


def sample_video_frames(video_bytes, suffix, max_frames=6):
    if not suffix:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total <= 0:
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append((i, frame, fps))
        else:
            idxs = np.unique(np.linspace(0, max(0, total - 1), min(max_frames, total), dtype=int))
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append((int(idx), frame, fps))

        cap.release()
    except Exception:
        frames = []
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

    return frames


def process_upload(data, file, append_to_db):
    if file is None:
        return None

    st.session_state.upload_counter = int(st.session_state.get("upload_counter", 0)) + 1
    base_ts = pd.Timestamp.now().replace(tzinfo=None)

    file_bytes = file.getvalue()
    ext = os.path.splitext(file.name)[1].lower()

    frames = []
    if ext in VIDEO_EXTS:
        frames = sample_video_frames(file_bytes, ext, max_frames=6)
    else:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames = [(0, frame, 1.0)]

    if not frames:
        return None

    added = []
    preview_frame = None
    preview_dets = []

    for n, (fidx, frame, fps) in enumerate(frames):
        seconds = float(fidx) / max(1.0, float(fps)) if fps and fps > 0 else float(n) * 2.0
        ts = base_ts + pd.Timedelta(seconds=seconds)
        sid = f"U-{st.session_state.upload_counter:03d}-{n + 1:02d}"

        dets = detect_from_frame(frame, sid, ts)

        if preview_frame is None:
            preview_frame = frame
            preview_dets = dets

        if append_to_db:
            rows, shift = data["tracker"].update(dets, sid, ts, source="upload")

            if rows:
                new_obs = pd.DataFrame(rows)
                data["obs"] = pd.concat([data["obs"], new_obs], ignore_index=True)

            survey_row = pd.DataFrame(
                [
                    {
                        "survey_id": sid,
                        "timestamp": ts,
                        "source": "upload",
                        "shift_x": shift[0],
                        "shift_y": shift[1],
                        "detection_count": len(dets),
                        "assigned_observations": len(rows),
                    }
                ]
            )
            data["surveys"] = pd.concat([data["surveys"], survey_row], ignore_index=True)
            data["frame_cache"][sid] = frame

            added.append(
                {
                    "survey_id": sid,
                    "timestamp": ts,
                    "detections": len(dets),
                    "assigned_tracks": len(rows),
                }
            )

    if append_to_db and added:
        refresh_derived_tables(data)

    return {
        "preview_frame": preview_frame,
        "preview_dets": preview_dets,
        "added": added,
        "appended": append_to_db,
    }


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------
def init_session():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "intro_done" not in st.session_state:
        st.session_state.intro_done = False
    if "current_survey" not in st.session_state:
        st.session_state.current_survey = 0
    if "selected_track" not in st.session_state:
        st.session_state.selected_track = "All"
    if "image_cache" not in st.session_state:
        st.session_state.image_cache = {}
    if "upload_counter" not in st.session_state:
        st.session_state.upload_counter = 0
    if "request_replay" not in st.session_state:
        st.session_state.request_replay = False


def sidebar_controls(data):
    st.sidebar.title("🎛️ Engine controls")

    n_surveys = max(1, len(data["surveys"]))
    if int(st.session_state.current_survey) >= n_surveys:
        st.session_state.current_survey = n_surveys - 1

    idx = st.sidebar.slider(
        "Survey timeline",
        min_value=0,
        max_value=n_surveys - 1,
        value=int(st.session_state.current_survey),
    )
    st.session_state.current_survey = int(idx)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶ Replay", use_container_width=True):
        st.session_state.request_replay = True
    if c2.button("⏹ Final", use_container_width=True):
        st.session_state.current_survey = n_surveys - 1

    delay = st.sidebar.slider("Replay delay (seconds)", 0.05, 1.5, 0.22, step=0.01)
    st.session_state.replay_delay = float(delay)

    show_ids = st.sidebar.checkbox("Show plant IDs", value=bool(st.session_state.get("show_ids", True)))
    st.session_state.show_ids = bool(show_ids)

    show_masks = st.sidebar.checkbox("Show segmentation masks", value=bool(st.session_state.get("show_masks", True)))
    st.session_state.show_masks = bool(show_masks)

    show_truth = st.sidebar.checkbox("Overlay ground truth", value=bool(st.session_state.get("show_truth", False)))
    st.session_state.show_truth = bool(show_truth)

    min_conf = st.sidebar.slider("Minimum track confidence", 0.0, 1.0, 0.25, step=0.05)
    st.session_state.min_conf = float(min_conf)

    anom = st.sidebar.slider("Anomaly sensitivity", 1.0, 4.0, 2.0, step=0.1)
    st.session_state.anom_thresh = float(anom)

    st.sidebar.subheader("🌿 Selected plant")
    track_options = ["All"]
    if not data["tracks"].empty and "track_id" in data["tracks"].columns:
        track_options += data["tracks"]["track_id"].astype(str).tolist()

    if st.session_state.selected_track not in track_options:
        st.session_state.selected_track = "All"

    selected = st.sidebar.selectbox(
        "Plant ID",
        track_options,
        index=track_options.index(st.session_state.selected_track),
    )
    st.session_state.selected_track = selected

    st.sidebar.caption(
        "The demo uses a built-in simulated instance-segmentation backend so it starts instantly "
        "without API keys or model downloads. Uploaded media is processed with OpenCV vegetation segmentation."
    )


def render_metric_row(data, container):
    tracks = data["tracks"]
    metrics = data["metrics"]

    active = int((tracks["status"] == "active").sum()) if not tracks.empty and "status" in tracks.columns else 0
    lost = int(tracks["status"].isin(["lost", "closed"]).sum()) if not tracks.empty and "status" in tracks.columns else 0
    threshold = float(st.session_state.get("anom_thresh", 2.0))
    anomalies = (
        int((tracks["max_anomaly_score"].fillna(0.0) >= threshold).sum())
        if not tracks.empty and "max_anomaly_score" in tracks.columns
        else 0
    )
    avg_conf = (
        _mean_or_zero(tracks["confidence"])
        if not tracks.empty and "confidence" in tracks.columns
        else 0.0
    )

    c1, c2, c3, c4, c5, c6 = container.columns(6)
    c1.metric("Active plant IDs", f"{active}")
    c2.metric("Lost / closed", f"{lost}")
    c3.metric("Anomalous plants", f"{anomalies}")
    c4.metric("Avg track confidence", f"{avg_conf:.2f}")
    c5.metric("ID-switch rate", f"{metrics.get('id_switch_rate', 0.0):.3f}")
    c6.metric("Detection F1", f"{metrics.get('detection_f1', 0.0):.3f}")


def render_selected_panel(data, idx, container):
    sel = st.session_state.get("selected_track", "All")
    surveys = data["surveys"]
    if surveys.empty:
        return

    idx = int(np.clip(idx, 0, len(surveys) - 1))
    sid = surveys.iloc[idx]["survey_id"]

    if sel == "All":
        container.subheader("Current survey summary")
        cur = data["obs"][data["obs"]["survey_id"] == sid].copy()
        if cur.empty:
            container.info("No assigned observations in this survey.")
            return

        top = cur.sort_values("anomaly_score", ascending=False).head(10)
        show_cols = [
            c
            for c in [
                "track_id",
                "gt_id",
                "x",
                "y",
                "canopy_area",
                "height_est",
                "confidence",
                "anomaly_score",
                "abnormal",
            ]
            if c in top.columns
        ]
        container.dataframe(top[show_cols], use_container_width=True)
        return

    tracks = data["tracks"]
    if tracks.empty or sel not in set(tracks["track_id"].astype(str)):
        container.warning("Selected plant is not available in the current temporal database.")
        return

    t = tracks[tracks["track_id"].astype(str) == sel].iloc[0]
    container.subheader(f"Plant identity: {sel}")

    m1, m2, m3, m4, m5 = container.columns(5)
    m1.metric("Status", str(t.get("status", "unknown")))
    m2.metric("Confidence", f"{_safe_float(t.get('confidence', 0.0), 0.0):.2f}")
    m3.metric("Canopy", f"{_safe_float(t.get('canopy_latest', 0.0), 0.0):.3f} m²")
    m4.metric("Height", f"{_safe_float(t.get('height_latest', 0.0), 0.0):.2f} m")
    m5.metric("Stage", str(t.get("growth_stage", "Unknown")))

    hist = data["obs"][data["obs"]["track_id"].astype(str) == sel].sort_values("timestamp")
    if hist.empty:
        container.info("No observation history for this plant.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["timestamp"],
            y=hist["canopy_area"],
            mode="lines+markers",
            name="Canopy area",
            line=dict(color="#2ca02c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["timestamp"],
            y=hist["height_est"],
            mode="lines+markers",
            name="Estimated height",
            yaxis="y2",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=35, r=35, t=30, b=30),
        yaxis=dict(title="Canopy (m²)"),
        yaxis2=dict(title="Height (m)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    container.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=hist["timestamp"],
            y=hist["confidence"],
            mode="lines+markers",
            name="Track confidence",
            line=dict(color="#ff7f0e", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=hist["timestamp"],
            y=hist["anomaly_score"],
            mode="lines+markers",
            name="Anomaly score",
            line=dict(color="#d62728", width=2),
        )
    )
    fig2.update_layout(
        height=250,
        margin=dict(l=35, r=35, t=30, b=30),
        yaxis=dict(title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    container.plotly_chart(fig2, use_container_width=True)


def render_live_content(data, container, idx):
    surveys = data["surveys"]
    if surveys.empty:
        container.info("No surveys loaded.")
        return

    idx = int(np.clip(idx, 0, len(surveys) - 1))
    sid = surveys.iloc[idx]["survey_id"]
    ts = surveys.iloc[idx]["timestamp"]

    show_ids = bool(st.session_state.get("show_ids", True))
    show_masks = bool(st.session_state.get("show_masks", True))
    show_truth = bool(st.session_state.get("show_truth", False))
    min_conf = float(st.session_state.get("min_conf", 0.25))
    anom_thresh = float(st.session_state.get("anom_thresh", 2.0))

    render_metric_row(data, container)

    container.markdown(f"### Survey replay: **{sid}** • {pd.to_datetime(ts).date()}")

    img = get_survey_image(data, idx, show_ids, show_masks, show_truth)

    col_img, col_map = container.columns([1.15, 1.0])
    with col_img:
        container.image(img, use_container_width=True)
        container.caption("Simulated UAV view with instance-segmentation masks and persistent plant IDs.")
    with col_map:
        container.plotly_chart(build_field_map(data, idx, show_ids, min_conf, anom_thresh), use_container_width=True)

    render_selected_panel(data, idx, container)


def render_live_tab(data):
    if st.session_state.get("request_replay", False):
        st.session_state.request_replay = False
        n = len(data["surveys"])
        if n == 0:
            st.info("No surveys available for replay.")
            return

        ph = st.empty()
        for i in range(n):
            st.session_state.current_survey = i
            render_live_content(data, ph, i)
            time.sleep(float(st.session_state.get("replay_delay", 0.22)))

        st.session_state.current_survey = max(0, n - 1)
        st.rerun()
        return

    render_live_content(data, st, int(st.session_state.current_survey))


def render_growth_tab(data):
    st.subheader("Individual growth curves and deviation analytics")

    obs = data["obs"]
    tracks = data["tracks"]

    if obs.empty or tracks.empty:
        st.info("No growth data available yet.")
        return

    metric = st.selectbox(
        "Growth metric",
        ["canopy_area", "height_est", "leaf_count", "ndvi"],
        index=0,
    )

    selected = st.session_state.get("selected_track", "All")
    threshold = float(st.session_state.get("anom_thresh", 2.0))

    if selected != "All":
        track_list = [selected]
    else:
        abnormal = tracks.sort_values("max_anomaly_score", ascending=False)["track_id"].head(6).astype(str).tolist()
        healthy = tracks[~tracks["abnormal"].fillna(False).astype(bool)]["track_id"].head(6).astype(str).tolist()
        track_list = list(dict.fromkeys(abnormal + healthy))

    fig = go.Figure()
    abnormal_set = set(tracks[tracks["abnormal"].fillna(False).astype(bool)]["track_id"].astype(str))
    for tid in track_list:
        grp = obs[obs["track_id"].astype(str) == tid].sort_values("timestamp")
        if grp.empty:
            continue
        is_ab = tid in abnormal_set
        color = "#d62728" if is_ab else "#2ca02c"
        fig.add_trace(
            go.Scatter(
                x=grp["timestamp"],
                y=grp[metric],
                mode="lines+markers",
                name=tid,
                line=dict(width=2, color=color),
                opacity=0.85,
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=35, r=15, t=35, b=35),
        xaxis_title="Survey date",
        yaxis_title=metric.replace("_", " "),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Growth-rate distribution")
        slopes = tracks["canopy_slope"].dropna() if "canopy_slope" in tracks.columns else pd.Series(dtype=float)
        if slopes.empty:
            st.info("Not enough temporal observations to fit growth slopes.")
        else:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=slopes, nbinsx=30, marker_color="#1f77b4"))
            fig_hist.update_layout(
                height=300,
                margin=dict(l=35, r=15, t=10, b=35),
                xaxis_title="Canopy growth slope (m²/day)",
                yaxis_title="Plants",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("#### Abnormal trajectory candidates")
        if "max_anomaly_score" not in tracks.columns:
            st.info("No anomaly scores available.")
        else:
            ab = tracks[tracks["max_anomaly_score"].fillna(0.0) >= threshold].copy()
            if ab.empty:
                st.success("No plants exceed the current anomaly threshold.")
            else:
                ab = ab.sort_values("max_anomaly_score", ascending=False)
                show_cols = [
                    c
                    for c in [
                        "track_id",
                        "primary_gt_id",
                        "status",
                        "x",
                        "y",
                        "canopy_latest",
                        "height_latest",
                        "canopy_slope",
                        "max_anomaly_score",
                        "confidence",
                        "cluster_id",
                    ]
                    if c in ab.columns
                ]
                st.dataframe(ab[show_cols].head(50), use_container_width=True)

    st.markdown("#### Spatial anomaly clusters")
    if "abnormal" not in obs.columns or "anomaly_score" not in obs.columns:
        st.info("No anomaly data available.")
        return

    abnormal_obs = obs[obs["abnormal"].fillna(False).astype(bool) & (obs["anomaly_score"] >= threshold)]
    if abnormal_obs.empty:
        st.info("No spatial anomaly cluster to display at the current threshold.")
    else:
        fig_cluster = go.Figure()
        customdata = list(
            zip(
                abnormal_obs["track_id"].astype(str),
                abnormal_obs["anomaly_score"].astype(float),
                abnormal_obs["cluster_id"].astype(int),
            )
        )
        fig_cluster.add_trace(
            go.Scatter(
                x=abnormal_obs["x"],
                y=abnormal_obs["y"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=abnormal_obs["cluster_id"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(width=1, color="white"),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Anomaly: %{customdata[1]:.2f}<br>"
                    "Cluster: %{customdata[2]}<br>"
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig_cluster.update_layout(
            height=420,
            margin=dict(l=35, r=15, t=20, b=35),
            xaxis=dict(title="Field X (m)", range=[-2, FIELD_W + 2], zeroline=False),
            yaxis=dict(title="Field Y (m)", range=[FIELD_H + 2, -2], zeroline=False),
        )
        st.plotly_chart(fig_cluster, use_container_width=True)


def render_identity_tab(data):
    st.subheader("Persistent identity, re-ID confidence, and track events")

    tracker = data["tracker"]
    tracks = data["tracks"]
    obs = data["obs"]
    events = data["events"]

    c1, c2 = st.columns([1.1, 0.9])

    with c1:
        st.markdown("#### Track confidence timeline")
        selected = st.session_state.get("selected_track", "All")

        if selected != "All":
            track_list = [selected]
        else:
            if tracks.empty or "track_id" not in tracks.columns:
                track_list = []
            else:
                track_list = tracks.sort_values("max_anomaly_score", ascending=False)["track_id"].head(8).astype(str).tolist()

        fig = go.Figure()
        for tid in track_list:
            grp = obs[obs["track_id"].astype(str) == tid].sort_values("timestamp")
            if grp.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=grp["timestamp"],
                    y=grp["confidence"],
                    mode="lines+markers",
                    name=tid,
                )
            )
        fig.update_layout(
            height=360,
            margin=dict(l=35, r=15, t=20, b=35),
            xaxis_title="Survey date",
            yaxis_title="Track confidence",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Appearance embedding nearest neighbors")
        if not tracker.tracks:
            st.info("No tracks available.")
        else:
            sel = st.session_state.get("selected_track", "All")
            if sel == "All" or sel not in tracker.tracks:
                sel = list(tracker.tracks.keys())[0]

            base = tracker.tracks[sel]["embedding"]
            rows = []
            for tid, t in tracker.tracks.items():
                if tid == sel:
                    continue
                sim = float(np.dot(np.asarray(base, dtype=float), np.asarray(t["embedding"], dtype=float)))
                rows.append(
                    {
                        "candidate_track": tid,
                        "primary_gt_id": t.get("last_gt_id"),
                        "status": t.get("status"),
                        "cosine_similarity": sim,
                        "confidence": t.get("confidence"),
                    }
                )

            st.caption(f"Nearest visual/geometric identities to **{sel}**")
            if rows:
                nn = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).head(12)
                st.dataframe(nn, use_container_width=True, hide_index=True)
            else:
                st.info("No other tracks available for nearest-neighbor comparison.")

    st.markdown("#### Identity events")
    if events.empty:
        st.info("No track events recorded.")
    else:
        options = sorted(events["type"].dropna().unique().tolist())
        chosen = st.multiselect("Event types", options, default=options)
        ev = events[events["type"].isin(chosen)].copy()
        ev = ev.sort_values("timestamp", ascending=False)
        show_cols = [c for c in ["timestamp", "survey_id", "type", "track_id", "gt_id", "message", "severity"] if c in ev.columns]
        st.dataframe(ev[show_cols], use_container_width=True, hide_index=True)


def render_temporal_tab(data):
    st.subheader("Temporal plant observation database")

    obs = data["obs"]
    tracks = data["tracks"]
    surveys = data["surveys"]
    events = data["events"]
    truth = data["truth"]

    f1, f2, f3 = st.columns(3)

    survey_options = ["All"] + (surveys["survey_id"].astype(str).tolist() if not surveys.empty else [])
    survey_filter = f1.selectbox("Survey", survey_options, index=0)

    track_options = ["All"] + (tracks["track_id"].astype(str).tolist() if not tracks.empty else [])
    track_filter = f2.selectbox("Track ID", track_options, index=0)

    gt_query = f3.text_input("Ground-truth ID contains", value="")

    filtered = obs.copy()
    if survey_filter != "All":
        filtered = filtered[filtered["survey_id"].astype(str) == survey_filter]
    if track_filter != "All":
        filtered = filtered[filtered["track_id"].astype(str) == track_filter]
    if gt_query and "gt_id" in filtered.columns:
        filtered = filtered[filtered["gt_id"].astype(str).str.contains(gt_query, case=False, na=False)]

    st.metric("Observations shown", f"{len(filtered):,}")

    display_cols = [
        "survey_id",
        "timestamp",
        "track_id",
        "gt_id",
        "x",
        "y",
        "canopy_area",
        "height_est",
        "leaf_count",
        "ndvi",
        "confidence",
        "detection_conf",
        "match_cost",
        "occlusion",
        "anomaly_score",
        "abnormal",
        "cluster_id",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    if display_cols:
        st.dataframe(filtered[display_cols].head(2000), use_container_width=True, hide_index=True)
    else:
        st.dataframe(filtered.head(2000), use_container_width=True, hide_index=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download observations CSV",
            data=obs.to_csv(index=False).encode("utf-8"),
            file_name="plant_observations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download tracks CSV",
            data=tracks.to_csv(index=False).encode("utf-8"),
            file_name="plant_tracks.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Download events CSV",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name="identity_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            "Download ground truth CSV",
            data=truth.to_csv(index=False).encode("utf-8"),
            file_name="ground_truth_observations.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_evaluation_tab(data):
    st.subheader("Detection, tracking, and growth-estimation evaluation")

    metrics = data["metrics"]
    det_eval = data["det_eval"]
    obs = data["obs"]
    truth = data["truth"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Detection precision", f"{metrics.get('detection_precision', 0.0):.3f}")
    c2.metric("Detection recall", f"{metrics.get('detection_recall', 0.0):.3f}")
    c3.metric("Detection F1", f"{metrics.get('detection_f1', 0.0):.3f}")
    c4.metric("ID-switch rate", f"{metrics.get('id_switch_rate', 0.0):.3f}")
    c5.metric("Track persistence", f"{metrics.get('track_persistence', 0.0):.3f}")
    c6.metric("Mean track coverage", f"{metrics.get('mean_track_coverage', 0.0):.3f}")

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Height MAE (m)", f"{metrics.get('height_mae', 0.0):.3f}")
    c8.metric("Height RMSE (m)", f"{metrics.get('height_rmse', 0.0):.3f}")
    c9.metric("Canopy MAE (m²)", f"{metrics.get('canopy_mae', 0.0):.3f}")
    c10.metric("Canopy RMSE (m²)", f"{metrics.get('canopy_rmse', 0.0):.3f}")
    c11.metric("NDVI MAE", f"{metrics.get('ndvi_mae', 0.0):.3f}")
    c12.metric("Track loss events", f"{metrics.get('track_loss_events', 0)}")

    st.markdown("#### Per-survey detector and tracker quality")
    if det_eval.empty:
        st.info("No detection evaluation records available.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=det_eval["timestamp"], y=det_eval["precision"], name="Precision"))
        fig.add_trace(go.Bar(x=det_eval["timestamp"], y=det_eval["recall"], name="Recall"))
        fig.add_trace(go.Scatter(x=det_eval["timestamp"], y=det_eval["f1"], name="F1", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=det_eval["timestamp"], y=det_eval["tracked_recall"], name="Tracked recall", mode="lines+markers"))
        fig.update_layout(
            height=380,
            margin=dict(l=35, r=15, t=30, b=35),
            barmode="group",
            xaxis_title="Survey date",
            yaxis_title="Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Growth estimation error")
    matched = obs[obs["gt_id"].notna()].copy() if not obs.empty else pd.DataFrame()
    if matched.empty or truth.empty:
        st.info("No ground-truth matched observations available for growth-error plotting.")
    else:
        merged = matched.merge(truth, on=["survey_id", "gt_id"], how="inner")
        if merged.empty:
            st.info("No merged ground-truth growth samples.")
        else:
            e1, e2 = st.columns(2)

            with e1:
                fig_sc = go.Figure()
                fig_sc.add_trace(
                    go.Scatter(
                        x=merged["canopy_true"],
                        y=merged["canopy_area"],
                        mode="markers",
                        marker=dict(size=7, opacity=0.55, color="#2ca02c"),
                        name="Canopy",
                    )
                )
                max_v = _safe_max(np.concatenate([merged["canopy_true"].to_numpy(), merged["canopy_area"].to_numpy()]), default=1.0)
                fig_sc.add_trace(
                    go.Scatter(
                        x=[0, max_v],
                        y=[0, max_v],
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Perfect",
                    )
                )
                fig_sc.update_layout(
                    height=360,
                    margin=dict(l=35, r=15, t=20, b=35),
                    xaxis_title="True canopy (m²)",
                    yaxis_title="Estimated canopy (m²)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            with e2:
                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Scatter(
                        x=merged["height_true"],
                        y=merged["height_est"],
                        mode="markers",
                        marker=dict(size=7, opacity=0.55, color="#1f77b4"),
                        name="Height",
                    )
                )
                max_h = _safe_max(np.concatenate([merged["height_true"].to_numpy(), merged["height_est"].to_numpy()]), default=1.0)
                fig_h.add_trace(
                    go.Scatter(
                        x=[0, max_h],
                        y=[0, max_h],
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Perfect",
                    )
                )
                fig_h.update_layout(
                    height=360,
                    margin=dict(l=35, r=15, t=20, b=35),
                    xaxis_title="True height (m)",
                    yaxis_title="Estimated height (m)",
                )
                st.plotly_chart(fig_h, use_container_width=True)

    st.markdown(
        """
        **Interpretation**
        - Detection metrics evaluate the segmentation/detection layer against simulated visible plants.
        - ID-switch rate measures how often a persistent track changes underlying plant identity.
        - Track persistence measures whether each true plant is represented by one stable `plant_id`.
        - Growth error compares per-plant estimated canopy, height, and color statistics to ground truth.
        """
    )


def render_upload_tab(data):
    st.subheader("Accept fixed camera, smartphone, or UAV image/video input")

    st.markdown(
        """
        Upload a field image or short video. The app extracts vegetation instances using OpenCV
        color/shape segmentation, builds appearance embeddings, and optionally appends the survey
        to the same temporal identity database used by the live demo.
        """
    )

    file = st.file_uploader(
        "Upload media",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
        key="crop_media_uploader",
    )

    append = st.checkbox("Append processed survey(s) to temporal database", value=True)

    if file is not None:
        if st.button("Run CV pipeline on uploaded media", use_container_width=True):
            with st.spinner("Segmenting vegetation, extracting embeddings, updating persistent identities..."):
                try:
                    result = process_upload(data, file, append_to_db=append)
                except Exception as exc:
                    result = None
                    st.error(f"Media processing failed: {exc}")

            if result is None:
                st.warning("Could not decode media or no frames were extracted.")
            else:
                if result["preview_frame"] is not None:
                    preview = draw_preview(result["preview_frame"], result["preview_dets"])
                    st.image(preview, use_container_width=True)
                    st.caption("Preview of detected plant instances on the first processed frame.")

                st.metric("Detections in preview frame", f"{len(result['preview_dets'])}")

                if result["appended"] and result["added"]:
                    st.success(f"Appended {len(result['added'])} survey(s) to the temporal database.")
                    st.dataframe(pd.DataFrame(result["added"]), use_container_width=True, hide_index=True)
                elif not result["appended"]:
                    st.info("Preview only. Check the append option to add this survey to the database.")


def render_dashboard(data):
    st.title("🌱 Individual Crop Plant Tracking & Growth Identity Engine")
    st.caption(
        "Instance-segmentation-driven plant detection • persistent `plant_id` re-identification • "
        "temporal growth curves • anomaly and track-loss analytics • replayable field surveys."
    )

    sidebar_controls(data)

    tabs = st.tabs(
        [
            "Live Field Map",
            "Growth Analytics",
            "Identity & Re-ID",
            "Temporal DB",
            "Evaluation",
            "Upload Media",
        ]
    )

    with tabs[0]:
        render_live_tab(data)
    with tabs[1]:
        render_growth_tab(data)
    with tabs[2]:
        render_identity_tab(data)
    with tabs[3]:
        render_temporal_tab(data)
    with tabs[4]:
        render_evaluation_tab(data)
    with tabs[5]:
        render_upload_tab(data)


def run_intro_animation(data):
    st.title("🌱 Individual Crop Plant Tracking & Growth Identity Engine")
    st.subheader("Automatic live demo starting")
    st.caption(
        "Generating realistic multi-survey UAV data, simulating instance segmentation, "
        "assigning persistent plant IDs, and building the temporal growth database."
    )

    progress = st.progress(0.0)
    placeholder = st.empty()

    surveys = data["surveys"]
    n = len(surveys)

    if n == 0:
        st.session_state.intro_done = True
        st.rerun()
        return

    for i in range(n):
        st.session_state.current_survey = i
        sid = surveys.iloc[i]["survey_id"]
        ts = surveys.iloc[i]["timestamp"]
        obs_count = int((data["obs"]["survey_id"] == sid).sum()) if not data["obs"].empty else 0

        img = get_survey_image(data, i, show_ids=True, show_masks=True, show_truth=False)

        placeholder.empty()
        placeholder.image(img, use_container_width=True)
        placeholder.markdown(
            f"**Survey {i + 1}/{n}** • `{sid}` • {pd.to_datetime(ts).date()} • "
            f"assigned plant observations: **{obs_count}**"
        )

        progress.progress(float((i + 1) / max(1, n)))
        time.sleep(0.18)

    st.session_state.current_survey = max(0, n - 1)
    st.session_state.intro_done = True
    st.balloons()
    time.sleep(0.35)
    st.rerun()


def main():
    init_session()

    if not st.session_state.initialized:
        with st.spinner("Building demo field, temporal database, and identity tracks..."):
            st.session_state.data = build_demo_data()
            st.session_state.initialized = True

    data = st.session_state.data

    if not st.session_state.intro_done:
        run_intro_animation(data)
        return

    render_dashboard(data)


if __name__ == "__main__":
    main()
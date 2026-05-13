"""
biomor_multiprocess_controller.py

Architecture: Master-Worker Multiprocess
- Master: CustomTkinter 60Hz Dashboard (non-blocking)
- Worker: PsychoPy Surround 3840x1080 (pure render + Arduino closed-loop)
- Execution: Fully automated session flow, zero manual triggers.
"""

import csv
import json
import math
import os
import sys
import random
import threading
import multiprocessing as mp
import queue
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk

try:
    import serial
    import serial.tools.list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ==============================================================================
# IPC Queues
# ==============================================================================
cmd_queue: Optional[mp.Queue] = None
telemetry_queue: Optional[mp.Queue] = None


def create_ipc_queues() -> Tuple[mp.Queue, mp.Queue]:
    global cmd_queue, telemetry_queue
    cmd_queue = mp.Queue(maxsize=32)
    telemetry_queue = mp.Queue(maxsize=256)
    return cmd_queue, telemetry_queue


# ==============================================================================
# Paradigms & Matrix
# ==============================================================================
EXPERIMENT_PATTERNS = {
    "Baseline Visual": {
        "type": "baseline_visual",
        "target_ttc_ms": None,
        "lv_ratio_ms": 100,
    },
    "Baseline Wind": {
        "type": "baseline_wind",
        "target_ttc_ms": None,
        "lv_ratio_ms": None,
    },
    "Looming + Wind (TTC -373ms / 30°)": {
        "type": "looming_wind",
        "target_ttc_ms": -373,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC -308ms / 36°)": {
        "type": "looming_wind",
        "target_ttc_ms": -308,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC -261ms / 42°)": {
        "type": "looming_wind",
        "target_ttc_ms": -261,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC -225ms / 48°)": {
        "type": "looming_wind",
        "target_ttc_ms": -225,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC -119ms / 80°)": {
        "type": "looming_wind",
        "target_ttc_ms": -119,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC 0ms / 180°)": {
        "type": "looming_wind",
        "target_ttc_ms": 0,
        "lv_ratio_ms": 100,
    },
    "Looming + Wind (TTC +200ms)": {
        "type": "looming_wind",
        "target_ttc_ms": 200,
        "lv_ratio_ms": 100,
    },
}
PATTERN_CHOICES = list(EXPERIMENT_PATTERNS.keys())


def generate_trial_matrix(pattern_key: str) -> List[Dict[str, Any]]:
    p = EXPERIMENT_PATTERNS[pattern_key]
    trials = []
    for direction in ["left"] * 9 + ["right"] * 9:
        d = {
            "type": p["type"],
            "target_ttc_ms": p["target_ttc_ms"],
            "lv_ratio_ms": p["lv_ratio_ms"],
        }
        if p["type"] == "baseline_visual":
            d["wind_dir"], d["screen_side"] = "none", direction
        else:
            d["wind_dir"], d["screen_side"] = direction, direction
        trials.append(d)
    random.shuffle(trials)
    return trials


# ==============================================================================
# Hardware Daemons
# ==============================================================================
class SerialDaemon:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.data_queue = queue.Queue(maxsize=8192)
        self._serial = None
        self._running = False

    def start(self):
        self._serial = serial.Serial(
            port=self.port, baudrate=self.baudrate, timeout=self.timeout
        )
        self._serial.reset_input_buffer()
        self._running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        from psychopy import core

        while self._running:
            try:
                raw = self._serial.readline()
                if not raw:
                    continue
                t_psy = core.getTime()
                parts = raw.decode("ascii", errors="ignore").strip().split(",")
                if len(parts) == 5:
                    self.data_queue.put_nowait(
                        (
                            t_psy,
                            int(parts[0]),
                            int(parts[1]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                        )
                    )
            except Exception:
                pass

    def send_arm_command(self, direction: str, delay_ms: int):
        cmd = f"<{('R' if direction == 'right' else 'L')},{delay_ms}>"
        if self._serial and self._serial.is_open:
            self._serial.write(cmd.encode("ascii"))
            print(f"[Serial] Armed: {cmd}")

    def drain_queue(self):
        items = []
        while True:
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def stop(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()


class MockSerialDaemon:
    def __init__(self):
        self.data_queue = queue.Queue()

    def start(self):
        pass

    def send_arm_command(self, d: str, ms: int):
        print(f"[MockArm] <{('R' if d=='right' else 'L')},{ms}>")

    def drain_queue(self):
        return []

    def stop(self):
        pass


# ==============================================================================
# Ground Truth Logger
# ==============================================================================
class GroundTruthLogger:
    COLUMNS = [
        "event_name",
        "timestamp",
        "session_num",
        "trial_in_session",
        "global_trial_id",
        "t_ard",
        "dx",
        "dy",
        "dz",
        "stim_state",
        "details",
    ]

    def __init__(self, output_dir: str):
        self.out = output_dir
        os.makedirs(self.out, exist_ok=True)
        self.g_tid, self.s_num, self.t_num = self._load_cache(), 0, 0
        self._file, self._writer = None, None

    def _load_cache(self):
        p = os.path.join(self.out, ".trial_cache.txt")
        return int(open(p).read().strip()) if os.path.exists(p) else 0

    def open_session(self, path: str, s_num: int):
        self.close()
        self.s_num, self.t_num = s_num, 0
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def is_open(self):
        return self._writer is not None

    def close(self):
        if self._file:
            self._file.close()
            self._file, self._writer = None, None

    def advance_trial(self):
        self.t_num += 1
        self.g_tid += 1
        with open(os.path.join(self.out, ".trial_cache.txt"), "w") as f:
            f.write(str(self.g_tid))

    def log_motion_batch(self, tuples: List[tuple]):
        if not self._writer:
            return
        for t_p, t_a, dx, dy, dz, s in tuples:
            self._writer.writerow(
                [
                    "motion",
                    f"{t_p:.6f}",
                    self.s_num,
                    self.t_num,
                    self.g_tid,
                    t_a,
                    dx,
                    dy,
                    dz,
                    s,
                    "",
                ]
            )

    def log_event(self, event_name: str, timestamp: float, **kwargs):
        if not self._writer:
            return
        self._writer.writerow(
            [
                event_name,
                f"{timestamp:.6f}",
                self.s_num,
                self.t_num,
                self.g_tid,
                "",
                "",
                "",
                "",
                "",
                json.dumps(kwargs),
            ]
        )

    def flush(self):
        if self._file:
            self._file.flush()


# ==============================================================================
# Worker Process: PsychoPy Render & Logic
# ==============================================================================
class LoomingWorker:
    VIEWING_DISTANCE_CM = 30.0
    SINGLE_SCREEN_WIDTH_CM = 53.0
    SINGLE_SCREEN_WIDTH_PX = 1920
    LEFT_CENTRE_PX = (-960, 0)
    RIGHT_CENTRE_PX = (960, 0)

    def __init__(self, config: Dict[str, Any], cmd_q: mp.Queue, telemetry_q: mp.Queue):
        self.config, self.cmd_queue, self.telemetry_queue = config, cmd_q, telemetry_q

        self.pattern_key = config["Experiment Pattern"]
        self.s_start, self.s_total = int(config["Session Number"]), int(
            config["Total Sessions"]
        )
        self.iti_min, self.iti_max = map(float, config["ITI Range (sec)"].split("-"))
        self.isi_min, self.isi_max = map(float, config["ISI Range (sec)"].split("-"))

        self.debug_mode = config.get("Debug Mode", False)
        self.output_dir = config.get("_output_dir", ".")
        self.init_deg = 2.0
        self.max_deg = 180.0

        if self.debug_mode:
            self.win_size, self.is_fullscr, self.c_l, self.c_r, self.d_scale = (
                (1200, 600),
                False,
                -300,
                300,
                0.3,
            )
            self.mask_size = (600, 600)
        else:
            self.win_size, self.is_fullscr, self.c_l, self.c_r, self.d_scale = (
                (3840, 1080),
                True,
                -960,
                960,
                1.0,
            )
            self.mask_size = (1920, 1080)

        self.init_r_px = self._deg_to_pix(self.init_deg)
        self.sync_size_px = self._deg_to_pix(2.0)

        from psychopy import core

        self.clock = core.Clock()
        self.abort_flag = False
        self._last_kin = {"dx": "—", "dy": "—", "dz": "—", "valve": "—"}

    def _deg_to_pix(self, deg: float) -> float:
        r_cm = math.tan(math.radians(deg / 2.0)) * self.VIEWING_DISTANCE_CM
        return (
            r_cm
            * (self.SINGLE_SCREEN_WIDTH_PX / self.SINGLE_SCREEN_WIDTH_CM)
            * self.d_scale
        )

    def _push(self, frame: dict):
        try:
            self.telemetry_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _check_abort(self) -> bool:
        if self.abort_flag:
            return True
        try:
            if self.cmd_queue.get_nowait().get("action") == "ABORT":
                return True
        except queue.Empty:
            pass
        return False

    def _drain(self):
        latest = self.serial_daemon.drain_queue()
        if latest and self.logger and self.logger.is_open():
            self.logger.log_motion_batch(latest)

        if latest:
            latest_data = latest[-1]
            self._last_kin = {
                "dx": latest_data[2],
                "dy": latest_data[3],
                "dz": latest_data[4],
                "valve": latest_data[5],
            }
        return getattr(
            self, "_last_kin", {"dx": "—", "dy": "—", "dz": "—", "valve": "—"}
        )

    def run(self):
        from psychopy import core, event
        from psychopy import visual

        self.core = core
        event.globalKeys.add(
            key="escape", func=lambda: setattr(self, "abort_flag", True)
        )

        self._push({"action": "worker_ready"})
        self.logger = GroundTruthLogger(self.output_dir)
        sp = self.config.get("Serial Port", "mock")
        self.serial_daemon = (
            MockSerialDaemon() if sp == "mock" or not HAS_SERIAL else SerialDaemon(sp)
        )
        self.serial_daemon.start()

        self.win = visual.Window(
            size=self.win_size,
            fullscr=self.is_fullscr,
            screen=(
                self.config.get("Stimulus Screen ID", 1) if not self.debug_mode else 0
            ),
            color=[0, 0, 0],
            colorSpace="rgb",
            units="pix",
            waitBlanking=not self.debug_mode,
        )

        # Stimuli
        self.stim_l = visual.Circle(
            self.win, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1], pos=(self.c_l, 0)
        )
        self.stim_r = visual.Circle(
            self.win, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1], pos=(self.c_r, 0)
        )
        self.mask_l = visual.Rect(
            self.win,
            width=self.mask_size[0],
            height=self.mask_size[1],
            pos=(self.c_l, 0),
            fillColor=[0, 0, 0],
        )
        self.mask_r = visual.Rect(
            self.win,
            width=self.mask_size[0],
            height=self.mask_size[1],
            pos=(self.c_r, 0),
            fillColor=[0, 0, 0],
        )

        # Photodiode Sync Patches (Bottom outer corners)
        sync_y = -self.mask_size[1] / 2 + self.sync_size_px / 2
        sync_offset_x = self.mask_size[0] / 2 - self.sync_size_px / 2
        self.sync_l = visual.Rect(
            self.win,
            width=self.sync_size_px,
            height=self.sync_size_px,
            pos=(self.c_l - sync_offset_x, sync_y),
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
        )
        self.sync_r = visual.Rect(
            self.win,
            width=self.sync_size_px,
            height=self.sync_size_px,
            pos=(self.c_r + sync_offset_x, sync_y),
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
        )

        try:
            self._baseline_render()
            self._drain()

            # Auto-adaptation phase (5 seconds) instead of manual trigger
            t0 = self.clock.getTime()
            while self.clock.getTime() - t0 < 5.0:
                if self._check_abort():
                    return
                self._baseline_render()
                kin = self._drain()
                self._push(
                    {
                        "action": "telemetry",
                        "phase": "Adaptation",
                        "session_num": 0,
                        **kin,
                    }
                )
                self.win.flip()

            for s_idx in range(self.s_total):
                curr_s = self.s_start + s_idx
                trials = generate_trial_matrix(self.pattern_key)
                self.logger.open_session(
                    os.path.join(
                        self.output_dir, f"{self.config['Subject ID']}_s{curr_s}.csv"
                    ),
                    curr_s,
                )

                for t_idx, trial in enumerate(trials):
                    if self._check_abort():
                        return

                    if t_idx > 0:
                        iti = random.uniform(self.iti_min, self.iti_max)
                        self._wait_iti(iti, trial, curr_s, t_idx, len(trials))
                    else:
                        self._pre_arm(trial)

                    self._run_trial(t_idx, trial, curr_s, len(trials))
                    self.logger.flush()

                self.logger.close()
                if s_idx < self.s_total - 1:
                    self._wait_isi(
                        random.uniform(self.isi_min, self.isi_max), curr_s + 1
                    )

            self._push({"action": "worker_done"})
        except Exception:
            self._push({"action": "worker_error"})
        finally:
            self.logger.close()
            self.serial_daemon.stop()
            self.win.close()
            self.core.quit()

    def _baseline_render(self):
        self.mask_l.draw()
        self.mask_r.draw()
        self.stim_l.radius = self.init_r_px
        self.stim_r.radius = self.init_r_px
        self.stim_l.draw()
        self.stim_r.draw()

    def _pre_arm(self, trial):
        if trial["type"] in ["baseline_visual", "baseline_wind"]:
            return
        if trial.get("target_ttc_ms") is None:
            return

        lv_s = trial.get("lv_ratio_ms", 100) / 1000.0
        t_col_s = lv_s / math.tan(math.radians(self.init_deg / 2))
        delay_ms = max(0, int(round(t_col_s * 1000)) + trial["target_ttc_ms"])
        self.serial_daemon.send_arm_command(trial["screen_side"], delay_ms)
        telemetry = self._drain()
        self._push(
            {
                **telemetry,
                "action": "telemetry",
                "phase": "Baseline+Wind",
                "delay": delay_ms,
                "side": trial["screen_side"],
            }
        )

    def _wait_iti(self, dur: float, next_t: dict, s_num: int, t_idx: int, tot: int):
        self.logger.log_event("iti_start", self.clock.getTime(), duration=dur)
        self._pre_arm(next_t)
        t0 = self.clock.getTime()
        while self.clock.getTime() - t0 < dur:
            if self._check_abort():
                self.core.quit()
            self._baseline_render()
            self.win.flip()
            kin = self._drain()
            self._push(
                {
                    "action": "telemetry",
                    "phase": f"ITI ({self.clock.getTime()-t0:.1f}s)",
                    "session_num": s_num,
                    "trial_idx": t_idx,
                    "total_trials": tot,
                    "theta": self.init_deg,
                    "side": "—",
                    **kin,
                }
            )

    def _wait_isi(self, dur: float, next_s: int):
        t0 = self.clock.getTime()
        while self.clock.getTime() - t0 < dur:
            if self._check_abort():
                self.core.quit()
            self._baseline_render()
            self.win.flip()
            kin = self._drain()
            self._push(
                {
                    "action": "telemetry",
                    "phase": f"ISI to S{next_s}",
                    "session_num": next_s - 1,
                    "theta": self.init_deg,
                    "side": "—",
                    **kin,
                }
            )

    def _run_trial(self, t_idx: int, trial: dict, s_num: int, tot: int):
        self.logger.advance_trial()
        t_start = self.clock.getTime()
        self.logger.log_event("trial_start", t_start, **trial)

        side = trial["screen_side"]
        active_stim, inact_stim = (
            (self.stim_r, self.stim_l)
            if side == "right"
            else (self.stim_l, self.stim_r)
        )
        active_mask, sync = (
            (self.mask_l, self.sync_r)
            if side == "right"
            else (self.mask_r, self.sync_l)
        )

        if trial["type"] in ["baseline_visual", "looming_wind"]:
            lv_s = trial["lv_ratio_ms"] / 1000.0
            t_col = lv_s / math.tan(math.radians(self.init_deg / 2))
            t0 = self.clock.getTime()
            theta = self.init_deg

            while True:
                if self._check_abort():
                    self.core.quit()
                elap = self.clock.getTime() - t0

                if elap >= t_col + 1.0:
                    break

                theta = self.init_deg
                theta = (
                    math.degrees(2 * math.atan(lv_s / (t_col - elap)))
                    if elap < t_col
                    else self.max_deg
                )
                theta = min(theta, self.max_deg)

                active_stim.radius = self._deg_to_pix(theta)
                active_stim.draw()
                active_mask.draw()
                inact_stim.radius = self.init_r_px
                inact_stim.draw()
                sync.draw()  # Constant high TTL for photodiode

                self.win.flip()

                kin = self._drain()
                self._push(
                    {
                        "action": "telemetry",
                        "phase": "Looming",
                        "theta": theta,
                        "side": side,
                        "session_num": s_num,
                        "trial_idx": t_idx + 1,
                        "total_trials": tot,
                        **kin,
                    }
                )

        elif trial["type"] == "baseline_wind":
            delay, post = random.uniform(0.1, 1.2), random.uniform(1.0, 2.0)
            t0 = self.clock.getTime()
            trig = False

            while True:
                if self._check_abort():
                    self.core.quit()
                now = self.clock.getTime()
                if trig and (now - t0 - delay) >= post:
                    break

                if not trig and (now - t0) >= delay:
                    self.serial_daemon.send_arm_command(trial["wind_dir"], 0)
                    trig = True

                self._baseline_render()
                self.win.flip()
                kin = self._drain()
                self._push(
                    {
                        "action": "telemetry",
                        "phase": "Baseline",
                        "theta": self.init_deg,
                        "side": "both",
                        **kin,
                    }
                )


# ==============================================================================
# Master Dashboard
# ==============================================================================
class MasterDashboard:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._worker = None
        self._running = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("BioMoR — Experiment Dashboard")
        self.root.resizable(True, True)
        try:
            self.root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass

        self.FONT_LABEL = ("Segoe UI", 13)
        self.FONT_HEADING = ("Segoe UI", 15, "bold")
        self.FONT_BUTTON = ("Segoe UI", 14, "bold")

        self._build_ui()
        self._center_window()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(18, 4))
        ctk.CTkLabel(
            header,
            text="BioMoR — Experiment Dashboard",
            font=("Segoe UI", 18, "bold"),
            text_color=("#1f6aa5", "#3b8ed0"),
        ).pack()

        # Main grid: left = status card, right = digital twin
        main_grid = ctk.CTkFrame(self.root, fg_color="transparent")
        main_grid.pack(fill="both", expand=True, padx=24, pady=(8, 4))
        main_grid.grid_columnconfigure(0, weight=1)
        main_grid.grid_columnconfigure(1, weight=1)

        # LEFT: Experiment Status Card
        left_card = self._card(main_grid, "Experiment Status")
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.lbl_status = ctk.CTkLabel(
            left_card,
            text="● Idle",
            font=self.FONT_HEADING,
            text_color="gray",
        )
        self.lbl_status.pack(anchor="w", padx=14, pady=(4, 2))

        status_grid = ctk.CTkFrame(left_card, fg_color="transparent")
        status_grid.pack(fill="x", padx=14, pady=(4, 8))

        rows = [
            ("Subject:", "subject_val"),
            ("Session:", "session_val"),
            ("Trial:", "trial_val"),
            ("Phase:", "phase_val"),
            ("Side:", "side_val"),
            ("Theta (°):", "theta_val"),
        ]
        for i, (label, attr) in enumerate(rows):
            ctk.CTkLabel(
                status_grid,
                text=label,
                font=self.FONT_LABEL,
                text_color=("gray35", "gray70"),
            ).grid(row=i, column=0, sticky="w", pady=(2, 2))
            val_label = ctk.CTkLabel(
                status_grid,
                text="—",
                font=self.FONT_LABEL,
                text_color=("gray85", "gray90"),
            )
            val_label.grid(row=i, column=1, sticky="w", padx=(12, 0), pady=(2, 2))
            setattr(self, f"lbl_{attr}", val_label)

        # Hardware row
        ctk.CTkLabel(
            status_grid,
            text="Hardware:",
            font=self.FONT_LABEL,
            text_color=("gray35", "gray70"),
        ).grid(row=len(rows), column=0, sticky="w", pady=(8, 2))
        self.lbl_hardware = ctk.CTkLabel(
            status_grid,
            text="dx: —  dy: —  dz: —  Valve: —",
            font=self.FONT_LABEL,
            text_color="cyan",
        )
        self.lbl_hardware.grid(
            row=len(rows), column=1, sticky="w", padx=(12, 0), pady=(8, 2)
        )

        # RIGHT: Digital Twin Monitor Card
        right_card = self._card(main_grid, "Digital Twin — Surround Display")
        right_card.grid(row=0, column=1, sticky="nsew")

        self.canvas = ctk.CTkCanvas(
            right_card,
            width=400,
            height=150,
            bg="black",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.canvas.pack(padx=14, pady=(4, 14))

        # Scale factors: canvas 400×150 maps to 3840×1080
        self._canvas_scale_x = 400.0 / 3840.0
        self._canvas_scale_y = 150.0 / 1080.0

        # Action bar
        action_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        action_frame.pack(fill="x", padx=24, pady=(8, 20))

        self.btn_start = ctk.CTkButton(
            action_frame,
            text="▶  Start Experiment",
            command=self._on_start,
            font=self.FONT_BUTTON,
            corner_radius=8,
            height=42,
            fg_color=("#1f6aa5", "#3b8ed0"),
            hover_color=("#154a78", "#2b6fa0"),
        )
        self.btn_start.pack(side="left", padx=(0, 12))

        self.btn_abort = ctk.CTkButton(
            action_frame,
            text="⏹  Abort",
            command=self._on_abort,
            font=self.FONT_BUTTON,
            corner_radius=8,
            height=42,
            fg_color=("firebrick", "darkred"),
            hover_color=("#8b1a1a", "#5a0000"),
            state="disabled",
        )
        self.btn_abort.pack(side="left")

        # Polling
        self.root.after(16, self._poll)

    @staticmethod
    def _card(parent, title):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=("gray90", "gray13"))
        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 15, "bold"),
            text_color=("gray25", "gray85"),
        ).pack(anchor="w", padx=14, pady=(12, 8))
        return card

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def _on_start(self):
        global cmd_queue, telemetry_queue
        if self._worker and self._worker.is_alive():
            return

        self.btn_start.configure(state="disabled")
        self.btn_abort.configure(state="normal")
        self.lbl_status.configure(text="● Starting...", text_color="cyan")

        self._worker = mp.Process(
            target=worker_entry, args=(self.config, cmd_queue, telemetry_queue)
        )
        self._worker.start()

    def _on_abort(self):
        global cmd_queue
        try:
            cmd_queue.put_nowait({"action": "ABORT"})
        except:
            pass
        self._set_idle_ui()

    def _set_idle_ui(self):
        self.btn_start.configure(state="normal")
        self.btn_abort.configure(state="disabled")
        self.lbl_status.configure(text="● Idle", text_color="gray")

    def _poll(self):
        global telemetry_queue
        if telemetry_queue is None:
            self.root.after(16, self._poll)
            return

        latest = None
        while True:
            try:
                latest = telemetry_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            self._apply_telemetry(latest)

        # Check if worker exited
        if self._worker and not self._worker.is_alive():
            if self.btn_abort.cget("state") == "normal":
                self._set_idle_ui()

        self.root.after(16, self._poll)

    def _apply_telemetry(self, frame: Dict[str, Any]):
        action = frame.get("action", "telemetry")

        if action == "worker_ready":
            self.lbl_status.configure(text="● Worker Ready", text_color="cyan")
            return

        if action == "worker_done":
            self._set_idle_ui()
            self.lbl_status.configure(text="● Worker Finished", text_color="gray")
            return

        if action == "worker_error":
            self._set_idle_ui()
            self.lbl_status.configure(text="● Worker Error", text_color="red")
            return

        # Regular telemetry
        trial_idx = frame.get("trial_idx", 0)
        total_trials = frame.get("total_trials", 0)
        phase = frame.get("phase", "unknown")
        side = frame.get("side", "—")
        theta = frame.get("theta", 0.0)
        session_num = frame.get("session_num", 0)

        self.lbl_subject_val.configure(text=self.config.get("Subject ID", "—"))
        self.lbl_session_val.configure(text=str(session_num))
        trial_str = f"{trial_idx}/{total_trials}" if total_trials else str(trial_idx)
        self.lbl_trial_val.configure(text=trial_str)
        self.lbl_phase_val.configure(text=phase)
        self.lbl_side_val.configure(text=side)
        self.lbl_theta_val.configure(text=f"{theta:.2f}")

        # Hardware
        dx = frame.get("dx", "—")
        dy = frame.get("dy", "—")
        dz = frame.get("dz", "—")
        valve = frame.get("valve", "—")
        self.lbl_hardware.configure(
            text=f"dx: {dx}  dy: {dy}  dz: {dz}  Valve: {valve}"
        )

        # Digital twin
        self._draw_twin(frame)

        # Status color based on phase
        if phase in ("Adaptation", "ITI wait [SPACE]"):
            self.lbl_status.configure(text=f"● {phase}", text_color="orange")
        elif phase == "Looming":
            self.lbl_status.configure(text="● Looming", text_color="lime")
        else:
            self.lbl_status.configure(text=f"● {phase}", text_color="cyan")

    def _draw_twin(self, frame: Dict[str, Any]):
        """Draw the looming stimulus on the canvas."""
        self.canvas.delete("all")

        side = frame.get("side", "left")
        theta = frame.get("theta", 0.0)

        # Convert theta to pixel radius on surround display
        VIEWING_DISTANCE_CM = 30.0
        SINGLE_SCREEN_WIDTH_CM = 53.0
        SINGLE_SCREEN_WIDTH_PX = 1920
        radius_cm = math.tan(math.radians(theta / 2.0)) * VIEWING_DISTANCE_CM
        radius_px = radius_cm * (SINGLE_SCREEN_WIDTH_PX / SINGLE_SCREEN_WIDTH_CM)
        radius_canvas = radius_px * self._canvas_scale_x

        # Determine screen centre on canvas (left half ~100, right half ~300)
        if side == "right":
            centre_x = 300
        else:
            centre_x = 100
        centre_y = 75

        # Clamp radius
        max_r = min(centre_x, 400 - centre_x, centre_y, 150 - centre_y) - 2
        if radius_canvas > max_r:
            radius_canvas = max_r
        if radius_canvas < 2:
            radius_canvas = 2

        x0 = centre_x - radius_canvas
        y0 = centre_y - radius_canvas
        x1 = centre_x + radius_canvas
        y1 = centre_y + radius_canvas

        self.canvas.create_oval(x0, y0, x1, y1, outline="white", width=1)

        # Draw screen boundary line
        self.canvas.create_line(200, 0, 200, 150, fill="#333333", dash=(4, 2))
        self.canvas.create_text(
            200,
            140,
            text="|",
            fill="#555555",
            font=("Consolas", 8),
            anchor="s",
        )

    def run(self):
        self.root.lift()
        self.root.focus_force()
        self.root.mainloop()


# ==============================================================================
# Config GUI & Entry
# ==============================================================================
def launch_config():
    """3-column configuration GUI matching 1.py style."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.title("BioMoR — Configuration")
    root.resizable(True, True)
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass

    FONT_LABEL = ("Segoe UI", 13)
    FONT_HEADING = ("Segoe UI", 15, "bold")
    FONT_BUTTON = ("Segoe UI", 14, "bold")

    result = {}

    # Header
    header = ctk.CTkFrame(root, fg_color="transparent")
    header.pack(fill="x", padx=24, pady=(18, 4))
    ctk.CTkLabel(
        header,
        text="BioMoR Multi-Screen Controller",
        font=("Segoe UI", 18, "bold"),
        text_color=("#1f6aa5", "#3b8ed0"),
    ).pack()

    # Card grid (three columns)
    grid = ctk.CTkFrame(root, fg_color="transparent")
    grid.pack(fill="both", expand=True, padx=24, pady=(8, 4))

    def _card(parent, title):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=("gray90", "gray13"))
        ctk.CTkLabel(
            card,
            text=title,
            font=FONT_HEADING,
            text_color=("gray25", "gray85"),
        ).pack(anchor="w", padx=14, pady=(12, 8))
        inner_grid = ctk.CTkFrame(card, fg_color="transparent")
        inner_grid.pack(fill="both", expand=True, padx=0, pady=(0, 8))
        card._inner_grid = inner_grid
        return card

    def _row(card, label_text, row_idx):
        ctk.CTkLabel(
            card,
            text=label_text,
            font=FONT_LABEL,
            text_color=("gray35", "gray70"),
        ).grid(row=row_idx, column=0, sticky="w", padx=14, pady=(6, 2))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=row_idx, column=1, sticky="ew", padx=(0, 14), pady=(6, 2))
        card.grid_columnconfigure(1, weight=1)
        return inner

    # LEFT — Experiment Control
    left_card = _card(grid, "Experiment Control")
    left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    r0 = _row(left_card._inner_grid, "✱ Subject ID:", 0)
    subject_var = ctk.StringVar(value="cricket_001")
    ctk.CTkEntry(r0, textvariable=subject_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r1 = _row(left_card._inner_grid, "✱ Start Session #:", 1)
    session_var = ctk.IntVar(value=1)
    ctk.CTkEntry(r1, textvariable=session_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r2 = _row(left_card._inner_grid, "Total Sessions:", 2)
    total_sessions_var = ctk.IntVar(value=2)
    ctk.CTkEntry(r2, textvariable=total_sessions_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r3 = _row(left_card._inner_grid, "Experiment Pattern:", 3)
    pattern_var = ctk.StringVar(value=PATTERN_CHOICES[0])
    ctk.CTkOptionMenu(
        r3,
        values=PATTERN_CHOICES,
        variable=pattern_var,
        width=170,
        corner_radius=6,
        font=FONT_LABEL,
    ).pack(fill="x")

    # MIDDLE — Timing Parameters
    mid_card = _card(grid, "Timing Parameters")
    mid_card.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

    r4 = _row(mid_card._inner_grid, "ITI Range (sec):", 0)
    iti_var = ctk.StringVar(value="60-90")
    ctk.CTkEntry(r4, textvariable=iti_var, width=170, corner_radius=6).pack(fill="x")

    r5 = _row(mid_card._inner_grid, "ISI Range (sec):", 1)
    isi_var = ctk.StringVar(value="300-600")
    ctk.CTkEntry(r5, textvariable=isi_var, width=170, corner_radius=6).pack(fill="x")

    # RIGHT — System Config
    right_card = _card(grid, "System Config")
    right_card.grid(row=0, column=2, sticky="nsew")

    # Serial Port
    ports = ["mock"] + (
        [p.device for p in serial.tools.list_ports.comports()] if HAS_SERIAL else []
    )
    port_var = ctk.StringVar(value=ports[0] if ports else "mock")
    r6 = _row(right_card._inner_grid, "Serial Port:", 0)
    ctk.CTkOptionMenu(
        r6,
        values=ports,
        variable=port_var,
        width=170,
        corner_radius=6,
        font=FONT_LABEL,
    ).pack(fill="x")

    r7 = _row(right_card._inner_grid, "Control Screen ID:", 0)
    screen_control_var = ctk.IntVar(value=0)
    ctk.CTkEntry(r7, textvariable=screen_control_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r8 = _row(right_card._inner_grid, "Stimulus Screen ID:", 1)
    screen_stimulus_var = ctk.IntVar(value=1)
    ctk.CTkEntry(r8, textvariable=screen_stimulus_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    switch_frame = ctk.CTkFrame(right_card._inner_grid, fg_color="transparent")
    switch_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 4))

    debug_var = ctk.BooleanVar(value=False)
    ctk.CTkSwitch(
        switch_frame,
        text="Debug Mode",
        variable=debug_var,
        font=FONT_LABEL,
        progress_color=("#1f6aa5", "#3b8ed0"),
    ).pack(anchor="w", pady=(0, 8))

    save_log_var = ctk.BooleanVar(value=False)
    ctk.CTkSwitch(
        switch_frame,
        text="Save Terminal Log",
        variable=save_log_var,
        font=FONT_LABEL,
        progress_color=("#1f6aa5", "#3b8ed0"),
    ).pack(anchor="w")

    for i in range(3):
        grid.grid_columnconfigure(i, weight=1, uniform="col")

    # Action bar
    action_frame = ctk.CTkFrame(root, fg_color="transparent")
    action_frame.pack(fill="x", padx=24, pady=(8, 20))

    def on_ok():
        result.update(
            {
                "Subject ID": subject_var.get().strip(),
                "Session Number": session_var.get(),
                "Total Sessions": total_sessions_var.get(),
                "Experiment Pattern": pattern_var.get(),
                "ITI Range (sec)": iti_var.get(),
                "ISI Range (sec)": isi_var.get(),
                "Serial Port": port_var.get(),
                "Control Screen ID": screen_control_var.get(),
                "Stimulus Screen ID": screen_stimulus_var.get(),
                "Debug Mode": debug_var.get(),
                "Save Terminal Log": save_log_var.get(),
            }
        )
        root.destroy()

    def on_cancel():
        result.clear()
        root.destroy()

    ctk.CTkButton(
        action_frame,
        text="Configure & Launch Dashboard",
        command=on_ok,
        font=FONT_BUTTON,
        corner_radius=8,
        height=42,
        fg_color=("#1f6aa5", "#3b8ed0"),
        hover_color=("#154a78", "#2b6fa0"),
    ).pack(side="left", padx=(0, 12))

    ctk.CTkButton(
        action_frame,
        text="Cancel",
        command=on_cancel,
        font=FONT_BUTTON,
        corner_radius=8,
        height=42,
        fg_color=("gray65", "gray30"),
        hover_color=("gray55", "gray25"),
    ).pack(side="left")

    # Center window
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"+{x}+{y}")
    root.lift()
    root.focus_force()
    root.mainloop()

    return result if result else None


def worker_entry(config, cq, tq):
    LoomingWorker(config, cq, tq).run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    config = launch_config()
    if not config:
        sys.exit(0)
    config["_output_dir"] = os.path.dirname(os.path.abspath(__file__))
    create_ipc_queues()
    MasterDashboard(config).run()

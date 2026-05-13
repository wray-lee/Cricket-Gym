"""
stimulus_controller.py  (Master-Worker Multiprocess Refactor)

Architecture
============
  Master Process (CustomTkinter, 60 Hz standalone display)
    - Persistent experiment dashboard with digital twin monitor.
    - Sends commands via cmd_queue to the Worker.
    - Receives telemetry frames via telemetry_queue at 60 Hz.

  Worker Process (PsychoPy, 3840×1080 Surround logical display, dual 100 Hz)
    - Single fullscreen window spanning two physical monitors via NVIDIA Surround.
    - Pure looming stimulus rendering with pixel-unit coordinates.
    - Physical cut masks prevent light pollution across the seam.
    - Streams per-frame telemetry back to the Master.

  Physical Constraints
    - Viewing distance: 30 cm
    - Single physical screen width: 53 cm (1920 px)
    - Surround logical screen: 3840 × 1080 px
    - Control Screen physical centre:  (-960, 0) px
    - Stimulus Screen physical centre: (960, 0) px
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
# Phase 1 — Inter-Process Communication Bridges
# ==============================================================================

# These are module-level so both Master and Worker can import them.
# The Master creates them before forking; the Worker inherits the handles
# via fork() / spawn inheritance (passed as args for spawn-compatibility).

cmd_queue: Optional[mp.Queue] = None  # Master → Worker
telemetry_queue: Optional[mp.Queue] = None  # Worker → Master


def create_ipc_queues() -> Tuple[mp.Queue, mp.Queue]:
    """Create the two inter-process queues.  Called once by the Master."""
    global cmd_queue, telemetry_queue
    cmd_queue = mp.Queue(maxsize=32)
    telemetry_queue = mp.Queue(maxsize=256)
    return cmd_queue, telemetry_queue


# ==============================================================================
# Trial Matrix Generator (unchanged logic, standalone)
# ==============================================================================


def generate_trial_matrix(
    lv_ratio_ms: float,
    initial_angle_deg: float,
    final_angle_deg: float,
    num_trials: int = 18,
    stimulus_mode: str = "Random L/R",
) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    if stimulus_mode == "Always Left":
        directions = ["left"] * num_trials
    elif stimulus_mode == "Always Right":
        directions = ["right"] * num_trials
    else:
        half = num_trials // 2
        extra_left = num_trials - half
        directions = ["left"] * extra_left + ["right"] * half
        random.shuffle(directions)

    for direction in directions:
        trials.append(
            {
                "type": "looming_visual",
                "direction": direction,
                "lv_ratio_ms": lv_ratio_ms,
                "initial_angle_deg": initial_angle_deg,
                "final_angle_deg": final_angle_deg,
            }
        )
    print(
        f"[TrialMatrix] {len(trials)} trials | "
        f"Mode={stimulus_mode} | "
        f"L={directions.count('left')} R={directions.count('right')}"
    )
    return trials


# ==============================================================================
# Serial Daemon (thread-safe, runs in Worker process)
# ==============================================================================


class SerialDaemon:
    """Daemon thread: 200 Hz readline → core.getTime() → Queue."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data_queue: queue.Queue = queue.Queue(maxsize=8192)
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._serial = serial.Serial(
            port=self.port, baudrate=self.baudrate, timeout=self.timeout
        )
        self._serial.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print(f"[SerialDaemon] Started on {self.port} @ {self.baudrate}")

    def _reader_loop(self):
        while self._running:
            try:
                raw = self._serial.readline()
                if not raw:
                    continue
                t_psy = core.getTime()
                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 5:
                    continue
                t_ard = int(parts[0])
                dx = int(parts[1])
                dy = int(parts[2])
                dz = int(parts[3])
                stim = int(parts[4])
                try:
                    self.data_queue.put_nowait((t_psy, t_ard, dx, dy, dz, stim))
                except queue.Full:
                    pass
            except (serial.SerialException, OSError):
                if self._running:
                    print("[SerialDaemon] Port error — stopping reader.")
                break
            except Exception:
                continue

    def drain_queue(self) -> List[tuple]:
        items = []
        while True:
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[SerialDaemon] Stopped.")


class MockSerialDaemon:
    """Drop-in mock when no Arduino is connected."""

    def __init__(self):
        self.data_queue: queue.Queue = queue.Queue()
        print("[MockSerialDaemon] No hardware — mock mode.")

    def start(self):
        pass

    def drain_queue(self) -> List[tuple]:
        return []

    def stop(self):
        pass


# ==============================================================================
# Ground Truth Logger (runs in Worker process)
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
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._global_trial_id: int = self._load_trial_cache()
        self._session_num: int = 0
        self._trial_in_session: int = 0
        self._csv_file = None
        self._csv_writer = None
        print(
            f"[GroundTruthLogger] global_trial_id = {self._global_trial_id} "
            f"(from .trial_cache.txt)"
        )

    def _cache_path(self) -> str:
        return os.path.join(self._output_dir, ".trial_cache.txt")

    def _load_trial_cache(self) -> int:
        p = self._cache_path()
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
                pass
        return 0

    def _save_trial_cache(self):
        with open(self._cache_path(), "w") as f:
            f.write(str(self._global_trial_id))

    def open_session(self, filepath: str, session_num: int):
        self.close()
        self._session_num = session_num
        self._trial_in_session = 0
        self._csv_file = open(filepath, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.COLUMNS)
        self._csv_file.flush()
        print(f"[GroundTruthLogger] Opened → {filepath}")

    def is_open(self) -> bool:
        return self._csv_writer is not None

    def close(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def advance_trial(self):
        self._trial_in_session += 1
        self._global_trial_id += 1
        self._save_trial_cache()

    @property
    def global_trial_id(self) -> int:
        return self._global_trial_id

    def log_motion_batch(self, tuples: List[tuple]):
        if not self._csv_writer or not tuples:
            return
        for t_psy, t_ard, dx, dy, dz, stim_state in tuples:
            self._csv_writer.writerow(
                [
                    "motion",
                    f"{t_psy:.6f}",
                    self._session_num,
                    self._trial_in_session,
                    self._global_trial_id,
                    t_ard,
                    dx,
                    dy,
                    dz,
                    stim_state,
                    "",
                ]
            )

    def log_event(self, event_name: str, timestamp: float, **kwargs):
        if not self._csv_writer:
            return
        details = json.dumps(kwargs, ensure_ascii=False) if kwargs else ""
        self._csv_writer.writerow(
            [
                event_name,
                f"{timestamp:.6f}",
                self._session_num,
                self._trial_in_session,
                self._global_trial_id,
                "",
                "",
                "",
                "",
                "",
                details,
            ]
        )

    def flush_file(self):
        if self._csv_file:
            self._csv_file.flush()


# ==============================================================================
# Console Logger (stdout tee, runs in Worker process)
# ==============================================================================


class ConsoleLogger:
    def __init__(self, log_path: str):
        self._original_stdout = sys.stdout
        self._log_file = open(log_path, "a", encoding="utf-8")
        print(f"[ConsoleLogger] Mirroring → {log_path}")

    def write(self, message: str) -> None:
        self._original_stdout.write(message)
        self._log_file.write(message)

    def flush(self) -> None:
        self._original_stdout.flush()
        self._log_file.flush()

    def close(self) -> None:
        sys.stdout = self._original_stdout
        self._log_file.close()


# ==============================================================================
# Phase 2 — Master Process: CustomTkinter Persistent Dashboard (60 Hz)
# ==============================================================================


class MasterDashboard:
    """
    Persistent experiment dashboard running on the 60 Hz standalone display.
    Replaces the old one-shot launch_experiment_gui().
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir: str = config.get("_output_dir", ".")
        self._worker_process: Optional[mp.Process] = None
        self._running = False

        # ── Theme ──
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("Looming — Experiment Dashboard")
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

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(18, 4))
        ctk.CTkLabel(
            header,
            text="Looming — Experiment Dashboard",
            font=("Segoe UI", 18, "bold"),
            text_color=("#1f6aa5", "#3b8ed0"),
        ).pack()

        # ── Main grid: left = controls, right = monitor panel ──
        main_grid = ctk.CTkFrame(self.root, fg_color="transparent")
        main_grid.pack(fill="both", expand=True, padx=24, pady=(8, 4))
        main_grid.grid_columnconfigure(0, weight=1)
        main_grid.grid_columnconfigure(1, weight=1)

        # ── LEFT: Experiment Status Card ──
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
            ("Direction:", "direction_val"),
            ("Theta (°):", "theta_val"),
        ]
        for i, (label, _) in enumerate(rows):
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
            setattr(self, f"lbl_{rows[i][1]}", val_label)

        # Arduino kinematics sub-panel
        ctk.CTkLabel(
            status_grid,
            text="Arduino:",
            font=self.FONT_LABEL,
            text_color=("gray35", "gray70"),
        ).grid(row=len(rows), column=0, sticky="w", pady=(8, 2))
        self.lbl_arduino = ctk.CTkLabel(
            status_grid,
            text="dx: —  dy: —  dz: —",
            font=self.FONT_LABEL,
            text_color="cyan",
        )
        self.lbl_arduino.grid(
            row=len(rows), column=1, sticky="w", padx=(12, 0), pady=(8, 2)
        )

        # ── RIGHT: Digital Twin Monitor Card ──
        right_card = self._card(main_grid, "Digital Twin — Surround Display")
        right_card.grid(row=0, column=1, sticky="nsew")

        # Canvas: 400×150 px, black background, represents 3840×1080 surround
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

        # ── Action Bar ──
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

        # ── Polling registration ──
        self.root.after(16, self._update_dashboard)

    @staticmethod
    def _card(parent, title: str) -> ctk.CTkFrame:
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

    # ── Button Handlers ──────────────────────────────────────────────

    def _on_start(self):
        """Launch the Worker process.  The GUI stays alive."""
        global cmd_queue, telemetry_queue
        if self._worker_process and self._worker_process.is_alive():
            print("[Dashboard] Worker already running.")
            return

        self.btn_start.configure(state="disabled")
        self.btn_abort.configure(state="normal")
        self.lbl_status.configure(text="● Running", text_color="lime")

        self._worker_process = mp.Process(
            target=worker_entry,
            args=(self.config, cmd_queue, telemetry_queue),
            name="LoomingWorker",
        )
        self._worker_process.start()
        print(f"[Dashboard] Worker PID={self._worker_process.pid} launched.")

    def _on_abort(self):
        """Send ABORT command to the Worker."""
        global cmd_queue
        try:
            cmd_queue.put_nowait({"action": "ABORT"})
            print("[Dashboard] ABORT command sent.")
        except queue.Full:
            print("[Dashboard] cmd_queue full — ABORT may be delayed.")
        self._set_idle_ui()

    def _set_idle_ui(self):
        self.btn_start.configure(state="normal")
        self.btn_abort.configure(state="disabled")
        self.lbl_status.configure(text="● Idle", text_color="gray")

    # ── 60 Hz Telemetry Polling ──────────────────────────────────────

    def _update_dashboard(self):
        """
        Non-blocking drain of telemetry_queue.
        Grabs the latest frame and updates the digital twin.
        Called every ~16 ms by root.after().
        """
        global telemetry_queue
        if telemetry_queue is None:
            self.root.after(16, self._update_dashboard)
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
        if self._worker_process is not None and not self._worker_process.is_alive():
            if self.btn_abort.cget("state") == "normal":
                self._set_idle_ui()

        self.root.after(16, self._update_dashboard)

    def _apply_telemetry(self, frame: Dict[str, Any]):
        """Parse one telemetry frame and update UI widgets."""
        action = frame.get("action", "telemetry")

        if action == "worker_ready":
            self.lbl_status.configure(text="● Worker Ready", text_color="lime")
            return

        if action == "worker_done":
            self._set_idle_ui()
            self.lbl_status.configure(text="● Worker Finished", text_color="gray")
            return

        if action == "worker_error":
            self._set_idle_ui()
            self.lbl_status.configure(text="● Worker Error", text_color="red")
            return

        # Regular telemetry frame
        trial_idx = frame.get("trial_idx", 0)
        total_trials = frame.get("total_trials", 0)
        phase = frame.get("phase", "unknown")
        side = frame.get("side", "—")
        theta = frame.get("theta", 0.0)
        session_num = frame.get("session_num", 0)

        self.lbl_subject_val.configure(text=str(self.config.get("Subject ID", "—")))
        self.lbl_session_val.configure(text=str(session_num))
        self.lbl_trial_val.configure(text=f"{trial_idx}/{total_trials}")
        self.lbl_phase_val.configure(text=phase)
        self.lbl_direction_val.configure(text=side)
        self.lbl_theta_val.configure(text=f"{theta:.2f}")

        # Arduino kinematics
        dx = frame.get("dx", "—")
        dy = frame.get("dy", "—")
        dz = frame.get("dz", "—")
        self.lbl_arduino.configure(text=f"dx: {dx}  dy: {dy}  dz: {dz}")

        # Digital twin canvas
        self._draw_twin(frame)

        # 更新焦点提示状态
        if phase in ["manual_wait", "ITI wait [SPACE]"]:
            self.lbl_status.configure(
                text='● Press and focus on Stimulus Monitors and Press "Space" to start',
                text_color="orange",
            )
        elif phase == "looming":
            self.lbl_status.configure(text="● Looming", text_color="lime")
        else:
            self.lbl_status.configure(text=f"● {phase}", text_color="cyan")

    def _draw_twin(self, frame: Dict[str, Any]):
        """Draw the microcosm mapping on the Canvas."""
        self.canvas.delete("all")

        side = frame.get("side", "left")
        theta = frame.get("theta", 0.0)

        # Convert theta to pixel radius on surround display for drawing
        radius_cm = math.tan(math.radians(theta / 2.0)) * 30.0
        radius_px = radius_cm * (1920.0 / 53.0)
        radius_canvas = radius_px * self._canvas_scale_x

        # Determine screen centre on canvas
        if side == "right":
            centre_x = 300  # right half centre [960 px → 300 canvas]
        else:
            centre_x = 100  # left half centre  [-960 px → 100 canvas]
        centre_y = 75

        # Clamp radius so it doesn't overflow the canvas
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

    # ── Mainloop ─────────────────────────────────────────────────────

    def run(self):
        self.root.lift()
        self.root.focus_force()
        self.root.mainloop()


# ==============================================================================
# Phase 2 — Configuration GUI (entry point before Dashboard)
# ==============================================================================


def _list_serial_ports() -> List[str]:
    ports = []
    if HAS_SERIAL:
        ports = [p.device for p in serial.tools.list_ports.comports()]
    defaults = ["mock"]
    if sys.platform == "win32":
        defaults.append("COM3")
    else:
        defaults.append("/dev/ttyACM0")
    seen = set()
    result = []
    for p in defaults + ports:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def launch_config_gui() -> Optional[Dict[str, Any]]:
    """
    Multi-column CustomTkinter configuration GUI.
    Returns a dict of experiment parameters, or None if cancelled.
    """
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title("Looming — Configuration")
    root.resizable(True, True)
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass

    FONT_LABEL = ("Segoe UI", 13)
    FONT_HEADING = ("Segoe UI", 15, "bold")
    FONT_BUTTON = ("Segoe UI", 14, "bold")

    result: Dict[str, Any] = {}

    # Header
    header = ctk.CTkFrame(root, fg_color="transparent")
    header.pack(fill="x", padx=24, pady=(18, 4))
    ctk.CTkLabel(
        header,
        text="Looming Visual Stimulus",
        font=("Segoe UI", 18, "bold"),
        text_color=("#1f6aa5", "#3b8ed0"),
    ).pack()

    # Card grid (three columns)
    grid = ctk.CTkFrame(root, fg_color="transparent")
    grid.pack(fill="both", expand=True, padx=24, pady=(8, 4))

    def _card(parent, title: str) -> ctk.CTkFrame:
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

    def _row(card, label_text: str, row_idx: int):
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

    # LEFT — Experiment & Trial Control
    left_card = _card(grid, "Experiment & Trial Control")
    left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    r0 = _row(left_card._inner_grid, "✱ Subject ID:", 0)
    subject_id_var = ctk.StringVar(value="cricket_001")
    ctk.CTkEntry(r0, textvariable=subject_id_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r1 = _row(left_card._inner_grid, "✱ Start Session #:", 1)
    session_num_var = ctk.IntVar(value=1)
    ctk.CTkEntry(r1, textvariable=session_num_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r2 = _row(left_card._inner_grid, "Number of Trials:", 2)
    num_trials_var = ctk.IntVar(value=18)
    ctk.CTkEntry(r2, textvariable=num_trials_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r3 = _row(left_card._inner_grid, "Stimulus Mode:", 3)
    stimulus_mode_var = ctk.StringVar(value="Random L/R")
    ctk.CTkOptionMenu(
        r3,
        values=["Always Left", "Always Right", "Random L/R"],
        variable=stimulus_mode_var,
        width=170,
        corner_radius=6,
        font=FONT_LABEL,
    ).pack(fill="x")

    # MIDDLE — Looming Physics
    mid_card = _card(grid, "Looming Physics")
    mid_card.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

    r4 = _row(mid_card._inner_grid, "l/v Ratio (ms):", 0)
    lv_ratio_var = ctk.DoubleVar(value=80.0)
    ctk.CTkEntry(r4, textvariable=lv_ratio_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r5 = _row(mid_card._inner_grid, "Initial Degree (°):", 1)
    init_deg_var = ctk.DoubleVar(value=2.0)
    ctk.CTkEntry(r5, textvariable=init_deg_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r6 = _row(mid_card._inner_grid, "Final Degree (°):", 2)
    final_deg_var = ctk.DoubleVar(value=180.0)
    ctk.CTkEntry(r6, textvariable=final_deg_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r7 = _row(mid_card._inner_grid, "ITI Range (sec):", 3)
    iti_range_var = ctk.StringVar(value="60-90")
    ctk.CTkEntry(r7, textvariable=iti_range_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    # RIGHT — System Config
    right_card = _card(grid, "System Config")
    right_card.grid(row=0, column=2, sticky="nsew")

    r8 = _row(right_card._inner_grid, "Serial Port:", 0)
    serial_ports = _list_serial_ports()
    serial_port_var = ctk.StringVar(value=serial_ports[0] if serial_ports else "mock")
    ctk.CTkOptionMenu(
        r8,
        values=serial_ports,
        variable=serial_port_var,
        width=170,
        corner_radius=6,
        font=FONT_LABEL,
    ).pack(fill="x")

    r9 = _row(right_card._inner_grid, "Control Screen ID:", 0)
    screen_control_var = ctk.IntVar(value=0)
    ctk.CTkEntry(r9, textvariable=screen_control_var, width=170, corner_radius=6).pack(
        fill="x"
    )

    r10 = _row(right_card._inner_grid, "Stimulus Screen ID:", 1)
    screen_stimulus_var = ctk.IntVar(value=1)
    ctk.CTkEntry(
        r10, textvariable=screen_stimulus_var, width=170, corner_radius=6
    ).pack(fill="x")

    switch_frame = ctk.CTkFrame(right_card._inner_grid, fg_color="transparent")
    switch_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 4))

    debug_mode_var = ctk.BooleanVar(value=False)
    ctk.CTkSwitch(
        switch_frame,
        text="Debug Mode",
        variable=debug_mode_var,
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

    def _on_ok():
        result["Subject ID"] = subject_id_var.get().strip()
        result["Session Number"] = int(session_num_var.get())
        result["Number of Trials"] = int(num_trials_var.get())
        result["Stimulus Mode"] = stimulus_mode_var.get()
        result["l/v Ratio (ms)"] = float(lv_ratio_var.get())
        result["Initial Degree (°)"] = float(init_deg_var.get())
        result["Final Degree (°)"] = float(final_deg_var.get())
        result["ITI Range (sec)"] = iti_range_var.get().strip()
        result["Serial Port"] = serial_port_var.get()
        result["Control Screen ID"] = int(screen_control_var.get())
        result["Stimulus Screen ID"] = int(screen_stimulus_var.get())
        result["Debug Mode"] = bool(debug_mode_var.get())
        result["Save Terminal Log"] = bool(save_log_var.get())
        root.destroy()

    def _on_cancel():
        root.destroy()

    ctk.CTkButton(
        action_frame,
        text="Configure & Launch Dashboard",
        command=_on_ok,
        font=FONT_BUTTON,
        corner_radius=8,
        height=42,
        fg_color=("#1f6aa5", "#3b8ed0"),
        hover_color=("#154a78", "#2b6fa0"),
    ).pack(side="left", padx=(0, 12))

    ctk.CTkButton(
        action_frame,
        text="Cancel",
        command=_on_cancel,
        font=FONT_BUTTON,
        corner_radius=8,
        height=42,
        fg_color=("gray65", "gray30"),
        hover_color=("gray55", "gray25"),
    ).pack(side="left")

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


# ==============================================================================
# Phase 3 — Worker Process: PsychoPy Pure Rendering Engine
# ==============================================================================


class LoomingWorker:
    """
    PsychoPy rendering engine running in a child process.
    Communicates exclusively via cmd_queue (in) and telemetry_queue (out).
    Zero knowledge of the 60 Hz control display.
    """

    # Physical constants
    VIEWING_DISTANCE_CM = 30.0
    SINGLE_SCREEN_WIDTH_CM = 53.0
    SINGLE_SCREEN_WIDTH_PX = 1920
    SCREEN_HEIGHT_PX = 1080
    SURROUND_WIDTH_PX = 3840

    # Pixel centres of left / right physical halves
    LEFT_CENTRE_PX = (-960, 0)
    RIGHT_CENTRE_PX = (960, 0)

    def __init__(
        self,
        config: Dict[str, Any],
        cmd_q: mp.Queue,
        telemetry_q: mp.Queue,
    ):
        self.config = config
        self.cmd_queue = cmd_q
        self.telemetry_queue = telemetry_q

        # ── Physics parameters ──
        self.lv_ratio_ms: float = float(config.get("l/v Ratio (ms)", 80))
        self.initial_angle_deg: float = float(config.get("Initial Degree (°)", 2.0))
        self.final_angle_deg: float = float(config.get("Final Degree (°)", 180.0))

        iti_range = config["ITI Range (sec)"].split("-")
        self.iti_min = float(iti_range[0])
        self.iti_max = float(iti_range[1])

        self.start_session_num = int(config["Session Number"])
        self.num_trials: int = int(config.get("Number of Trials", 18))
        self.stimulus_mode: str = str(config.get("Stimulus Mode", "Random L/R"))
        self.debug_mode: bool = bool(config.get("Debug Mode", False))
        self.save_terminal_log: bool = bool(config.get("Save Terminal Log", False))
        self.output_dir: str = config.get("_output_dir", ".")

        # ── Debug/Release 帧大小与锚点路由（根据模式选择器）──
        if self.debug_mode:
            self.win_size = (1200, 600)
            self.is_fullscr = False
            self.center_l = -300
            self.center_r = 300
            self.mask_size = (600, 600)
            self.debug_scale = 0.3  # 缩小视觉以适应小窗口
        else:
            self.win_size = (3840, 1080)
            self.is_fullscr = True
            self.center_l = -960
            self.center_r = 960
            self.mask_size = (1920, 1080)
            self.debug_scale = 1.0

        # ── Pre-compute initial/final radii in pixels ──
        self.init_r_pix = self._deg_to_pix(self.initial_angle_deg)
        self.final_r_pix = self._deg_to_pix(self.final_angle_deg)

        # ── Trial matrix ──
        self.trials = generate_trial_matrix(
            self.lv_ratio_ms,
            self.initial_angle_deg,
            self.final_angle_deg,
            self.num_trials,
            self.stimulus_mode,
        )

        # Pre-compute sync patch size in pixels
        self.sync_size_px = self._deg_to_pix(2.0)

        # ── Runtime handles ──
        self.clock = core.Clock()
        self.kb: Optional[keyboard.Keyboard] = None
        self.win_stim: Optional[visual.Window] = None
        self.stim_left: Optional[visual.Circle] = None
        self.stim_right: Optional[visual.Circle] = None
        self.mask_left: Optional[visual.Rect] = None
        self.mask_right: Optional[visual.Rect] = None
        self.serial_daemon = None
        self.logger: Optional[GroundTruthLogger] = None
        self._console_logger: Optional[ConsoleLogger] = None
        self.emergency_abort = False

        print(
            f"[LoomingWorker] l/v={self.lv_ratio_ms}ms, "
            f"θ₀={self.initial_angle_deg}°, θ_max={self.final_angle_deg}°"
        )
        t_col = (self.lv_ratio_ms / 1000.0) / math.tan(
            math.radians(self.initial_angle_deg) / 2
        )
        print(f"[LoomingWorker] t_collision = {t_col * 1000:.1f} ms")
        print(
            f"[LoomingWorker] init_r_pix={self.init_r_pix:.1f}, final_r_pix={self.final_r_pix:.1f}"
        )

    # ── Pixel coordinate converter ───────────────────────────────────

    def _deg_to_pix(self, theta_deg: float) -> float:
        """Convert visual angle (diameter, degrees) to pixel radius."""
        radius_cm = math.tan(math.radians(theta_deg / 2.0)) * self.VIEWING_DISTANCE_CM
        return (
            radius_cm
            * (self.SINGLE_SCREEN_WIDTH_PX / self.SINGLE_SCREEN_WIDTH_CM)
            * self.debug_scale
        )

    # ── Telemetry push helper ────────────────────────────────────────

    def _push_telemetry(self, frame: Dict[str, Any]):
        """Non-blocking push to telemetry queue. Silently drops on full."""
        try:
            self.telemetry_queue.put_nowait(frame)
        except queue.Full:
            pass

    # ── Drain serial + log ───────────────────────────────────────────

    def _drain_and_log(self):
        data = self.serial_daemon.drain_queue()
        if data and self.logger and self.logger.is_open():
            self.logger.log_motion_batch(data)
        return data

    # ── Latest kinematics snapshot for telemetry ─────────────────────

    def _latest_kinematics(self, data_batch: List[tuple]) -> Dict[str, int]:
        if data_batch:
            _, _, dx, dy, dz, _ = data_batch[-1]
            return {"dx": dx, "dy": dy, "dz": dz}
        return {"dx": "—", "dy": "—", "dz": "—"}

    # ── Main entry point ─────────────────────────────────────────────

    def run(self):
        """Top-level experiment loop in the Worker process."""
        self._push_telemetry({"action": "worker_ready"})

        # ── Abort callback (globalKeys) ──
        def _flag_abort():
            self.emergency_abort = True

        event.globalKeys.clear()
        event.globalKeys.add(key="escape", func=_flag_abort)

        # ── Console logger ──
        if self.save_terminal_log:
            log_path = os.path.join(
                self.output_dir, f"{self.config['Subject ID']}_terminal.txt"
            )
            self._console_logger = ConsoleLogger(log_path)
            sys.stdout = self._console_logger

        # ── GroundTruthLogger ──
        self.logger = GroundTruthLogger(self.output_dir)

        # ── Serial daemon ──
        serial_port = self.config.get("Serial Port", "")
        use_mock = serial_port.lower() in ("mock", "", "none") or not HAS_SERIAL
        if use_mock:
            self.serial_daemon = MockSerialDaemon()
        else:
            self.serial_daemon = SerialDaemon(port=serial_port)
        self.serial_daemon.start()

        # ── Single Surround Window ──
        self._init_surround_window()

        # ── Stimuli (pixel units) ──
        self._init_stimuli()

        self.kb = keyboard.Keyboard()

        # ── Render baseline ──
        self._render_baseline()
        self._drain_and_log()

        try:
            print("=== Experiment Started ===")
            current_session = self.start_session_num

            while True:
                # Check for abort command from Master
                if self._check_cmd_abort():
                    break

                self.trials = generate_trial_matrix(
                    self.lv_ratio_ms,
                    self.initial_angle_deg,
                    self.final_angle_deg,
                    self.num_trials,
                    self.stimulus_mode,
                )

                csv_filename = (
                    f"{self.config['Subject ID']}_session_{current_session}.csv"
                )
                csv_path = os.path.join(self.output_dir, csv_filename)
                self.logger.open_session(csv_path, current_session)

                print(f"\n{'='*60}")
                print(
                    f"  Session {current_session} — "
                    f"{len(self.trials)} trials | Mode={self.stimulus_mode}"
                )
                print(f"{'='*60}")

                for trial_idx, trial in enumerate(self.trials):
                    if self._check_cmd_abort():
                        break

                    if trial_idx > 0:
                        iti_dur = random.uniform(self.iti_min, self.iti_max)
                        print(f"\n--- ITI {trial_idx+1}: {iti_dur:.1f}s ---")
                        self._wait_iti(iti_dur, trial_idx, current_session)
                    else:
                        print("\n--- First Trial: Manual Trigger Wait ---")
                    if self._check_cmd_abort():
                        break

                    print(
                        f"Trial {trial_idx+1}/{len(self.trials)} | "
                        f"Dir={trial['direction']}"
                    )
                    self._run_single_trial(trial_idx, trial, current_session)
                    self.logger.flush_file()

                self.logger.flush_file()
                self.logger.close()
                print(f"[Session {current_session}] Saved → {csv_filename}")

                current_session += 1
                print(
                    f"\n--- Session {current_session - 1} complete. "
                    f"Press [SPACE] to begin Session {current_session}, "
                    f"[ESCAPE] to exit ---"
                )
            self._wait_manual_trigger(current_session)
            self._push_telemetry({"action": "worker_done"})

        except Exception as e:
            print(f"\n[Worker] Fatal Error: {e}")
            traceback.print_exc()
            self._push_telemetry({"action": "worker_error"})

        finally:
            self._shutdown()

    # ── Command check ────────────────────────────────────────────────

    def _check_cmd_abort(self) -> bool:
        """Non-blocking check for ABORT command from Master."""
        try:
            cmd = self.cmd_queue.get_nowait()
            if cmd.get("action") == "ABORT":
                print("\n[Worker] ABORT command received from Master.")
                return True
        except queue.Empty:
            pass
        return False

    # ── Window init ──────────────────────────────────────────────────

    def _init_surround_window(self):
        """Create the single 3840×1080 fullscreen surround window."""
        self.win_stim = visual.Window(
            size=self.win_size,
            fullscr=self.is_fullscr,
            screen=(
                int(self.config.get("Control Screen ID", 0))
                if not self.debug_mode
                else 0
            ),
            color=[0, 0, 0],
            colorSpace="rgb",
            units="pix",
            allowGUI=self.debug_mode,
            waitBlanking=not self.debug_mode,
            checkTiming=False,
            numSamples=0,
        )
        self.win_stim.recordFrameIntervals = not self.debug_mode

    # ── Stimulus init ────────────────────────────────────────────────

    def _init_stimuli(self):
        """Create stimuli and masks in absolute pixel coordinates."""
        # Left-side looming stimulus
        self.stim_left = visual.Circle(
            self.win_stim,
            fillColor=[-1, -1, -1],
            lineColor=[-1, -1, -1],
            radius=self.init_r_pix,
            pos=(self.center_l, 0),
            units="pix",
        )

        # Right-side looming stimulus
        self.stim_right = visual.Circle(
            self.win_stim,
            fillColor=[-1, -1, -1],
            lineColor=[-1, -1, -1],
            radius=self.init_r_pix,
            pos=(self.center_r, 0),
            units="pix",
        )

        # Physical cut masks — black rectangles that absorb overflow light
        # Left mask covers the right half
        self.mask_left = visual.Rect(
            self.win_stim,
            width=self.mask_size[0],
            height=self.mask_size[1],
            pos=(self.center_l, 0),
            fillColor=[0, 0, 0],
            lineColor=[0, 0, 0],
            units="pix",
        )

        # Right mask covers the left half
        self.mask_right = visual.Rect(
            self.win_stim,
            width=self.mask_size[0],
            height=self.mask_size[1],
            pos=(self.center_r, 0),
            fillColor=[0, 0, 0],
            lineColor=[0, 0, 0],
            units="pix",
        )

        # Photodiode Sync Patches (Bottom outer corners)
        sync_y = -self.mask_size[1] / 2 + self.sync_size_px / 2
        sync_offset_x = self.mask_size[0] / 2 - self.sync_size_px / 2

        self.sync_left = visual.Rect(
            self.win_stim,
            width=self.sync_size_px,
            height=self.sync_size_px,
            pos=(self.center_l - sync_offset_x, sync_y),
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
            units="pix",
        )

        self.sync_right = visual.Rect(
            self.win_stim,
            width=self.sync_size_px,
            height=self.sync_size_px,
            pos=(self.center_r + sync_offset_x, sync_y),
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
            units="pix",
        )

    # ── Baseline render ──────────────────────────────────────────────

    def _render_baseline(self):
        """Both sides show the baseline dot. No masks needed."""
        self.stim_left.radius = self.init_r_pix
        self.stim_right.radius = self.init_r_pix
        self.stim_left.draw()
        self.stim_right.draw()
        self.win_stim.flip(clearBuffer=True)

    # ── Manual trigger wait ──────────────────────────────────────────

    def _wait_manual_trigger(self, session_num: int):
        self.logger.log_event("manual_wait_start", self.clock.getTime())
        self._push_telemetry(
            {
                "action": "telemetry",
                "trial_idx": 0,
                "total_trials": self.num_trials,
                "phase": "manual_wait",
                "side": "—",
                "theta": self.initial_angle_deg,
                "session_num": session_num,
                "dx": "—",
                "dy": "—",
                "dz": "—",
            }
        )

        self.kb.clearEvents()
        while True:
            if self._check_cmd_abort():
                core.quit()
            if self.emergency_abort:
                core.quit()

            data_batch = self._drain_and_log()
            latest_kin = self._latest_kinematics(data_batch)

            self._render_baseline()
            self._push_telemetry(
                {
                    "action": "telemetry",
                    "trial_idx": 0,
                    "total_trials": self.num_trials,
                    "phase": "manual_wait",
                    "side": "—",
                    "theta": self.initial_angle_deg,
                    "session_num": current_session,
                    **latest_kin,
                }
            )

            keys = self.kb.getKeys(["space", "escape"], waitRelease=False)
            if any(k.name == "escape" for k in keys):
                print("\n[Aborted] Emergency Exit.")
                core.quit()
            if any(k.name == "space" for k in keys):
                trigger_time = self.clock.getTime()
                self.logger.log_event("manual_trigger", trigger_time)
                print(f"  [ManualTrigger] SPACE at t={trigger_time:.4f}")
                break

    # ── ITI wait ─────────────────────────────────────────────────────

    def _wait_iti(self, duration: float, trial_idx: int, session_num: int):
        self.logger.log_event("iti_start", self.clock.getTime(), duration=duration)
        t0 = self.clock.getTime()
        countdown_done = False

        while True:
            if self._check_cmd_abort():
                core.quit()
            if self.emergency_abort:
                core.quit()

            now = self.clock.getTime()
            elapsed = now - t0
            data_batch = self._drain_and_log()
            latest_kin = self._latest_kinematics(data_batch)

            if elapsed < duration:
                # Countdown phase
                remaining = duration - elapsed
                phase = f"ITI ({elapsed:.1f}s/{duration:.1f}s)"
            else:
                # Manual trigger wait
                if not countdown_done:
                    self.logger.log_event(
                        "iti_countdown_complete", now, duration=duration
                    )
                    countdown_done = True
                    self.kb.clearEvents()
                phase = "ITI wait [SPACE]"

                keys = self.kb.getKeys(["space", "escape"], waitRelease=False)
                if any(k.name == "escape" for k in keys):
                    print("\n[Aborted] Emergency Exit (ITI).")
                    core.quit()
                if any(k.name == "space" for k in keys):
                    trigger_time = self.clock.getTime()
                    self.logger.log_event(
                        "manual_trigger",
                        trigger_time,
                        iti_elapsed=elapsed,
                        iti_duration=duration,
                    )
                    print(f"  [ManualTrigger] SPACE at t={trigger_time:.4f}")
                    break

            self._render_baseline()
            self._push_telemetry(
                {
                    "action": "telemetry",
                    "trial_idx": trial_idx + 1,
                    "total_trials": self.num_trials,
                    "phase": phase,
                    "side": "—",
                    "theta": self.initial_angle_deg,
                    "session_num": session_num,
                    **latest_kin,
                }
            )

    # ── Single trial ─────────────────────────────────────────────────

    def _run_single_trial(
        self, trial_idx: int, trial: Dict[str, Any], session_num: int
    ):
        t_start = self.clock.getTime()
        self.logger.advance_trial()

        direction = trial["direction"]
        lv_ratio_ms = trial["lv_ratio_ms"]
        initial_deg = trial["initial_angle_deg"]
        final_deg = trial["final_angle_deg"]

        self.logger.log_event(
            "trial_start",
            t_start,
            trial_index=trial_idx,
            direction=direction,
            lv_ratio_ms=lv_ratio_ms,
            initial_angle_deg=initial_deg,
            final_angle_deg=final_deg,
        )

        # Physics
        lv_sec = lv_ratio_ms / 1000.0
        t_collision = lv_sec / math.tan(math.radians(initial_deg) / 2)
        print(
            f"  [Phys] θ₀={initial_deg}° θ_max={final_deg}° "
            f"l/v={lv_ratio_ms}ms → t_col={t_collision*1000:.1f}ms"
        )

        t0_loom = self.clock.getTime()
        first_logged = False
        coll_logged = False

        # Pre-select the active stim centre
        if direction == "right":
            active_centre = (self.center_r, 0)
        else:
            active_centre = (self.center_l, 0)

        while True:
            if self._check_cmd_abort():
                core.quit()
            if self.emergency_abort:
                core.quit()

            now = self.clock.getTime()
            elapsed = now - t0_loom

            # Stop 1.0 s after collision
            if elapsed >= t_collision + 1.0:
                self.logger.log_event(
                    "looming_completed",
                    now,
                    elapsed=elapsed,
                    t_collision=t_collision,
                )
                break

            # Angular computation (clamped to final_deg)
            if elapsed < t_collision:
                ttc_rem = t_collision - elapsed
                theta = math.degrees(2 * math.atan(lv_sec / ttc_rem))
                if theta > final_deg:
                    theta = final_deg
            else:
                theta = final_deg
                if not coll_logged:
                    self.logger.log_event(
                        "collision_reached",
                        now,
                        elapsed=elapsed,
                        t_collision=t_collision,
                        theta=theta,
                    )
                    coll_logged = True

            current_r_pix = self._deg_to_pix(theta)

            # ── Phase 4 Rendering Pipeline (per spec) ──
            if direction == "left":
                # Active: left  |  Block: right mask + right baseline
                self.stim_left.radius = current_r_pix
                self.stim_left.draw()
                self.mask_right.draw()
                self.stim_right.radius = self.init_r_pix
                self.stim_right.draw()
                self.sync_left.draw()
            else:
                # Active: right  |  Block: left mask + left baseline
                self.stim_right.radius = current_r_pix
                self.stim_right.draw()
                self.mask_left.draw()
                self.stim_left.radius = self.init_r_pix
                self.stim_left.draw()
                self.sync_right.draw()

            flip_t = self.win_stim.flip(clearBuffer=True)
            if flip_t is None:
                flip_t = self.clock.getTime()

            if not first_logged:
                self.logger.log_event(
                    "first_frame",
                    flip_t,
                    initial_angle=theta,
                    active_screen=direction,
                )
                first_logged = True

            # Drain, log, push telemetry
            data_batch = self._drain_and_log()
            latest_kin = self._latest_kinematics(data_batch)
            self._push_telemetry(
                {
                    "action": "telemetry",
                    "trial_idx": trial_idx + 1,
                    "total_trials": self.num_trials,
                    "phase": "looming",
                    "side": direction,
                    "theta": theta,
                    "session_num": session_num,
                    **latest_kin,
                }
            )

        # Post-trial baseline
        self._render_baseline()
        self._drain_and_log()

    # ── Shutdown ─────────────────────────────────────────────────────

    def _shutdown(self):
        if self.logger:
            self.logger.close()
        if self.serial_daemon:
            self.serial_daemon.stop()
        if self.win_stim:
            self.win_stim.close()
        if self._console_logger:
            self._console_logger.close()


# ==============================================================================
# Worker process entry point (top-level for mp.Process)
# ==============================================================================


def worker_entry(config: Dict[str, Any], cmd_q: mp.Queue, telemetry_q: mp.Queue):
    """Entry point for the child process spawned by mp.Process."""
    global visual, core, event, monitors, keyboard
    from psychopy import visual, core, event, monitors
    from psychopy.hardware import keyboard

    worker = LoomingWorker(config, cmd_q, telemetry_q)
    worker.run()


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    # macOS: use 'spawn' to avoid issues with OpenGL/Cocoa in forked processes
    if sys.platform == "darwin":
        mp.set_start_method("spawn", force=True)
    else:
        # Windows/Linux: 'spawn' is safest for GUI + OpenGL
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

    # Step 1 — Config GUI
    config = launch_config_gui()
    if not config:
        print("Cancelled.")
        sys.exit(0)

    print("\n--- Configuration ---")
    for k, v in config.items():
        print(f"  {k}: {v}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config["_output_dir"] = script_dir

    # Step 2 — Create IPC queues
    cmd_q, telemetry_q = create_ipc_queues()

    # Step 3 — Launch persistent dashboard (Master)
    dashboard = MasterDashboard(config)
    dashboard.run()

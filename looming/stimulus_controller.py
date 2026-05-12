"""
stimulus_controller.py

BioMoR — Cricket VR Closed-Loop Stimulus Controller (PsychoPy)

Architecture (Photodiode T₀)
=============================
  SerialDaemon (daemon thread)
    200 Hz readline → core.getTime() stamp
    → Queue⟨T_psy, T_ard, dx, dy, dz, stim_state⟩

  Main Thread (PsychoPy)
    ITI phase:     compute delay_ms, send <DIR,DELAY_MS> to pre-arm Arduino.
    Stimulus Phase: draw sync_patch continuously to provide a steady high-TTL gate
                    for the photodiode.  Disappears naturally during ITI/Baseline.
    Every frame:   drain queue → write ALL tuples to CSV → render → flip

  Trigger flow:
    1. Python sends <L,5729> during ITI  → Arduino arms valve + delay.
    2. Looming Frame 0 flashes sync_patch → photodiode fires ISR → T₀ recorded.
    3. Arduino internally fires valve at T₀ + delay_ms.

Data Integrity
--------------
  - 200 Hz kinematics are NEVER discarded.  Every tuple from the queue
    is written to the session CSV with session_num, trial_in_session,
    and global_trial_id.
  - global_trial_id persists to .trial_cache.txt — survives crashes.

Physical Constraints
--------------------
  - 2° steady-state baseline in ALL idle states.
  - t_collision = (lv_ratio_ms/1000) / tan(radians(2°)/2)
    For l/v = 100 ms → ≈5729 ms.
"""

import csv
import json
import math
import os
import sys
import random
import threading
import queue
from typing import Any, Dict, List, Optional

from psychopy import visual, core, event, gui, monitors
from psychopy.hardware import keyboard

try:
    import serial
    import serial.tools.list_ports

    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ==============================================================================
# Pattern Definitions (9 paradigms)
# ==============================================================================

EXPERIMENT_PATTERNS = {
    "Baseline Visual (仅视觉，无风)": {
        "type": "baseline_visual",
        "target_ttc_ms": None,
        "lv_ratio_ms": 100,
        "description": "纯视觉 Looming，不触发任何风刺激",
    },
    "Baseline Wind (仅风，随机延迟)": {
        "type": "baseline_wind",
        "target_ttc_ms": None,
        "lv_ratio_ms": None,
        "description": "纯风刺激，无视觉 Looming，屏幕保持 2° 黑点基线",
    },
    "Looming + Wind (TTC -373ms / 30°)": {
        "type": "looming_wind",
        "target_ttc_ms": -373,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞前 373ms 触发 (θ≈30°)",
    },
    "Looming + Wind (TTC -308ms / 36°)": {
        "type": "looming_wind",
        "target_ttc_ms": -308,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞前 308ms 触发 (θ≈36°)",
    },
    "Looming + Wind (TTC -261ms / 42°)": {
        "type": "looming_wind",
        "target_ttc_ms": -261,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞前 261ms 触发 (θ≈42°)",
    },
    "Looming + Wind (TTC -225ms / 48°)": {
        "type": "looming_wind",
        "target_ttc_ms": -225,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞前 225ms 触发 (θ≈48°)",
    },
    "Looming + Wind (TTC -119ms / 80°)": {
        "type": "looming_wind",
        "target_ttc_ms": -119,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞前 119ms 触发 (θ≈80°)",
    },
    "Looming + Wind (TTC 0ms / 180°)": {
        "type": "looming_wind",
        "target_ttc_ms": 0,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞瞬间触发 (θ=180°)",
    },
    "Looming + Wind (TTC +200ms)": {
        "type": "looming_wind",
        "target_ttc_ms": 200,
        "lv_ratio_ms": 100,
        "description": "Looming + 风，风在碰撞后 200ms 触发",
    },
}

PATTERN_CHOICES = list(EXPERIMENT_PATTERNS.keys())


# ==============================================================================
# Trial Matrix Generator
# ==============================================================================


def generate_trial_matrix(pattern_key: str) -> List[Dict[str, Any]]:
    """18 trials (9L + 9R), single paradigm, shuffled."""
    if pattern_key not in EXPERIMENT_PATTERNS:
        raise ValueError(f"Unknown pattern: '{pattern_key}'")

    p = EXPERIMENT_PATTERNS[pattern_key]
    trials: List[Dict[str, Any]] = []

    for direction in ["left"] * 9 + ["right"] * 9:
        d: Dict[str, Any] = {
            "type": p["type"],
            "target_ttc_ms": p["target_ttc_ms"],
            "lv_ratio_ms": p["lv_ratio_ms"],
        }
        if p["type"] == "baseline_visual":
            d["wind_dir"] = "none"
            d["screen_side"] = direction
        else:
            d["wind_dir"] = direction
        trials.append(d)

    assert len(trials) == 18
    random.shuffle(trials)
    return trials


# ==============================================================================
# Serial Daemon Thread
# ==============================================================================


class SerialDaemon:
    """
    Daemon thread: 200 Hz readline → core.getTime() → Queue.
    Pre-arm:  write(b'<L,5729>') — packet command for T₀ architecture.
    """

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

    def send_arm_command(self, direction: str, delay_ms: int):
        """
        Pre-arm Arduino for T₀-anchored valve firing.
        Sends packet: <L,5729> or <R,300>
        """
        dir_char = "R" if direction == "right" else "L"
        cmd = f"<{dir_char},{delay_ms}>"
        self.write(cmd.encode("ascii"))
        print(f"[SerialDaemon] Armed: {cmd}")

    def write(self, data: bytes):
        if self._serial and self._serial.is_open:
            self._serial.write(data)

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

    def send_arm_command(self, direction: str, delay_ms: int):
        dir_char = "R" if direction == "right" else "L"
        print(f"[MockArm] <{dir_char},{delay_ms}> at t={core.getTime():.4f}")

    def write(self, data: bytes):
        pass

    def drain_queue(self) -> List[tuple]:
        return []

    def stop(self):
        pass


# ==============================================================================
# Ground Truth Logger — Streaming CSV + .trial_cache.txt persistence
# ==============================================================================


class GroundTruthLogger:
    """
    Streaming CSV writer.  Every 200 Hz motion tuple and every event is
    written to disk immediately.  No in-memory accumulation.

    CSV columns (union schema):
      event_name | timestamp | session_num | trial_in_session |
      global_trial_id | t_ard | dx | dy | dz | stim_state | details

    Motion rows:   event_name='motion',  t_ard/dx/dy/dz/stim_state filled.
    Event rows:    event_name=<name>,     details=JSON of kwargs.

    global_trial_id is persisted to .trial_cache.txt in output_dir.
    """

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

    # ---- Trial cache persistence ----

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

    # ---- Session file management ----

    def open_session(self, filepath: str, session_num: int):
        """Open a new CSV file for this session."""
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

    # ---- Trial counter ----

    def advance_trial(self):
        """Call at trial START. Increments both counters, persists to disk."""
        self._trial_in_session += 1
        self._global_trial_id += 1
        self._save_trial_cache()

    @property
    def global_trial_id(self) -> int:
        return self._global_trial_id

    # ---- High-frequency motion logging ----

    def log_motion_batch(self, tuples: List[tuple]):
        """
        Write a batch of (t_psy, t_ard, dx, dy, dz, stim_state) tuples
        directly to the open CSV.  Called every frame.
        NO flush() here — OS buffered I/O only.  Explicit flush is
        deferred to flush_file() at trial boundaries.
        """
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

    # ---- Sparse event logging ----

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

    # ---- Safe low-frequency flush (call at trial boundaries only) ----

    def flush_file(self):
        """Force buffered CSV data to disk.  Call only during non-timing-critical
        moments (e.g. post-trial, before ITI, session end) to avoid VSync jitter."""
        if self._csv_file:
            self._csv_file.flush()


# ==============================================================================
# Console Logger (stdout tee)
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
# Looming Engine
# ==============================================================================


class LoomingEngine:
    """
    VSync-locked visual stimulus engine.

    Key invariant:  _drain_and_log() is called EVERY frame in EVERY loop
    (ITI, ISI, looming, baseline_wind, wait-for-start).  This ensures
    zero motion data loss from the 200 Hz Arduino stream.

    Photodiode sync: self.sync_patch — white square at screen corner,
    drawn CONTINUOUSLY throughout the entire looming stimulus phase to provide
    a steady high-TTL gate for the photodiode.  Disappears during ITI/Baseline.
    """

    def __init__(self, exp_info: Dict[str, Any]):
        self.exp_info = exp_info

        iti_range = self.exp_info["ITI Range (sec)"].split("-")
        self.iti_min = float(iti_range[0])
        self.iti_max = float(iti_range[1])

        isi_range = self.exp_info["ISI Range (sec)"].split("-")
        self.isi_min = float(isi_range[0])
        self.isi_max = float(isi_range[1])

        self.start_session_num = int(self.exp_info["Session Number"])
        self.total_sessions = int(self.exp_info.get("Total Sessions", 1))

        self.debug_mode: bool = bool(
            self.exp_info.get("Debug Mode (Single Screen)", True)
        )
        self.screen_left: int = int(self.exp_info.get("Left Screen ID", 1))
        self.screen_right: int = int(self.exp_info.get("Right Screen ID", 2))
        self.save_terminal_log: bool = bool(
            self.exp_info.get("Save Terminal Log (Debug)", False)
        )

        self.pattern_key: str = self.exp_info.get(
            "Experiment Pattern", PATTERN_CHOICES[0]
        )
        print(f"[LoomingEngine] Pattern: {self.pattern_key}")
        print(f"[LoomingEngine] Baseline: gray bg + 2° dark dot (Anti-Startle)")

        self.trials = generate_trial_matrix(self.pattern_key)
        self.clock = core.Clock()
        self.kb: Optional[keyboard.Keyboard] = None

        # Physical angle constraints
        self.initial_angle_deg = 2.0
        self.max_angle_deg = 180.0

        # Populated in run_experiment()
        self.serial_daemon = None
        self.logger: Optional[GroundTruthLogger] = None
        self._console_logger: Optional[ConsoleLogger] = None

        # Window & stimulus handles
        self.win_control: Optional[visual.Window] = None
        self.stim_ctrl_left: Optional[visual.Circle] = None
        self.stim_ctrl_right: Optional[visual.Circle] = None
        self.label_left: Optional[visual.TextStim] = None
        self.label_right: Optional[visual.TextStim] = None
        self.win_left: Optional[visual.Window] = None
        self.win_right: Optional[visual.Window] = None
        self.stim_left: Optional[visual.Circle] = None
        self.stim_right: Optional[visual.Circle] = None

        # Photodiode sync patches — created per-window in run_experiment()
        self.sync_patch_ctrl_left: Optional[visual.Rect] = None
        self.sync_patch_ctrl_right: Optional[visual.Rect] = None
        self.sync_patch_left: Optional[visual.Rect] = None
        self.sync_patch_right: Optional[visual.Rect] = None

    # ------------------------------------------------------------------
    # Drain + log helper — called EVERY frame
    # ------------------------------------------------------------------
    def _drain_and_log(self):
        """
        Non-blocking drain of serial queue.  Writes ALL motion tuples
        to the open CSV immediately.  If no session file is open (e.g.
        during adaptation), data is drained but discarded to prevent
        queue overflow.
        """
        data = self.serial_daemon.drain_queue()
        if data and self.logger and self.logger.is_open():
            self.logger.log_motion_batch(data)

    # ------------------------------------------------------------------
    # Sync patch factory
    # ------------------------------------------------------------------
    @staticmethod
    def _make_sync_patch(win: visual.Window, side: str = "left") -> visual.Rect:
        """
        Create a high-contrast white square for photodiode detection.
        X-axis mirrored for V-array spatial symmetry:
          left  screen → bottom-LEFT  corner  (x = -offset)
          right screen → bottom-RIGHT corner  (x = +offset)
        Size: 2° × 2° (small but detectable).
        """
        x_offset = 25
        x_pos = -x_offset if side == "left" else x_offset
        return visual.Rect(
            win,
            width=2.0,
            height=2.0,
            fillColor=[1, 1, 1],  # pure white
            lineColor=[1, 1, 1],
            pos=(x_pos, -15),  # mirrored corner (deg units)
            units="deg",
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_experiment(self, output_dir: str = "."):
        # ---- 全局安全逃生舱（状态机模式）----
        # globalKeys 回调只设标志位，不在 Pyglet C 回调栈里抛 SystemExit，
        # 避免 ctypes 异常吞噬。主线程轮询读取标志后在 Python 层执行 core.quit()。
        self.emergency_abort = False

        def _flag_abort():
            self.emergency_abort = True

        event.globalKeys.clear()
        event.globalKeys.add(key="escape", func=_flag_abort)

        # ---- Console logger (debug) ----
        if self.debug_mode and self.save_terminal_log:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(
                output_dir, f"{self.exp_info['Subject ID']}_terminal.txt"
            )
            self._console_logger = ConsoleLogger(log_path)
            sys.stdout = self._console_logger

        # ---- GroundTruthLogger ----
        self.logger = GroundTruthLogger(output_dir)

        # ---- Serial daemon ----
        serial_port = self.exp_info.get("Serial Port", "")
        use_mock = serial_port.lower() in ("mock", "", "none") or not HAS_SERIAL
        if use_mock:
            self.serial_daemon = MockSerialDaemon()
        else:
            self.serial_daemon = SerialDaemon(port=serial_port)
        self.serial_daemon.start()

        # ---- Monitor profile ----
        my_monitor = monitors.Monitor("cricket_monitor")
        my_monitor.setWidth(53.0)
        my_monitor.setDistance(30.0)
        my_monitor.setSizePix((1920, 1080))
        my_monitor.saveMon()

        # ---- 硬件拓扑冲突校验 (防呆机制) ----
        if not self.debug_mode:
            # 在 Prod 模式下，控制屏(默认为0)、左屏、右屏必须是三个独立的物理监视器
            assigned_screens = {0, self.screen_left, self.screen_right}
            if len(assigned_screens) < 3:
                print("\n" + "!" * 60)
                print("[硬件配置致命错误] 屏幕 ID 发生冲突！")
                print(
                    f"当前配置：控制台=0, 左屏={self.screen_left}, 右屏={self.screen_right}"
                )
                print("在 Prod 模式下，必须分配三个完全不同的物理显示器 ID。")
                print("多个全屏窗口抢占同一显示器会导致显卡底层死锁（白屏卡死）。")
                print(
                    "--> 如果您在单屏幕上进行测试，请务必在启动界面勾选 'Debug Mode'！"
                )
                print("")
                print("[Hardware Config Error] Screen ID conflict detected!")
                print(
                    f"Config now：Console=0, 左屏={self.screen_left}, 右屏={self.screen_right}"
                )
                print(
                    "You must assign three distinct physical monitor IDs in Production mode."
                )
                print(
                    "Multiple fullscreen windows on the same display will cause GPU-level deadlock (white screen freeze)."
                )
                print(
                    "--> Please check the 'Debug Mode' checkbox on the startup interface!"
                )
                print("!" * 60 + "\n")
                self._shutdown()
                core.quit()

        # ---- Windows ----
        if self.debug_mode:
            print("[LoomingEngine] Debug Mode — control panel on screen 0")
            self.win_control = visual.Window(
                size=(1000, 500),
                monitor=my_monitor,
                screen=0,
                color=[0, 0, 0],
                colorSpace="rgb",
                units="deg",
                fullscr=False,
                allowGUI=True,
                waitBlanking=False,  # 三屏统一关闭 VSync，杜绝多屏死锁
                checkTiming=False,
            )
            self.win_control.recordFrameIntervals = False
        else:
            print(
                f"[LoomingEngine] Production — L={self.screen_left} R={self.screen_right}"
            )
            self.win_control = visual.Window(
                size=(1000, 500),
                monitor=my_monitor,
                screen=0,
                color=[0, 0, 0],
                colorSpace="rgb",
                units="deg",
                fullscr=False,
                allowGUI=True,
                waitBlanking=False,
                checkTiming=False,
            )
            self.win_control.recordFrameIntervals = False
            self.win_left = visual.Window(
                size=(1920, 1080),
                monitor=my_monitor,
                screen=self.screen_left,
                color=[0, 0, 0],
                colorSpace="rgb",
                units="deg",
                fullscr=True,
                allowGUI=False,
                waitBlanking=True,  # Master VSync 锁相 — 全系统唯一硬件时钟源
                checkTiming=True,
            )
            self.win_left.recordFrameIntervals = True
            self.win_right = visual.Window(
                size=(1920, 1080),
                monitor=my_monitor,
                screen=self.screen_right,
                color=[0, 0, 0],
                colorSpace="rgb",
                units="deg",
                fullscr=True,
                allowGUI=False,
                waitBlanking=False,
                checkTiming=False,  # 打破双全屏 VSync 死锁：以 win_left 为唯一主时钟
            )
            self.win_right.recordFrameIntervals = False

        # ---- 显存初始化首帧：清除窗口白屏，强制基线灰 ----
        self.win_control.flip()
        if not self.debug_mode:
            self.win_left.flip()
            self.win_right.flip()

        # ---- Control panel stimuli ----
        self.stim_ctrl_left = visual.Circle(
            self.win_control,
            fillColor=[-1, -1, -1],
            lineColor=[-1, -1, -1],
            radius=self.initial_angle_deg / 2.0,
            pos=(-15, 0),
        )
        self.stim_ctrl_right = visual.Circle(
            self.win_control,
            fillColor=[-1, -1, -1],
            lineColor=[-1, -1, -1],
            radius=self.initial_angle_deg / 2.0,
            pos=(15, 0),
        )
        self.label_left = visual.TextStim(
            self.win_control,
            text="[LEFT MONITOR]",
            pos=(-15, 10),
            color=[1, 1, 1],
            height=1.2,
        )
        self.label_right = visual.TextStim(
            self.win_control,
            text="[RIGHT MONITOR]",
            pos=(15, 10),
            color=[1, 1, 1],
            height=1.2,
        )
        self.preview_text = visual.TextStim(
            self.win_control,
            text="[Hardware Live] Waiting for data...",
            pos=(0, -12),
            color="green",
            height=1.0,
        )

        if not self.debug_mode:
            self.stim_left = visual.Circle(
                self.win_left,
                fillColor=[-1, -1, -1],
                lineColor=[-1, -1, -1],
                radius=self.initial_angle_deg / 2.0,
                pos=(0, 0),
            )
            self.stim_right = visual.Circle(
                self.win_right,
                fillColor=[-1, -1, -1],
                lineColor=[-1, -1, -1],
                radius=self.initial_angle_deg / 2.0,
                pos=(0, 0),
            )

        # ---- Photodiode sync patches ----
        # Control panel: dual mirror indicators (L/R) for V-array symmetry.
        # Positioned within the smaller 1000×500 control window viewport.
        self.sync_patch_ctrl_left = visual.Rect(
            self.win_control,
            width=2,
            height=2,
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
            pos=(-10, -10),  # upper-left area — mirrors left physical screen
            units="deg",
        )
        self.sync_patch_ctrl_right = visual.Rect(
            self.win_control,
            width=2,
            height=2,
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
            pos=(10, -10),  # upper-right area — mirrors right physical screen
            units="deg",
        )
        if not self.debug_mode:
            self.sync_patch_left = self._make_sync_patch(self.win_left, side="left")
            self.sync_patch_right = self._make_sync_patch(self.win_right, side="right")

        self.kb = keyboard.Keyboard()
        self._render_static_baseline()

        try:
            self.win_control.winHandle.activate()
        except Exception:
            pass

        print(f"[Config] Pattern: {self.pattern_key}")
        print(
            f"[Config] {self.total_sessions} session(s) from #{self.start_session_num}"
        )
        print(
            f"[Config] t_collision(l/v=100ms,θ₀=2°) = "
            f"{(0.1 / math.tan(math.radians(2.0)/2)) * 1000:.1f} ms"
        )

        # ---- Wait-for-Start + Session loop ----
        # try 上移至适应期之前：适应期的硬件/窗口异常也能滑入 finally 销毁僵尸窗口
        try:
            print(
                "\n[Ready] Animal adaptation phase. "
                "Press [SPACE] to start, [ESCAPE] to abort..."
            )
            self.kb.clearEvents()
            start_experiment = False
            while not start_experiment:
                # 获取实时数据用于 Preview
                data_batch = self.serial_daemon.drain_queue()
                if data_batch:
                    # 提取最新的一条元组数据: (t_psy, t_ard, dx, dy, dz, stim_state)
                    latest = data_batch[-1]
                    self.preview_text.text = f"[Hardware Live] dx: {latest[2]:>3} | dy: {latest[3]:>3} | dz: {latest[4]:>3} | valve: {latest[5]}"

                # 渲染 2° 物理基线，并将 Preview 文本安全地叠加到控制面板 (win_control)
                self._render_static_baseline(extra_ctrl_stims=[self.preview_text])

                keys = self.kb.getKeys(["space", "escape"], waitRelease=False)
                escape_polled = any(k.name == "escape" for k in keys)
                if self.emergency_abort or escape_polled:
                    print("\n[Aborted] Emergency Exit Triggered.")
                    core.quit()  # 主 Python 线程退出 → SystemExit → finally → _shutdown()
                if any(k.name == "space" for k in keys):
                    start_experiment = True

            print("\n=== Experiment Started ===")

            # ---- Session loop ----
            for session_idx in range(self.total_sessions):
                current_session = self.start_session_num + session_idx
                self.trials = generate_trial_matrix(self.pattern_key)

                # Open session CSV
                csv_filename = (
                    f"{self.exp_info['Subject ID']}" f"_session_{current_session}.csv"
                )
                csv_path = os.path.join(output_dir, csv_filename)
                self.logger.open_session(csv_path, current_session)

                print(f"\n{'='*60}")
                print(
                    f"  Session {current_session} "
                    f"({session_idx+1}/{self.total_sessions}) — "
                    f"{len(self.trials)} trials | {self.pattern_key}"
                )
                print(f"{'='*60}")

                for trial_idx, trial in enumerate(self.trials):
                    if trial_idx > 0:
                        iti_dur = random.uniform(self.iti_min, self.iti_max)
                        print(f"\n--- ITI {trial_idx+1}: {iti_dur:.1f}s ---")
                        self._wait_iti(iti_dur, trial)
                    else:
                        # First trial — still need to pre-arm before looming
                        self._pre_arm_arduino(trial)

                    print(
                        f"Trial {trial_idx+1}/{len(self.trials)} | "
                        f"{trial['type']} | "
                        f"TTC={trial.get('target_ttc_ms','-')}ms | "
                        f"Wind={trial.get('wind_dir','-')}"
                    )
                    self._run_single_trial(trial_idx, trial)
                    # Safe flush after each trial (non-timing-critical)
                    self.logger.flush_file()

                # Final flush + close session CSV
                self.logger.flush_file()
                self.logger.close()
                print(f"[Session {current_session}] Saved → {csv_filename}")

                # ISI between sessions
                if session_idx < self.total_sessions - 1:
                    ns = self.start_session_num + session_idx + 1
                    isi_dur = random.uniform(self.isi_min, self.isi_max)
                    print(f"\n--- ISI → Session {ns}: {isi_dur:.1f}s ---")
                    self._wait_isi(isi_dur, ns)

            print("\n=== All Sessions Completed ===")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    def _shutdown(self):
        if self.logger:
            self.logger.close()
        if self.serial_daemon:
            self.serial_daemon.stop()
        self._cleanup_windows()
        if self._console_logger:
            self._console_logger.close()

    # ------------------------------------------------------------------
    def _render_static_baseline(self, extra_ctrl_stims: list = None):
        """2° dark dot on mid-gray — universal baseline for all 9 paradigms."""
        self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0
        self.stim_ctrl_right.radius = self.initial_angle_deg / 2.0
        self.stim_ctrl_left.draw()
        self.stim_ctrl_right.draw()
        self.label_left.draw()
        self.label_right.draw()
        if extra_ctrl_stims:
            for s in extra_ctrl_stims:
                s.draw()
        self.win_control.flip()

        if not self.debug_mode:
            if self.stim_left is not None:
                self.stim_left.radius = self.initial_angle_deg / 2.0
                self.stim_left.draw()
            self.win_left.flip()
            if self.stim_right is not None:
                self.stim_right.radius = self.initial_angle_deg / 2.0
                self.stim_right.draw()
            self.win_right.flip()

    # ------------------------------------------------------------------
    def _cleanup_windows(self):
        if self.win_control:
            self.win_control.close()
        if self.win_left:
            self.win_left.close()
        if self.win_right:
            self.win_right.close()

    # ------------------------------------------------------------------
    # Pre-arm Arduino during ITI
    # ------------------------------------------------------------------
    def _compute_arm_delay_ms(self, trial: Dict[str, Any]) -> Optional[int]:
        """
        Compute the delay (ms from T₀) at which Arduino should fire the valve.
        Returns None if this trial should NOT fire any valve.

        For looming_wind: delay = t_collision_ms + target_ttc_ms
            (target_ttc_ms is negative for pre-collision, positive for post)
        For baseline_wind: handled separately (random delay, no T₀ anchor)
        For baseline_visual: no wind at all
        """
        if trial["type"] == "baseline_visual":
            return None
        if trial["type"] == "baseline_wind":
            return None  # baseline_wind uses its own random-delay logic

        # looming_wind
        _ttc = trial.get("target_ttc_ms")
        lv_ms = trial.get("lv_ratio_ms")
        if _ttc is None or lv_ms is None:
            return None

        lv_sec = lv_ms / 1000.0
        t_collision_sec = lv_sec / math.tan(math.radians(self.initial_angle_deg) / 2)
        t_collision_ms = int(round(t_collision_sec * 1000))
        delay_ms = t_collision_ms + _ttc  # ttc is negative before collision
        return max(0, delay_ms)

    def _pre_arm_arduino(self, trial: Dict[str, Any]):
        """
        Send pre-arm command to Arduino for the NEXT trial.
        Called during ITI so Arduino is ready before looming starts.
        """
        delay_ms = self._compute_arm_delay_ms(trial)
        if delay_ms is None:
            return  # No valve for this trial type

        wind_dir = trial.get("wind_dir", "left")
        self.serial_daemon.send_arm_command(wind_dir, delay_ms)
        self.logger.log_event(
            "arduino_armed",
            self.clock.getTime(),
            direction=wind_dir,
            delay_ms=delay_ms,
            trial_type=trial["type"],
            target_ttc_ms=trial.get("target_ttc_ms"),
        )

    # ------------------------------------------------------------------
    def _wait_iti(self, duration: float, next_trial: Dict[str, Any] = None):
        """
        ITI: render baseline, drain queue, and pre-arm Arduino for next trial.
        Pre-arm is sent at the START of ITI so Arduino has plenty of time.
        """
        self.logger.log_event("iti_start", self.clock.getTime(), duration=duration)

        # Pre-arm for the upcoming trial
        if next_trial is not None:
            self._pre_arm_arduino(next_trial)

        t0 = self.clock.getTime()
        while (self.clock.getTime() - t0) < duration:
            self._render_static_baseline()
            self._drain_and_log()  # ← every frame
            if self.emergency_abort or self.kb.getKeys(["escape"]):
                print("\n[Aborted] Emergency Exit Triggered (ITI).")
                core.quit()

    # ------------------------------------------------------------------
    def _wait_isi(self, duration: float, next_session_num: int):
        """ISI: no session file is open, drain-and-discard."""
        countdown = visual.TextStim(
            self.win_control,
            text="",
            pos=(0, -8),
            color="yellow",
            height=1.5,
            bold=True,
        )
        t0 = self.clock.getTime()
        while True:
            remaining = duration - (self.clock.getTime() - t0)
            if remaining <= 0:
                break
            countdown.text = (
                f"ISI: {int(remaining)}s\n" f"Next → Session {next_session_num}"
            )
            self._render_static_baseline(extra_ctrl_stims=[countdown])
            self.serial_daemon.drain_queue()  # discard — between sessions
            if self.emergency_abort or self.kb.getKeys(["escape"]):
                print("\n[Aborted] Emergency Exit Triggered (ISI).")
                core.quit()

    # ------------------------------------------------------------------
    def _resolve_active_window(self, trial):
        side = trial.get("screen_side")
        wind = trial.get("wind_dir", "none")
        if side == "right" or wind == "right":
            return (self.stim_ctrl_right, self.win_right, self.stim_right, "right")
        else:
            return (self.stim_ctrl_left, self.win_left, self.stim_left, "left")

    # ------------------------------------------------------------------
    def _resolve_sync_patch(self, side: str):
        """
        Return (physical_patch, ctrl_mirror_patch) for the active side.
        In debug mode physical_patch is None (no physical screens).
        """
        ctrl_mirror = (
            self.sync_patch_ctrl_right if side == "right" else self.sync_patch_ctrl_left
        )
        if self.debug_mode:
            return None, ctrl_mirror
        if side == "right":
            return self.sync_patch_right, ctrl_mirror
        return self.sync_patch_left, ctrl_mirror

    # ------------------------------------------------------------------
    def _run_single_trial(self, trial_idx: int, trial: Dict[str, Any]):
        t_start = self.clock.getTime()

        # Advance habituation counters + persist
        self.logger.advance_trial()

        _ttc = trial.get("target_ttc_ms")
        wind_dir = trial.get("wind_dir", "none")

        self.logger.log_event(
            "trial_start",
            t_start,
            trial_index=trial_idx,
            target_ttc_ms=_ttc,
            pattern=self.pattern_key,
            **{k: v for k, v in trial.items() if k != "target_ttc_ms"},
        )

        active_ctrl, active_win, active_stim, side = self._resolve_active_window(trial)
        print(f"  [Route] screen={side}")

        # ==============================================================
        # LOOMING (baseline_visual | looming_wind)
        # ==============================================================
        if trial["type"] in ["baseline_visual", "looming_wind"]:
            lv_sec = trial["lv_ratio_ms"] / 1000.0

            # t_collision from 2° baseline (pure physics)
            t_collision = lv_sec / math.tan(math.radians(self.initial_angle_deg) / 2)

            print(
                f"  [Phys] θ₀=2° l/v={trial['lv_ratio_ms']}ms → "
                f"t_col={t_collision*1000:.1f}ms"
            )

            # Inactive side
            if side == "left":
                _inact_ctrl = self.stim_ctrl_right
                _inact_win = self.win_right if not self.debug_mode else None
                _inact_stim = self.stim_right if not self.debug_mode else None
            else:
                _inact_ctrl = self.stim_ctrl_left
                _inact_win = self.win_left if not self.debug_mode else None
                _inact_stim = self.stim_left if not self.debug_mode else None

            # Sync patches for photodiode trigger (physical + control mirror)
            sync_patch, sync_patch_ctrl_mirror = self._resolve_sync_patch(side)

            t0_loom = self.clock.getTime()
            first_logged = False
            coll_logged = False
            frame_count = 0
            frame_ts: List[float] = [] if self.debug_mode else None

            while True:
                now = self.clock.getTime()
                elapsed = now - t0_loom

                if elapsed >= t_collision + 1.0:
                    self.logger.log_event(
                        "looming_completed",
                        now,
                        elapsed=elapsed,
                        t_collision=t_collision,
                    )
                    break

                # Angular computation
                if elapsed < t_collision:
                    ttc_rem = t_collision - elapsed
                    theta = math.degrees(2 * math.atan(lv_sec / ttc_rem))
                else:
                    theta = self.max_angle_deg
                    if not coll_logged:
                        self.logger.log_event(
                            "collision_reached",
                            now,
                            elapsed=elapsed,
                            t_collision=t_collision,
                        )
                        coll_logged = True

                # ---- Control panel ----
                active_ctrl.radius = theta / 2.0
                active_ctrl.draw()
                _inact_ctrl.radius = self.initial_angle_deg / 2.0
                _inact_ctrl.draw()
                self.label_left.draw()
                self.label_right.draw()

                # Draw side-specific sync mirror on control panel (continuously)
                if sync_patch_ctrl_mirror is not None:
                    sync_patch_ctrl_mirror.draw()

                # ---- Physical window draws (must happen BEFORE flip sequence) ----
                if not self.debug_mode and active_win is not None:
                    active_stim.radius = theta / 2.0
                    active_stim.draw()
                    # Frame 0: flash sync patch on the active physical window
                    if sync_patch is not None:
                        sync_patch.draw()
                    if _inact_stim is not None and _inact_win is not None:
                        _inact_stim.radius = self.initial_angle_deg / 2.0
                        _inact_stim.draw()

                # ---- 强制绝对硬件翻转序列 (Hardware Flip Sequence) ----
                if not self.debug_mode:
                    # 生产模式：先推所有从机，最后推主机阻塞并获取时间戳
                    self.win_control.flip()
                    self.win_right.flip()
                    flip_t = self.win_left.flip()
                else:
                    # Debug 模式：仅单屏幕工作，直接翻转并获取时间戳
                    flip_t = self.win_control.flip()
                    if flip_t is None:
                        flip_t = core.getTime()

                if self.debug_mode:
                    frame_ts.append(flip_t)
                if not first_logged:
                    self.logger.log_event(
                        "first_frame",
                        flip_t,
                        initial_angle=theta,
                        active_screen=side,
                        sync_patch_drawn=True,
                    )
                    first_logged = True

                frame_count += 1

                # ★ Drain & log every frame — no data loss
                self._drain_and_log()

                if self.emergency_abort or self.kb.getKeys(["escape"]):
                    print("\n[Aborted] Emergency Exit Triggered (Looming).")
                    core.quit()

            if self.debug_mode and frame_ts and len(frame_ts) > 1:
                import numpy as np

                ifis = np.diff(frame_ts) * 1000.0
                print(
                    f"  [IFI] frames={len(frame_ts)} "
                    f"mean={ifis.mean():.2f}ms std={ifis.std():.2f}ms "
                    f"min={ifis.min():.2f}ms max={ifis.max():.2f}ms"
                )

        # ==============================================================
        # BASELINE WIND (2° visual baseline maintained, wind only)
        # ==============================================================
        elif trial["type"] == "baseline_wind":
            delay = random.uniform(0.1, 1.2)
            post_obs = random.uniform(1.0, 2.0)
            print(f"  [Wind] delay={delay*1000:.0f}ms post={post_obs:.2f}s")
            self.logger.log_event(
                "baseline_wind_start", self.clock.getTime(), random_delay_sec=delay
            )

            t0 = self.clock.getTime()
            fire_time = None
            trigger_sent = False

            while True:
                now = self.clock.getTime()
                elapsed = now - t0

                if fire_time is not None and (now - fire_time) >= post_obs:
                    break

                if elapsed >= delay and not trigger_sent:
                    # baseline_wind: direct single-byte trigger (no T₀ architecture)
                    # Send as arm + immediate synthetic T₀
                    dir_char = "R" if wind_dir == "right" else "L"
                    self.serial_daemon.send_arm_command(wind_dir, 0)
                    self.logger.log_event(
                        "wind_triggered",
                        now,
                        random_delay_sec=delay,
                        wind_direction=wind_dir,
                    )
                    trigger_sent = True
                    fire_time = now

                # 2° baseline render (no visual change)
                self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0
                self.stim_ctrl_right.radius = self.initial_angle_deg / 2.0
                self.stim_ctrl_left.draw()
                self.stim_ctrl_right.draw()
                self.label_left.draw()
                self.label_right.draw()

                # Draw 2° baseline on physical windows before flip sequence
                if not self.debug_mode:
                    if self.stim_left is not None:
                        self.stim_left.radius = self.initial_angle_deg / 2.0
                        self.stim_left.draw()
                    if self.stim_right is not None:
                        self.stim_right.radius = self.initial_angle_deg / 2.0
                        self.stim_right.draw()

                # ---- 强制绝对硬件翻转序列 (Hardware Flip Sequence) ----
                if not self.debug_mode:
                    # 生产模式：先推所有从机，最后推主机阻塞节拍
                    self.win_control.flip()
                    self.win_right.flip()
                    self.win_left.flip()
                else:
                    # Debug 模式：仅单屏幕工作，单次翻转
                    self.win_control.flip()

                # ★ Drain & log every frame
                self._drain_and_log()

                if self.emergency_abort or self.kb.getKeys(["escape"]):
                    print("\n[Aborted] Emergency Exit Triggered (Wind).")
                    core.quit()

        # Post-trial baseline
        self._render_static_baseline()
        self._drain_and_log()


# ==============================================================================
# GUI & Entrypoint (customtkinter — dark industrial-grade UI)
# ==============================================================================

_CTK_AVAILABLE = True
try:
    import customtkinter as ctk
except ImportError:
    _CTK_AVAILABLE = False


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


def launch_experiment_gui() -> Optional[Dict[str, Any]]:
    """Launch a modern customtkinter configuration panel.
    Returns the EXACT same config dict as the original PsychoPy GUI,
    with identical keys and value types.  Returns None if cancelled.
    """
    if not _CTK_AVAILABLE:
        # Fallback to original PsychoPy GUI if customtkinter is missing
        import psychopy.gui as _psychopy_gui

        dlg = _psychopy_gui.Dlg(title="BioMoR Looming Paradigms")
        dlg.addText("=== Core Experiment Parameters ===")
        dlg.addField("Subject ID", label="✱Subject ID:", initial="cricket_001")
        dlg.addField("Session Number", label="✱Start Session #:", initial=1)
        dlg.addField("Total Sessions", label="Total Sessions:", initial=2)
        dlg.addField(
            "Experiment Pattern",
            label="✱Pattern:",
            choices=PATTERN_CHOICES,
            tip="Single-pattern locking per subject",
        )
        dlg.addField("ITI Range (sec)", label="ITI Range (sec):", initial="60-90")
        dlg.addField("ISI Range (sec)", label="ISI Range (sec):", initial="300-600")
        dlg.addText("\n\n=== Hardware ===")
        dlg.addField(
            "Serial Port",
            label="Serial Port:",
            choices=_list_serial_ports(),
            tip='Arduino port or "mock"',
        )
        dlg.addField("Left Screen ID", label="Left Screen ID:", initial="1")
        dlg.addField("Right Screen ID", label="Right Screen ID:", initial="2")
        dlg.addText("\n\n=== Debug ===")
        dlg.addField("Debug Mode (Single Screen)", label="Debug Mode:", initial=True)
        dlg.addField(
            "Save Terminal Log (Debug)", label="Save Terminal Log:", initial=False
        )
        ok_data = dlg.show()
        return ok_data if dlg.OK else None

    # ---- CTk modern UI below ----
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title("BioMoR — Looming Paradigm Controller")
    root.geometry("900x600")
    root.resizable(True, True)

    result: Dict[str, Any] = {}
    cancelled = True

    # ---- Font & style constants ----
    FONT = ("Segoe UI", 13)
    FONT_BOLD = ("Segoe UI", 13, "bold")
    FONT_HEADER = ("Segoe UI", 15, "bold")
    FONT_SMALL = ("Segoe UI", 11)

    # =================================================================
    # Outer frame (pack) — all children use grid inside their own frame
    # =================================================================
    outer = ctk.CTkFrame(root, fg_color="transparent")
    outer.pack(fill="both", expand=True, padx=16, pady=(16, 8))

    # ---- Title ----
    title_label = ctk.CTkLabel(
        outer,
        text="🧬 BioMoR — Cricket VR Closed-Loop Configuration",
        font=("Segoe UI", 18, "bold"),
    )
    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 14), sticky="w")

    # =================================================================
    # Card 1: Experiment & Subject (LEFT)
    # =================================================================
    card1 = ctk.CTkFrame(outer, corner_radius=8, border_width=1, border_color="#3a3a3a")
    card1.grid(row=1, column=0, padx=(0, 8), sticky="nsew")
    outer.columnconfigure(0, weight=1)

    ctk.CTkLabel(card1, text="🧪 Subject and experiment", font=FONT_HEADER).grid(
        row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w"
    )

    _make_label(card1, "✱ Subject number", 1)
    entry_subject = ctk.CTkEntry(card1, font=FONT, corner_radius=6, width=220)
    entry_subject.insert(0, "cricket_001")
    entry_subject.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card1, "✱ Start Session number", 3)
    entry_session = ctk.CTkEntry(card1, font=FONT, corner_radius=6, width=220)
    entry_session.insert(0, "1")
    entry_session.grid(row=4, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card1, "Total Session count", 5)
    entry_total = ctk.CTkEntry(card1, font=FONT, corner_radius=6, width=220)
    entry_total.insert(0, "2")
    entry_total.grid(row=6, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    # =================================================================
    # Card 2: Paradigm & Timing (CENTER)
    # =================================================================
    card2 = ctk.CTkFrame(outer, corner_radius=8, border_width=1, border_color="#3a3a3a")
    card2.grid(row=1, column=1, padx=8, sticky="nsew")
    outer.columnconfigure(1, weight=1)

    ctk.CTkLabel(card2, text="⏱️ Paradigm and Timing", font=FONT_HEADER).grid(
        row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w"
    )

    _make_label(card2, "✱ Experiment Paradigm", 1)
    pattern_var = ctk.StringVar(value=PATTERN_CHOICES[0])
    opt_pattern = ctk.CTkOptionMenu(
        card2,
        values=PATTERN_CHOICES,
        variable=pattern_var,
        font=FONT,
        corner_radius=6,
        dropdown_font=FONT_SMALL,
    )
    opt_pattern.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card2, "ITI Range (sec)", 3)
    entry_iti = ctk.CTkEntry(card2, font=FONT, corner_radius=6, width=220)
    entry_iti.insert(0, "60-90")
    entry_iti.grid(row=4, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card2, "ISI Range (sec)", 5)
    entry_isi = ctk.CTkEntry(card2, font=FONT, corner_radius=6, width=220)
    entry_isi.insert(0, "300-600")
    entry_isi.grid(row=6, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    # =================================================================
    # Card 3: System & Hardware (RIGHT)
    # =================================================================
    card3 = ctk.CTkFrame(outer, corner_radius=8, border_width=1, border_color="#3a3a3a")
    card3.grid(row=1, column=2, padx=(8, 0), sticky="nsew")
    outer.columnconfigure(2, weight=1)

    ctk.CTkLabel(card3, text="⚙️ System and Hardware", font=FONT_HEADER).grid(
        row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w"
    )

    _make_label(card3, "Serial Port", 1)
    serial_ports = _list_serial_ports()
    serial_var = ctk.StringVar(value=serial_ports[0] if serial_ports else "mock")
    opt_serial = ctk.CTkOptionMenu(
        card3,
        values=serial_ports,
        variable=serial_var,
        font=FONT,
        corner_radius=6,
        dropdown_font=FONT_SMALL,
    )
    opt_serial.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card3, "Left Screen ID(Logger Screen ID)", 3)
    entry_left = ctk.CTkEntry(card3, font=FONT, corner_radius=6, width=220)
    entry_left.insert(0, "1")
    entry_left.grid(row=4, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    _make_label(card3, "Right Screen ID(Reference Screen ID)", 5)
    entry_right = ctk.CTkEntry(card3, font=FONT, corner_radius=6, width=220)
    entry_right.insert(0, "2")
    entry_right.grid(row=6, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="ew")

    # ---- Debug toggles ----
    debug_frame = ctk.CTkFrame(card3, fg_color="transparent")
    debug_frame.grid(row=7, column=0, columnspan=2, padx=14, pady=(10, 4), sticky="ew")

    debug_var = ctk.BooleanVar(value=True)
    switch_debug = ctk.CTkSwitch(
        debug_frame,
        text="🖥️  Debug Mode (Single Screen)",
        variable=debug_var,
        font=FONT,
    )
    switch_debug.grid(row=0, column=0, padx=(0, 10), pady=4, sticky="w")

    log_var = ctk.BooleanVar(value=False)
    switch_log = ctk.CTkSwitch(
        debug_frame,
        text="📄  Save Terminal Log (Debug)",
        variable=log_var,
        font=FONT,
    )
    switch_log.grid(row=1, column=0, padx=(0, 10), pady=4, sticky="w")

    # =================================================================
    # Bottom action bar
    # =================================================================
    action_frame = ctk.CTkFrame(root, fg_color="transparent")
    action_frame.pack(fill="x", padx=16, pady=(0, 16))

    def _on_submit():
        nonlocal cancelled, result
        try:
            result = {
                "Subject ID": entry_subject.get().strip(),
                "Session Number": int(entry_session.get().strip()),
                "Total Sessions": int(entry_total.get().strip()),
                "Experiment Pattern": pattern_var.get(),
                "ITI Range (sec)": entry_iti.get().strip(),
                "ISI Range (sec)": entry_isi.get().strip(),
                "Serial Port": serial_var.get(),
                "Left Screen ID": int(entry_left.get().strip()),
                "Right Screen ID": int(entry_right.get().strip()),
                "Debug Mode (Single Screen)": debug_var.get(),
                "Save Terminal Log (Debug)": log_var.get(),
            }
        except (ValueError, TypeError):
            return  # stays open until inputs are valid
        cancelled = False
        root.destroy()

    btn_start = ctk.CTkButton(
        action_frame,
        text="▶  Start Experiment",
        command=_on_submit,
        font=FONT_BOLD,
        corner_radius=8,
        height=44,
        fg_color="#1f6aa5",
        hover_color="#185078",
    )
    btn_start.pack(side="left", padx=(0, 12))

    btn_cancel = ctk.CTkButton(
        action_frame,
        text="✕  Cancel",
        command=root.destroy,
        font=FONT_BOLD,
        corner_radius=8,
        height=44,
        fg_color="#555555",
        hover_color="#444444",
    )
    btn_cancel.pack(side="left")

    # =================================================================
    # Grid weight config for cards to fill evenly
    # =================================================================
    for card in (card1, card2, card3):
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

    root.mainloop()
    return result if not cancelled else None


def _make_label(parent: ctk.CTkFrame, text: str, row: int):
    """Helper: create a small label above an entry inside a card."""
    lbl = ctk.CTkLabel(parent, text=text, font=("Segoe UI", 12), anchor="w")
    lbl.grid(row=row, column=0, columnspan=2, padx=14, pady=(4, 0), sticky="w")


if __name__ == "__main__":
    config = launch_experiment_gui()
    if not config:
        print("Cancelled.")
        core.quit()

    print("\n--- Configuration ---")
    for k, v in config.items():
        print(f"  {k}: {v}")

    engine = LoomingEngine(exp_info=config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine.run_experiment(output_dir=script_dir)

"""
stimulus_controller.py

TTC-driven cross-modal paradigm.  The physical engine uses forward
elapsed time, supporting trigger firing at any signed TTC:
  negative = before collision, 0 = at collision, positive = after.

A high-precision visual stimulus engine for presenting looming stimuli 
driven by classical l/v physical kinematics. This script is designed as 
a scientific-grade software controller for the BioMoR project.

Unified Control Panel Architecture:
  - Control Monitor (Screen 0): always-on widescreen window displaying a
    real-time side-by-side mirror of both stimulus channels with
    "[LEFT MONITOR]" / "[RIGHT MONITOR]" labels.
  - Debug Mode:  only win_control exists (VSync master).
  - Production Mode: win_control (async, non-blocking) + two fullscreen
    physical stimulus windows (win_left, win_right) with VSync enabled.

Core capabilities:
1. Pre-Experiment GUI dialog for parameter configuration.
2. Built-in, structured 18-trial Paradigm matrix (TTC-driven).
3. Exact VSync-locked timing via PsychoPy (sub-millisecond precision).
4. Modular Hardware Mock Interface for external triggering (e.g., Arduino).
5. Precision logging system tracking paradigm configurations down to VSyncs.

=== 重构说明 (Refactoring Notes) ===
  - 引入"单范式锁定"机制：每个被试 (Subject) 仅运行 GUI 中选定的一种范式。
  - Introduce "Single-Pattern Locking" mechanism: each subject only runs one paradigm selected in the GUI.
  - Trial Matrix 重构为纯空间随机化：18 试次 = 9 Left + 9 Right，完全打乱。
  - Trial Matrix refactored to pure spatial randomization: 18 trials = 9 Left + 9 Right, completely shuffled.
  - 稳态视觉基线 (2° 黑点)：全部 9 种范式（含 Baseline Wind）的所有
    空闲状态均显示灰底 + 2° 黑点，作为归一化基线并消除 Startle Reflex。
  - Steady-state visual baseline (2° dark dot): all 9 paradigms (including Baseline Wind) 
    display a gray background + 2° dark dot during all idle states, serving as a normalization baseline and eliminating the Startle Reflex.
    (pattern.md: "Begin Degree: 2°（as baseline on the same time）")
"""

import math
import os
import sys
import random
from typing import Any, Dict, List, Optional
import pandas as pd
from psychopy import visual, core, event, gui, monitors
from psychopy.hardware import keyboard


# ==============================================================================
# 范式定义 (Pattern Definitions)
# ==============================================================================
# 定义 9 种实验范式的完整参数表。
# Define the complete parameter table for the 9 experimental paradigms.
# 每种范式用一个唯一的 key 标识，在 GUI 下拉菜单中显示完整名称。
# Each paradigm is identified by a unique key and its full name is displayed in the GUI dropdown menu.
# 'type' 字段决定 Trial 运行逻辑（与原有 baseline_visual / looming_wind /
# baseline_wind 三分支完全对应）。
# The 'type' field determines the logic for running the Trial (perfectly corresponds to the 
# original baseline_visual / looming_wind / baseline_wind three branches).
# ==============================================================================

EXPERIMENT_PATTERNS = {
    'Baseline Visual (仅视觉，无风)': {
        'type': 'baseline_visual',
        'target_ttc_ms': None,
        'lv_ratio_ms': 100,
        'description': '纯视觉 Looming，不触发任何风刺激',
    },
    'Baseline Wind (仅风，随机延迟)': {
        'type': 'baseline_wind',
        'target_ttc_ms': None,
        'lv_ratio_ms': None,
        'description': '纯风刺激，无视觉 Looming，屏幕保持 2° 黑点基线',
    },
    'Looming + Wind (TTC -373ms / 30°)': {
        'type': 'looming_wind',
        'target_ttc_ms': -373,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞前 373ms 触发 (θ≈30°)',
    },
    'Looming + Wind (TTC -308ms / 36°)': {
        'type': 'looming_wind',
        'target_ttc_ms': -308,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞前 308ms 触发 (θ≈36°)',
    },
    'Looming + Wind (TTC -261ms / 42°)': {
        'type': 'looming_wind',
        'target_ttc_ms': -261,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞前 261ms 触发 (θ≈42°)',
    },
    'Looming + Wind (TTC -225ms / 48°)': {
        'type': 'looming_wind',
        'target_ttc_ms': -225,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞前 225ms 触发 (θ≈48°)',
    },
    'Looming + Wind (TTC -119ms / 80°)': {
        'type': 'looming_wind',
        'target_ttc_ms': -119,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞前 119ms 触发 (θ≈80°)',
    },
    'Looming + Wind (TTC 0ms / 180°)': {
        'type': 'looming_wind',
        'target_ttc_ms': 0,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞瞬间触发 (θ=180°)',
    },
    'Looming + Wind (TTC +200ms)': {
        'type': 'looming_wind',
        'target_ttc_ms': 200,
        'lv_ratio_ms': 100,
        'description': 'Looming + 风，风在碰撞后 200ms 触发',
    },
}

# GUI 下拉列表的显示顺序（与 pattern.md 定义一致）
# Display order for the GUI dropdown list (consistent with pattern.md definitions)
PATTERN_CHOICES = list(EXPERIMENT_PATTERNS.keys())


# ==============================================================================
# Module 2: Paradigm & Trial Matrix Generator
# ==============================================================================

def generate_trial_matrix(pattern_key: str) -> List[Dict[str, Any]]:
    """
    【重构】单范式试次矩阵生成器。
    [Refactoring] Single-paradigm trial matrix generator.

    根据 GUI 中选定的唯一范式 (pattern_key)，生成恰好 18 个试次，
    并在左右方向上实现绝对空间平衡（9 Left + 9 Right），然后完全打乱。
    Generate exactly 18 trials based on the unique paradigm (pattern_key) selected in the GUI,
    achieving absolute spatial balance in the left/right directions (9 Left + 9 Right), and then completely shuffle them.

    重构要点：
    Refactoring highlights:
    ----------
    - 原逻辑：在 18 个试次中混合所有 7 种 TTC + 2 个视觉基线 + 2 个风基线。
    - Original logic: Mixed all 7 TTCs + 2 visual baselines + 2 wind baselines within 18 trials.
    - 新逻辑：一个 Session 只包含一种范式的 18 个试次，
      唯一的随机化维度是空间方向（left/right）。
    - New logic: A Session contains exactly 18 trials of a single paradigm, 
      where the only randomized dimension is the spatial direction (left/right).
    - 对于 baseline_visual 类型：wind_dir 设为 'none'，
      但通过 screen_side 字段控制刺激呈现在左屏还是右屏，
      确保 9 次左屏 + 9 次右屏的空间平衡。
    - For the 'baseline_visual' type: 'wind_dir' is set to 'none', 
      but the 'screen_side' field controls whether the stimulus appears on the left or right screen 
      to ensure a spatial balance of 9 left screen + 9 right screen presentations.
    - 对于 baseline_wind 和 looming_wind 类型：
      wind_dir 直接控制风的方向和屏幕路由。
    - For the 'baseline_wind' and 'looming_wind' types:
      'wind_dir' directly controls the wind direction and screen routing.

    Parameters
    ----------
    pattern_key : str
        EXPERIMENT_PATTERNS 中定义的范式键名，
        例如 'Looming + Wind (TTC -373ms / 30°)'。
        The paradigm key defined in EXPERIMENT_PATTERNS,
        e.g., 'Looming + Wind (TTC -373ms / 30°)'.

    Returns
    -------
    List[Dict[str, Any]]
        包含 18 个完全打乱的试次字典列表。
        A list of 18 completely shuffled trial dictionaries.
    """
    if pattern_key not in EXPERIMENT_PATTERNS:
        raise ValueError(f"未知范式: '{pattern_key}'。"
                         f"可用范式: {list(EXPERIMENT_PATTERNS.keys())}")

    pattern = EXPERIMENT_PATTERNS[pattern_key]
    trial_type = pattern['type']
    target_ttc_ms = pattern['target_ttc_ms']
    lv_ratio_ms = pattern['lv_ratio_ms']

    trials: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 生成 18 个试次：9 Left + 9 Right，确保绝对空间平衡
    # Generate 18 trials: 9 Left + 9 Right, ensuring absolute spatial balance
    # ------------------------------------------------------------------
    directions = ['left'] * 9 + ['right'] * 9

    for direction in directions:
        trial_dict: Dict[str, Any] = {
            'type': trial_type,
            'target_ttc_ms': target_ttc_ms,
            'lv_ratio_ms': lv_ratio_ms,
        }

        if trial_type == 'baseline_visual':
            # 纯视觉试次：无风，但通过 screen_side 控制呈现屏幕
            # Pure visual trial: no wind, but presenting screen is controlled via screen_side
            trial_dict['wind_dir'] = 'none'
            trial_dict['screen_side'] = direction
        else:
            # baseline_wind 或 looming_wind：wind_dir 同时控制风向和屏幕路由
            # baseline_wind or looming_wind: wind_dir simultaneously controls wind direction and screen routing
            trial_dict['wind_dir'] = direction

        trials.append(trial_dict)

    assert len(trials) == 18, f"Expected 18 trials, got {len(trials)}"

    # 完全打乱试次顺序
    # Completely shuffle the trial order
    random.shuffle(trials)
    return trials


# ==============================================================================
# Module 3: Hardware Interface Abstraction Layer (The Mock Trigger Interface)
# ==============================================================================

class HardwareTrigger:
    """
    A unified interface for interacting with external hardware devices like Arduino.
    """
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.trigger_log: List[Dict[str, Any]] = []
        if self.mode == "mock":
            print("[HardwareTrigger] Initialized in MOCK mode.")

    def send_trigger(self, event_code: str, current_time: float, 
                     wind_delay_ms: Optional[float] = None, 
                     wind_direction: Optional[str] = None) -> None:
        """
        Sends a hardware trigger containing timing and directional parameters.
        """
        self.trigger_log.append({
            'event_code': event_code, 
            'request_time': current_time,
            'wind_delay_ms': wind_delay_ms,
            'wind_direction': wind_direction
        })
        
        if self.mode == "mock":
            print(f"[HardwareTrigger] MOCK FIRED: {event_code} at t={current_time:.4f} "
                  f"| Delay: {wind_delay_ms}ms | Dir: {wind_direction}")


# ==============================================================================
# Module 4: High Precision Log System (The Ground Truth Logger)
# ==============================================================================

class GroundTruthLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log_event(self, event_name: str, timestamp: float, **kwargs) -> None:
        event_dict = {'event_name': event_name, 'timestamp': timestamp}
        event_dict.update(kwargs)
        self.events.append(event_dict)

    def save_log(self, output_path: str) -> None:
        if not self.events:
            return
        df = pd.DataFrame(self.events)
        df.to_csv(output_path, index=False)
        print(f"[GroundTruthLogger] Ground Truth data saved successfully to {output_path}")


# ==============================================================================
# Module 4b: Console Logger (Stdout Tee — Debug Only)
# ==============================================================================

class ConsoleLogger:
    """
    A stdout hijacker that duplicates all print output to both the original
    terminal and a text log file, similar to the Unix ``tee`` command.

    Only activated when Debug Mode is ON and 'Save Terminal Log' is True.
    """

    def __init__(self, log_path: str):
        self._original_stdout = sys.stdout
        self._log_file = open(log_path, 'a', encoding='utf-8')
        print(f"[ConsoleLogger] Terminal output is being mirrored to: {log_path}")

    def write(self, message: str) -> None:
        self._original_stdout.write(message)
        self._log_file.write(message)

    def flush(self) -> None:
        self._original_stdout.flush()
        self._log_file.flush()

    def close(self) -> None:
        """Restore original stdout and close the log file."""
        sys.stdout = self._original_stdout
        self._log_file.close()


# ==============================================================================
# Module 5: Visual Stimulus Core & Integration (The Looming Engine)
# ==============================================================================

class LoomingEngine:
    """
    Manages the overall experiment execution, GUI parameter integration,
    ITI randomization, and the rigid VSync physics rendering.

    Unified Control Panel Architecture:
      win_control (Screen 0) — always present in both Debug and Production
          modes.  Renders a side-by-side mirror of the left and right
          stimulus channels with "[LEFT MONITOR]" / "[RIGHT MONITOR]"
          text labels, giving the experimenter a real-time global view.

      Debug Mode:
          Only win_control exists.  It drives VSync timing
          (waitBlanking=True, checkTiming=True, recordFrameIntervals=True).

      Production Mode:
          win_control is demoted to an async observer
          (waitBlanking=False, checkTiming=False) so it never blocks the
          VSync cadence of the two physical stimulus windows:
            win_left  — fullscreen on Left Screen ID
            win_right — fullscreen on Right Screen ID
          Only the trial-active physical window is flipped per frame.

    VSync safety:
      The tight render loop *never* calls .flip() on both physical windows
      in the same iteration.  Only the trial-active window is flipped;
      the inactive window retains its last buffer (static baseline).
      win_control is always flipped but in production its waitBlanking is
      disabled so it cannot cause a deadlock.

    === 重构说明 ===
    === Refactoring Notes ===
    - 新增 self.pattern_key 属性：标识当前选定范式。
    - Added self.pattern_key attribute: identifies the currently selected paradigm.
    - _render_static_baseline() 统一渲染 2° 黑点基线（所有 9 种范式）。
    - _render_static_baseline() uniformly renders the 2° dark dot baseline (for all 9 paradigms).
    - generate_trial_matrix() 现在接收 pattern_key 参数。
    - generate_trial_matrix() now receives the pattern_key parameter.
    """

    def __init__(self, exp_info: Dict[str, Any]):
        self.exp_info = exp_info

        # ---- Parse GUI fields ----
        iti_range = self.exp_info['ITI Range (sec)'].split('-')
        self.iti_min = float(iti_range[0])
        self.iti_max = float(iti_range[1])

        isi_range = self.exp_info['ISI Range (sec)'].split('-')
        self.isi_min = float(isi_range[0])
        self.isi_max = float(isi_range[1])

        self.start_session_num = int(self.exp_info['Session Number'])
        self.total_sessions = int(self.exp_info.get('Total Sessions', 1))

        # Display topology parameters from GUI
        self.debug_mode: bool = bool(self.exp_info.get('Debug Mode (Single Screen)', True))
        self.screen_left: int = int(self.exp_info.get('Left Screen ID', 1))
        self.screen_right: int = int(self.exp_info.get('Right Screen ID', 2))
        self.save_terminal_log: bool = bool(self.exp_info.get('Save Terminal Log (Debug)', False))

        # ==================================================================
        # 【重构核心】解析 GUI 选中的范式
        # [Refactoring Core] Parse the paradigm selected in the GUI
        # ------------------------------------------------------------------
        # pattern_key: 范式键名，用于传递给 generate_trial_matrix()
        # pattern_key: paradigm key name, used to pass to generate_trial_matrix()
        # 所有 9 种范式共享统一的 2° 黑点稳态基线（含 Baseline Wind）。
        # All 9 paradigms share a uniform 2° dark dot steady-state baseline (including Baseline Wind).
        # pattern.md: "Begin Degree: 2°（as baseline on the same time）"
        # 2° 黑点作为全局视觉基线，用于后续数据归一化。
        # The 2° dark dot acts as a global visual baseline, used for subsequent data normalization.
        # ==================================================================
        self.pattern_key: str = self.exp_info.get('Experiment Pattern', PATTERN_CHOICES[0])
        print(f"[LoomingEngine] 选定范式: {self.pattern_key}")
        print(f"[LoomingEngine] 基线策略: 灰底 + 2° 黑点 (通用基线，用于归一化)")

        # 使用选定范式生成试次矩阵
        # Generate the trial matrix using the selected paradigm
        self.trials = generate_trial_matrix(self.pattern_key)

        self.trigger_interface = HardwareTrigger(mode="mock")
        self.logger = GroundTruthLogger()
        self.clock = core.Clock()
        # NOTE: self.kb is initialised in run_experiment() AFTER windows
        # are created — the pyglet keyboard backend needs an active window.
        self.kb: Optional[keyboard.Keyboard] = None

        #===================================================================
        # Physical stimulus angle constraints (degrees) --- Start Degree
        self.initial_angle_deg = 2.0
        self.max_angle_deg = 180.0
        #===================================================================

        # ---- Unified Control Panel objects (populated in run_experiment) ----
        self.win_control: Optional[visual.Window] = None
        self.stim_ctrl_left: Optional[visual.Circle] = None
        self.stim_ctrl_right: Optional[visual.Circle] = None
        self.label_left: Optional[visual.TextStim] = None
        self.label_right: Optional[visual.TextStim] = None

        # ---- Physical stimulus objects (production mode only) ----
        self.win_left: Optional[visual.Window] = None
        self.win_right: Optional[visual.Window] = None
        self.stim_left: Optional[visual.Circle] = None
        self.stim_right: Optional[visual.Circle] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_experiment(self, output_dir: str = "."):
        """
        Initialises the Unified Control Panel (and physical stimulus
        windows in Production mode), steps through the randomised
        paradigm, and handles clean shutdowns.
        """
        # --- Debug-only: activate terminal log tee ---
        self._console_logger: Optional[ConsoleLogger] = None
        if self.debug_mode and self.save_terminal_log:
            os.makedirs(output_dir, exist_ok=True)
            log_filename = (f"{self.exp_info['Subject ID']}"
                        #    f"_session_{self.exp_info['Session Number']}"
                           f"_terminal.txt")
            log_path = os.path.join(output_dir, log_filename)
            self._console_logger = ConsoleLogger(log_path)
            sys.stdout = self._console_logger

        # --- Register a virtual monitor profile (bypass calibration error) ---
        my_monitor = monitors.Monitor('cricket_monitor')
        my_monitor.setWidth(53.0)            # Physical screen width (cm)
        my_monitor.setDistance(30.0)          # Subject-to-screen distance (cm)
        my_monitor.setSizePix((1920, 1080))  # Screen resolution
        my_monitor.saveMon()

        # ==============================================================
        # Window Initialisation — Unified Control Panel Architecture
        # ==============================================================

        if self.debug_mode:
            # ----- DEBUG MODE -----
            # Only win_control: drives VSync, checkTiming, frame recording.
            print("[LoomingEngine] Debug Mode ON — single control panel on screen 0")

            self.win_control = visual.Window(
                size=(1000, 500),
                monitor=my_monitor,
                screen=0,
                color=[0, 0, 0],           # Mid-gray — global static baseline
                colorSpace='rgb',
                units='deg',
                fullscr=False,
                allowGUI=True,
                waitBlanking=True,
                checkTiming=True,
            )
            self.win_control.recordFrameIntervals = True

        else:
            # ----- PRODUCTION MODE -----
            # win_control: async observer (NO VSync blocking).
            # win_left / win_right: fullscreen with VSync.
            print(f"[LoomingEngine] Production Mode — control panel on screen 0, "
                  f"LEFT on screen {self.screen_left}, "
                  f"RIGHT on screen {self.screen_right}")

            self.win_control = visual.Window(
                size=(1000, 500),
                monitor=my_monitor,
                screen=0,
                color=[0, 0, 0],
                colorSpace='rgb',
                units='deg',
                fullscr=False,
                allowGUI=True,
                waitBlanking=False,
                checkTiming=False,
            )
            self.win_control.recordFrameIntervals = False

            # Left physical stimulus window (VSync master)
            self.win_left = visual.Window(
                size=(1920, 1080),
                monitor=my_monitor,
                screen=self.screen_left,
                color=[0, 0, 0],
                colorSpace='rgb',
                units='deg',
                fullscr=True,
                allowGUI=False,
                waitBlanking=True,
                checkTiming=False,
            )
            self.win_left.recordFrameIntervals = False

            # Right physical stimulus window (VSync master)
            self.win_right = visual.Window(
                size=(1920, 1080),
                monitor=my_monitor,
                screen=self.screen_right,
                color=[0, 0, 0],
                colorSpace='rgb',
                units='deg',
                fullscr=True,
                allowGUI=False,
                waitBlanking=True,
                checkTiming=False,
            )
            self.win_right.recordFrameIntervals = False

        # ---- Control panel stimulus mirrors (always present) ----
        self.stim_ctrl_left = visual.Circle(
            self.win_control, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1],
            radius=self.initial_angle_deg / 2.0,
            pos=(-15, 0)
        )
        self.stim_ctrl_right = visual.Circle(
            self.win_control, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1],
            radius=self.initial_angle_deg / 2.0,
            pos=(15, 0)
        )
        self.label_left = visual.TextStim(
            self.win_control, text="[LEFT MONITOR]",
            pos=(-15, 10), color=[1, 1, 1], height=1.2
        )
        self.label_right = visual.TextStim(
            self.win_control, text="[RIGHT MONITOR]",
            pos=(15, 10), color=[1, 1, 1], height=1.2
        )

        # ---- Physical stimulus objects (production only) ----
        if not self.debug_mode:
            self.stim_left = visual.Circle(
                self.win_left, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1],
                radius=self.initial_angle_deg / 2.0,
                pos=(0, 0)
            )
            self.stim_right = visual.Circle(
                self.win_right, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1],
                radius=self.initial_angle_deg / 2.0,
                pos=(0, 0)
            )

        # ---- Initialise keyboard AFTER windows exist ----
        # The pyglet backend routes key events to the focused window;
        # creating the Keyboard before any window exists means getKeys()
        # silently returns nothing.
        self.kb = keyboard.Keyboard()

        # ---- Initial baseline render on all active windows ----
        self._render_static_baseline()

        # Force the control window to the foreground so it receives key events.
        # Without this, windowed-mode windows may not have OS focus and
        # kb.getKeys() will silently return empty.
        try:
            self.win_control.winHandle.activate()
        except Exception:
            pass  # Some backends may not support .activate()

        print(f"[Config] 范式: {self.pattern_key}")
        print(f"[Config] Will run {self.total_sessions} session(s) "
              f"starting from Session {self.start_session_num}.")

        # ==============================================================
        # Wait-for-Start: animal adaptation / manual launch gate
        # Fires ONCE before all sessions — auto-cycling handles the rest.
        # Control panel is flipped every iteration to prevent ghosting.
        #
        # 【重构】此阶段已根据范式类型渲染稳态基线：
        # [Refactoring] This phase now renders the steady-state baseline based on the paradigm type:
        #   - baseline_wind → 纯灰屏幕
        #   - baseline_wind → pure gray screen
        #   - 其他范式     → 灰底 + 2° 黑点（Anti-Startle）
        #   - other paradigms → gray background + 2° dark dot (Anti-Startle)
        # ==============================================================
        print("\n[Ready] Animal adaptation phase. "
              "Press [SPACE] to start, or [ESCAPE] to abort...")

        self.kb.clearEvents()  # flush any stale keypresses
        start_experiment = False
        while not start_experiment:
            self._render_static_baseline()

            keys = self.kb.getKeys(['space', 'escape'], waitRelease=False)
            for key in keys:
                if key.name == 'escape':
                    print("\n[Aborted] Experiment cancelled during adaptation phase.")
                    self._cleanup_windows()
                    core.quit()
                if key.name == 'space':
                    start_experiment = True

        # SPACE was pressed — proceed with experiment
        print("\n=== Experiment Started ===")

        # ==============================================================
        # Outer Session Loop: auto-cycles through all requested sessions.
        # The SPACE gate above is intentionally OUTSIDE this loop so the
        # experimenter presses SPACE once and the system auto-runs all
        # sessions with ISI rest periods in between.
        # ==============================================================
        try:
            for session_idx in range(self.total_sessions):
                current_session_num = self.start_session_num + session_idx

                # 【重构】使用选定的单一范式重新生成试次矩阵
                # [Refactoring] Regenerate the trial matrix using the selected single paradigm
                self.trials = generate_trial_matrix(self.pattern_key)
                self.logger.events = []

                print(f"\n{'='*50}")
                print(f"  Session {current_session_num} "
                      f"({session_idx + 1}/{self.total_sessions}) — "
                      f"{len(self.trials)} trials | "
                      f"范式: {self.pattern_key}")
                print(f"{'='*50}")

                # ---- Inner Trial Loop ----
                for trial_idx, trial in enumerate(self.trials):
                    if trial_idx > 0:
                        iti_dur = random.uniform(self.iti_min, self.iti_max)
                        print(f"\n--- ITI before Trial {trial_idx + 1}/{len(self.trials)}: "
                              f"{iti_dur:.1f}s ---")
                        self._wait_iti(iti_dur)

                    print(f"Running Trial {trial_idx + 1} | Type: {trial['type']} "
                          f"| TTC: {trial.get('target_ttc_ms', '-')}ms "
                          f"| Wind: {trial['wind_dir']}")
                    self._run_single_trial(trial_idx, trial)

                # ---- Save this session's CSV immediately ----
                os.makedirs(output_dir, exist_ok=True)
                prefix = self.exp_info['Subject ID']
                filename = f"{prefix}_session_{current_session_num}.csv"
                self.logger.save_log(os.path.join(output_dir, filename))
                print(f"[Session {current_session_num}] Data saved: {filename}")

                # ---- ISI between sessions (skip after the last one) ----
                if session_idx < self.total_sessions - 1:
                    next_session_num = self.start_session_num + session_idx + 1
                    isi_dur = random.uniform(self.isi_min, self.isi_max)
                    print(f"\n--- ISI before Session {next_session_num}: "
                          f"{isi_dur:.1f}s ---")
                    self._wait_isi(isi_dur, next_session_num)

            print("\n=== All Sessions Completed ===")
        finally:
            self._cleanup_windows()

    # ------------------------------------------------------------------
    # Static baseline renderer (shared by Wait-for-Start, ITI, cleanup)
    # ------------------------------------------------------------------
    def _render_static_baseline(self, extra_ctrl_stims: list = None):
        """
        【重构】渲染通用稳态基线 — 灰底 + 2° 黑点。
        [Refactoring] Render the universal steady-state baseline — gray background + 2° dark dot.

        所有 9 种范式（含 Baseline Wind）共享同一基线状态：
        All 9 paradigms (including Baseline Wind) share the same baseline state:
        灰色背景 + 屏幕中央 2° 静态黑点。
        Gray background + 2° static dark dot in the center of the screen.

        设计依据 (pattern.md):
        Design rationale (pattern.md):
          "Begin Degree: 2°（as baseline on the same time）"
        2° 黑点同时满足两个核心需求：
        The 2° dark dot simultaneously satisfies two core requirements:
          1. 归一化基线：为后续行为数据分析提供统一的视觉参考。
          1. Normalization baseline: provides a unified visual reference for subsequent behavioral data analysis.
          2. Anti-Startle：消除视觉刺激突然出现导致的惊跳反射，
             确保 Looming 从 2° 无缝膨胀（对 Baseline Wind 则
             保持 2° 不变，仅触发风刺激）。
          2. Anti-Startle: eliminates startle reflex caused by sudden visual stimulus appearance,
             ensuring Looming seamlessly expands from 2° (for Baseline Wind, it
             remains at 2°, only triggering the wind stimulus).

        控制面板 (win_control) 上同时在左右镜像位置绘制对应的 2° 小圆，
        为实验者提供实时视觉反馈。
        The corresponding 2° small circles are drawn at mirrored left and right positions on the control panel 
        (win_control) to provide real-time visual feedback to the experimenter.

        Parameters
        ----------
        extra_ctrl_stims : list, optional
            Additional PsychoPy stimuli to draw on win_control before
            the single flip (e.g. ISI countdown text).  This avoids a
            double-flip flicker.
        """
        # 重置控制面板镜像刺激的半径为初始 2° 基线值
        # Reset the radius of the control panel mirrored stimuli to the initial 2° baseline value
        # （Looming 试次结束后半径可能被放大到 180°）
        # (The radius might have been expanded to 180° after a Looming trial ends)
        self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0
        self.stim_ctrl_right.radius = self.initial_angle_deg / 2.0

        # 控制面板：绘制两个 2° 小圆 + 标签
        # Control panel: Draw two 2° small circles + labels
        self.stim_ctrl_left.draw()
        self.stim_ctrl_right.draw()
        self.label_left.draw()
        self.label_right.draw()
        if extra_ctrl_stims:
            for stim in extra_ctrl_stims:
                stim.draw()
        self.win_control.flip()

        # 物理屏幕：绘制中央 2° 黑点 (production only)
        # Physical screen: Draw the central 2° dark dot (production only)
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
    # Window cleanup helper
    # ------------------------------------------------------------------
    def _cleanup_windows(self):
        """Close all open windows safely."""
        if self.win_control is not None:
            self.win_control.close()
        if self.win_left is not None:
            self.win_left.close()
        if self.win_right is not None:
            self.win_right.close()

    # ------------------------------------------------------------------
    # ITI handler
    # ------------------------------------------------------------------
    def _wait_iti(self, duration: float):
        """
        ITI 期间渲染通用稳态基线（灰底 + 2° 黑点）。
        Render the universal steady-state baseline (gray background + 2° dark dot) during ITI.

        所有范式均使用统一的 2° 黑点基线。
        All paradigms use the uniform 2° dark dot baseline.
        Uses _render_static_baseline() to keep all active windows alive
        and prevent OS ghosting.  This is safe because ITI is not
        timing-critical.
        """
        self.logger.log_event("iti_start", self.clock.getTime(), duration=duration)
        t0 = self.clock.getTime()
        while True:
            if (self.clock.getTime() - t0) >= duration:
                break

            self._render_static_baseline()

            if self.kb.getKeys(['escape']):
                print("Experiment aborted by user during ITI.")
                self._cleanup_windows()
                core.quit()

    # ------------------------------------------------------------------
    # ISI handler (Inter-Session Interval)
    # ------------------------------------------------------------------
    def _wait_isi(self, duration: float, next_session_num: int):
        """
        Inter-Session Interval wait with countdown display on win_control.
        Maintains static baseline on all physical windows via
        _render_static_baseline() and overlays a yellow countdown on the
        control panel.
        """
        self.logger.log_event("isi_start", self.clock.getTime(), duration=duration)
        countdown_text = visual.TextStim(
            self.win_control,
            text='',
            pos=(0, -8),
            color='yellow',
            height=1.5,
            bold=True,
        )
        t0 = self.clock.getTime()
        while True:
            elapsed = self.clock.getTime() - t0
            remaining = duration - elapsed
            if remaining <= 0:
                break

            # Update text
            countdown_text.text = (f"ISI Countdown: {int(remaining)}s\n"
                                   f"Next: Session {next_session_num}")

            # Render everything using a SINGLE flip per frame
            self._render_static_baseline(extra_ctrl_stims=[countdown_text])

            if self.kb.getKeys(['escape']):
                print("Experiment aborted by user during ISI.")
                self._cleanup_windows()
                core.quit()

        self.logger.log_event("isi_end", self.clock.getTime())

    # ------------------------------------------------------------------
    # Trial routing helpers
    # ------------------------------------------------------------------
    def _resolve_active_window(self, trial: Dict[str, Any]):
        """
        Determine active control-panel mirror stimulus and, in production
        mode, the corresponding physical window/stimulus.

        Returns
        -------
        (active_ctrl_stim, active_phys_win, active_phys_stim, side_label)
            active_ctrl_stim : Circle on win_control that will be animated.
            active_phys_win  : physical Window (None in debug mode).
            active_phys_stim : physical Circle (None in debug mode).
            side_label       : 'left' or 'right'.
        """
        # screen_side provides an explicit override for baseline_visual
        # trials that have no wind but must be routed to a specific screen.
        screen_side = trial.get('screen_side')
        wind_dir = trial.get('wind_dir', 'none')

        if screen_side == 'right' or wind_dir == 'right':
            return (self.stim_ctrl_right,
                    self.win_right, self.stim_right, 'right')
        else:
            # 'left', 'none', or explicit screen_side='left'
            return (self.stim_ctrl_left,
                    self.win_left, self.stim_left, 'left')

    # ------------------------------------------------------------------
    # Single trial executor
    # ------------------------------------------------------------------
    def _run_single_trial(self, trial_idx: int, trial: Dict[str, Any]):
        t_start = self.clock.getTime()

        # Ground Truth: log the target TTC for this trial
        _gt_target_ttc_ms = trial.get('target_ttc_ms')

        self.logger.log_event(
            "trial_start", t_start,
            trial_index=trial_idx,
            target_ttc_ms=_gt_target_ttc_ms,
            pattern=self.pattern_key,  # 【新增】记录当前范式
                                       # [Added] Log the current paradigm
            **{k: v for k, v in trial.items()
               if k != 'target_ttc_ms'}  # avoid duplicate key
        )

        trigger_sent = False

        # Resolve active stimulus targets for this trial
        active_ctrl_stim, active_phys_win, active_phys_stim, side_label = \
            self._resolve_active_window(trial)
        print(f"  [Routing] Active screen: {side_label}")

        # ============================================================
        # LOOMING trials  (baseline_visual | looming_wind)
        #
        # Elapsed-time physics engine:
        #   t_collision = absolute time (from looming onset) when the
        #                 object reaches the observer (θ = 180°).
        #   t_trigger   = t_collision + (target_ttc_ms / 1000)
        #                 negative target_ttc_ms → trigger BEFORE collision
        #                 positive target_ttc_ms → trigger AFTER collision
        #
        # The loop runs until elapsed >= t_collision + 1.0 s so that
        # post-collision triggers (e.g. +200 ms) have time to fire.
        #
        # 【重构说明】Looming 起始时，黑点已经以 2° 大小停留在屏幕上
        # [Refactoring Notes] At Looming onset, the dark dot is already resting on the screen at a 2° size
        # （由 _render_static_baseline 持续渲染）。Looming 开始后，
        # (Continuously rendered by _render_static_baseline). Once Looming begins, 
        # 引擎从 initial_angle_deg=2° 开始逐帧膨胀，实现无缝衔接。
        # the engine expands frame-by-frame starting from initial_angle_deg=2°, achieving a seamless transition.
        # ============================================================
        if trial['type'] in ['baseline_visual', 'looming_wind']:
            lv_ratio_sec = trial['lv_ratio_ms'] / 1000.0

            # Absolute time from looming onset to collision
            t_collision = lv_ratio_sec / math.tan(
                math.radians(self.initial_angle_deg) / 2)

            # Absolute time for airflow trigger
            # Convention: target_ttc_ms is signed TTC *remaining* at trigger.
            #   negative → before collision → t_trigger < t_collision
            #   zero     → at collision     → t_trigger = t_collision
            #   positive → after collision  → t_trigger > t_collision
            t_trigger = None
            if trial['type'] == 'looming_wind' and _gt_target_ttc_ms is not None:
                t_trigger = t_collision + (_gt_target_ttc_ms / 1000.0)
                print(f"  [TTC] target_ttc_ms={_gt_target_ttc_ms}ms → "
                      f"t_collision={t_collision*1000:.2f}ms, "
                      f"t_trigger={t_trigger*1000:.2f}ms")

            # ===========================================================
            # Looming expansion (forward elapsed-time engine)
            #
            # Per-frame rendering order:
            #   1. Update active_ctrl_stim radius → draw both ctrl stims
            #      + labels → flip win_control.
            #   2. (Production only) Update active_phys_stim radius →
            #      draw → flip active_phys_win.  This flip provides the
            #      authoritative VSync timestamp.
            #   3. (Debug only) Use win_control flip time as timestamp.
            # ===========================================================

            # 【修正】确定非活动侧刺激对象，用于 Looming 期间维持 2° 基线
            # [Fix] Determine the inactive side stimulus object to maintain the 2° baseline during Looming
            if side_label == 'left':
                _inactive_ctrl = self.stim_ctrl_right
                _inactive_phys_win = self.win_right if not self.debug_mode else None
                _inactive_phys_stim = self.stim_right if not self.debug_mode else None
            else:
                _inactive_ctrl = self.stim_ctrl_left
                _inactive_phys_win = self.win_left if not self.debug_mode else None
                _inactive_phys_stim = self.stim_left if not self.debug_mode else None

            # 非活动物理窗口设为非阻塞，避免双重 VSync 卡顿
            # Set the inactive physical window to non-blocking to avoid double VSync stuttering
            if _inactive_phys_win is not None:
                _inactive_phys_win.waitBlanking = False

            looming_t0 = self.clock.getTime()
            first_frame_logged = False
            collision_logged = False

            # Debug-only: per-trial frame timestamp collector
            if self.debug_mode:
                frame_timestamps: List[float] = []

            while True:
                current_time = self.clock.getTime()
                elapsed = current_time - looming_t0

                # Termination: 1 s after collision to allow late triggers
                if elapsed >= t_collision + 1.0:
                    self.logger.log_event("looming_completed", current_time,
                                          elapsed=elapsed,
                                          t_collision=t_collision)
                    break

                # ----------------------------------------------------------
                # Piecewise angular computation
                # ----------------------------------------------------------
                if elapsed < t_collision:
                    # Pre-collision: normal looming expansion
                    ttc_remaining = t_collision - elapsed
                    theta_rad = 2 * math.atan(lv_ratio_sec / ttc_remaining)
                    theta_deg = math.degrees(theta_rad)
                else:
                    # Post-collision (afterimage): screen fully covered
                    theta_deg = self.max_angle_deg  # 180.0
                    if not collision_logged:
                        self.logger.log_event(
                            "collision_reached", current_time,
                            elapsed=elapsed,
                            t_collision=t_collision)
                        collision_logged = True

                # ----------------------------------------------------------
                # Airflow trigger: fire when elapsed reaches t_trigger
                # ----------------------------------------------------------
                if not trigger_sent and t_trigger is not None \
                        and elapsed >= t_trigger:
                    actual_ttc = t_collision - elapsed  # signed TTC at fire
                    self.trigger_interface.send_trigger(
                        event_code="LOOMING_WIND_NODE",
                        current_time=current_time,
                        wind_delay_ms=0,
                        wind_direction=trial['wind_dir']
                    )
                    self.logger.log_event(
                        "wind_triggered", current_time,
                        target_ttc_ms=_gt_target_ttc_ms,
                        actual_ttc_sec=actual_ttc,
                        elapsed=elapsed,
                        t_trigger=t_trigger,
                        t_collision=t_collision
                    )
                    trigger_sent = True

                # ---- Control panel mirror (always) ----
                active_ctrl_stim.radius = theta_deg / 2.0
                active_ctrl_stim.draw()
                # 【修正】同时绘制非活动侧 2° 基线黑点，防止另一侧基线消失
                # [Fix] Simultaneously draw the 2° baseline dark dot on the inactive side to prevent its baseline from disappearing
                # [Fix] Draw 2° baseline dark dot on inactive screen
                _inactive_ctrl.radius = self.initial_angle_deg / 2.0
                _inactive_ctrl.draw()
                self.label_left.draw()
                self.label_right.draw()
                ctrl_flip = self.win_control.flip()

                # ---- Physical window (production only) ----
                if not self.debug_mode and active_phys_win is not None:
                    active_phys_stim.radius = theta_deg / 2.0
                    active_phys_stim.draw()
                    flip_time = active_phys_win.flip()
                    # 【修正】刷新非活动侧物理屏幕 2° 基线（非阻塞 flip）
                    # [Fix] Refresh the 2° baseline on the inactive physical screen (non-blocking flip)
                    if _inactive_phys_stim is not None and _inactive_phys_win is not None:
                        _inactive_phys_stim.radius = self.initial_angle_deg / 2.0
                        _inactive_phys_stim.draw()
                        _inactive_phys_win.flip()
                else:
                    flip_time = ctrl_flip

                if self.debug_mode:
                    frame_timestamps.append(flip_time)
                if not first_frame_logged:
                    self.logger.log_event("first_frame", flip_time,
                                          initial_angle=theta_deg,
                                          active_screen=side_label)
                    first_frame_logged = True

                if self.kb.getKeys(['escape']):
                    core.quit()

            # 恢复非活动物理窗口的 VSync 阻塞设置
            # Restore the VSync blocking setting for the inactive physical window
            if _inactive_phys_win is not None:
                _inactive_phys_win.waitBlanking = True

            # Debug-only: IFI diagnostic summary for this trial
            if self.debug_mode and len(frame_timestamps) > 1:
                import numpy as np
                ifis = np.diff(frame_timestamps) * 1000.0  # ms
                print(f"  [IFI Diag] Trial {trial_idx+1}: "
                      f"frames={len(frame_timestamps)}, "
                      f"mean={ifis.mean():.2f}ms, "
                      f"std={ifis.std():.2f}ms, "
                      f"min={ifis.min():.2f}ms, "
                      f"max={ifis.max():.2f}ms")

        # ============================================================
        # BASELINE WIND  (pure air-pump control — 视觉保持 2° 基线)
        # BASELINE WIND  (pure air-pump control — visually maintain 2° baseline)
        # 屏幕持续显示 2° 黑点（通用基线），不做任何视觉变化。
        # The screen continuously displays the 2° dark dot (universal baseline) with no visual changes.
        # 仅在随机延迟后触发风刺激。2° 黑点确保归一化一致性。
        # Wind stimulus is only triggered after a random delay. The 2° dark dot ensures normalization consistency.
        # Control panel is flipped every frame to stay alive.
        # Physical window (production) is also flipped to stay alive.
        # ============================================================
        elif trial['type'] == 'baseline_wind':
            # Random delay before wind fires, matched to the new paradigm
            # time range to control for habituation to standing duration.
            random_delay_sec = random.uniform(0.1, 1.2)
            # Post-trigger observation: 1–2 s after the wind fires,
            # then end the trial.  Total ≈ random_delay + post_observe.
            post_trigger_wait = random.uniform(1.0, 2.0)
            print(f"  [Baseline Wind] Random delay: "
                  f"{random_delay_sec*1000:.0f}ms | "
                  f"post-trigger obs: {post_trigger_wait:.2f}s")
            self.logger.log_event("baseline_wind_start", self.clock.getTime(),
                                  random_delay_sec=random_delay_sec)

            t0 = self.clock.getTime()
            _wind_fired_time = None  # track when trigger actually fires

            while True:
                current_time = self.clock.getTime()
                elapsed = current_time - t0

                # End trial after post-trigger observation period
                if _wind_fired_time is not None and \
                   (current_time - _wind_fired_time) >= post_trigger_wait:
                    break

                # Fire wind trigger after the random delay
                if elapsed >= random_delay_sec and not trigger_sent:
                    self.trigger_interface.send_trigger(
                        event_code="BASELINE_WIND_TRIGGER",
                        current_time=current_time,
                        wind_delay_ms=0,
                        wind_direction=trial['wind_dir']
                    )
                    self.logger.log_event("wind_triggered", current_time,
                                          random_delay_sec=random_delay_sec)
                    trigger_sent = True
                    _wind_fired_time = current_time

                # 【修正】Baseline Wind 试次内也渲染 2° 黑点基线
                # [Fix] Also render the 2° dark dot baseline within the Baseline Wind trials
                # 保持与所有其他范式一致的视觉基线，用于归一化
                # Maintain a visual baseline consistent with all other paradigms for normalization purposes
                self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0
                self.stim_ctrl_right.radius = self.initial_angle_deg / 2.0
                self.stim_ctrl_left.draw()
                self.stim_ctrl_right.draw()
                self.label_left.draw()
                self.label_right.draw()
                self.win_control.flip()

                # Physical window (production only — 2° 黑点基线)
                # Physical window (production only — 2° dark dot baseline)
                if not self.debug_mode and active_phys_win is not None:
                    active_phys_stim.radius = self.initial_angle_deg / 2.0
                    active_phys_stim.draw()
                    active_phys_win.flip()

                if self.kb.getKeys(['escape']):
                    core.quit()

        # ============================================================
        # POST-TRIAL CLEANUP: force all windows to static baseline.
        # 所有范式统一恢复 2° 黑点基线
        # All paradigms uniformly revert to the 2° dark dot baseline
        # ============================================================
        self._render_static_baseline()


# ==============================================================================
# Module 1: Pre-Experiment GUI & Entrypoint
# ==============================================================================

def launch_experiment_gui() -> Optional[Dict[str, Any]]:
    """
    Raises a blocking Dlg (grouped layout) native to PsychoPy to receive
    real-time experiment parameters.  Uses addText() section headers to
    visually separate core experiment settings from debug / hardware knobs,
    improving operator ergonomics and reducing configuration errors.

    Compatible with PsychoPy >= 2024.1 dictionary-based Dlg API:
    addField(key, label=..., initial=...) where *key* is the internal
    identifier and *label* is the text displayed in the GUI.

    Three-Monitor Architecture GUI:
      - 'Left Screen ID' (default 1): OS display index for left stimulus.
      - 'Right Screen ID' (default 2): OS display index for right stimulus.
      - Debug Mode: opens two small side-by-side windows on Screen 0.

    === 重构说明 ===
    === Refactoring Notes ===
    新增 Experiment Pattern 下拉选择框，包含 9 种实验范式：
    Added Experiment Pattern dropdown menu, containing 9 experimental paradigms:
    实验者在 GUI 中为当前被试选定一种范式后，整个 Session 仅运行该范式的 18 个试次。
    After the experimenter selects a paradigm for the current subject in the GUI, the entire Session will only run 18 trials of that paradigm.
    这实现了"单范式锁定"机制 (Single-Pattern per Subject)。
    This implements the "Single-Pattern Locking" mechanism (Single-Pattern per Subject).

    Returns a dict with keys consumed by LoomingEngine.__init__().
    """
    dlg = gui.Dlg(title="BioMoR Looming Paradigms")

    # ---- Section 1: Core Experiment Parameters ----
    dlg.addText("=== Core Experiment Parameters ===")
    dlg.addField('Subject ID',
                 label='✱Subject ID:',
                 initial='cricket_001',
                 tip='Enter the subject ID')
    dlg.addField('Session Number',
                 label='✱Start Session Number:',
                 initial=1,
                 tip='Enter the starting session number')
    dlg.addField('Total Sessions',
                 label='Total Sessions to run:',
                 initial=2,
                 tip='Number of sessions to run consecutively (pattern.md: 2 sessions per cricket)')

    # ==================================================================
    # 【重构核心】Experiment Pattern 下拉选择框
    # [Refactoring Core] Experiment Pattern dropdown menu
    # ------------------------------------------------------------------
    # 使用 choices= 参数创建下拉菜单（PsychoPy Dlg 原生支持）。
    # Created dropdown menu using choices= parameter (natively supported by PsychoPy Dlg).
    # 下拉列表包含 pattern.md 中定义的全部 9 种范式：
    # The dropdown list contains all 9 paradigms defined in pattern.md:
    #   1. Baseline Visual (仅视觉，无风)
    #   1. Baseline Visual (Visual only, no wind)

    #   2. Baseline Wind (仅风，随机延迟)
    #   2. Baseline Wind (Wind only, random delay)

    #   3-8. Looming + Wind (6 种负 TTC / 对应角度)
    #   3-8. Looming + Wind (6 negative TTCs / corresponding angles)

    #   9. Looming + Wind (TTC +200ms)
    #   9. Looming + Wind (TTC +200ms)
    
    # 默认选中第一项 (Baseline Visual)，实验者可从下拉菜单中选择。
    # Default selection is the first item (Baseline Visual), and the experimenter can select from the dropdown menu.
    # ==================================================================
    dlg.addField('Experiment Pattern',
                 label='✱Experiment Pattern:',
                 choices=PATTERN_CHOICES,
                 tip='选择当前被试的实验范式。整个 Session 将仅运行所选范式的 18 个试次。')

    dlg.addField('ITI Range (sec)',
                 label='ITI Range (sec):',
                 initial='60-90',
                 tip='Enter the ITI range (e.g. 60-90)')
    dlg.addField('ISI Range (sec)',
                 label='ISI Range (sec):',
                 initial='300-600',
                 tip='Enter the ISI range in seconds (pattern.md: 5~10 min = 300-600)')
    dlg.addField('Left Screen ID',
                 label='Left Screen ID:',
                 initial='1',
                 tip='OS display index for the LEFT stimulus screen (e.g. 1)')
    dlg.addField('Right Screen ID',
                 label='Right Screen ID:',
                 initial='2',
                 tip='OS display index for the RIGHT stimulus screen (e.g. 2)')

    # ---- Section 2: Hardware & Debug Settings ----
    dlg.addText("\n\n")
    dlg.addText("=== Hardware & Debug Settings ===")
    dlg.addField('Debug Mode (Single Screen)',
                 label='Debug Mode (Single Screen):',
                 initial=True,
                 tip='Check for two side-by-side debug windows on screen 0; '
                     'uncheck for fullscreen production on left/right screens')
    dlg.addField('Save Terminal Log (Debug)',
                 label='Save Terminal Log (Debug):',
                 initial=False,
                 tip='Only active in Debug Mode. '
                     'Saves all console prints to a .txt file.')

    ok_data = dlg.show()

    if not dlg.OK:
        return None

    # ok_data is a dict keyed by the field keys defined above
    return ok_data


if __name__ == "__main__":
    # 1. Blocking GUI Parameter Polling
    config = launch_experiment_gui()
    if not config:
        print("User cancelled GUI configuration. Exiting.")
        core.quit()
        
    print("\n--- Session Configuration Captured ---")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    # 2. Main Experiment Loop execution
    engine = LoomingEngine(exp_info=config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine.run_experiment(output_dir=script_dir)

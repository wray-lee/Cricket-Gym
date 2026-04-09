"""
stimulus_controller.py

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
2. Built-in, structured 16-trial Paradigm matrix.
3. Exact VSync-locked timing via PsychoPy (sub-millisecond precision).
4. Modular Hardware Mock Interface for external triggering (e.g., Arduino).
5. Precision logging system tracking paradigm configurations down to VSyncs.
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
# Module 2: Paradigm & Trial Matrix Generator
# ==============================================================================

def generate_trial_matrix() -> List[Dict[str, Any]]:
    """
    Generates a randomized sequence of 16 trials according to the 8 core conditions 
    repeated twice for 'left' and 'right' wind directions.
    """
    trials = []
    
    # The 8 core conditions defined in the BioMoR paradigm specification
    core_conditions = [
        {'type': 'baseline_visual', 'lv_ratio_ms': 100,  'wind_delay_ms': None},
        {'type': 'baseline_wind',   'lv_ratio_ms': None, 'wind_delay_ms': 0},
        {'type': 'looming_wind',    'lv_ratio_ms': 100,  'wind_delay_ms': 0},
        {'type': 'looming_wind',    'lv_ratio_ms': 100,  'wind_delay_ms': 200},
        {'type': 'looming_wind',    'lv_ratio_ms': 100,  'wind_delay_ms': 1000},
        {'type': 'looming_wind',    'lv_ratio_ms': 20,   'wind_delay_ms': 0},
        {'type': 'looming_wind',    'lv_ratio_ms': 20,   'wind_delay_ms': 200},
        {'type': 'looming_wind',    'lv_ratio_ms': 20,   'wind_delay_ms': 1000},
    ]
    
    for cond in core_conditions:
        for direction in ['left', 'right']:
            trial = cond.copy()
            # If baseline visual, no wind is physically required, but mechanically 
            # we assign none to prevent errors or unexpected hardware triggers.
            if trial['type'] == 'baseline_visual':
                trial['wind_dir'] = 'none' 
            else:
                trial['wind_dir'] = direction
            trials.append(trial)
            
    # Randomize the 16 trials to avoid pattern recognition
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
    """

    def __init__(self, exp_info: Dict[str, Any]):
        self.exp_info = exp_info

        # ---- Parse GUI fields ----
        self.target_angle_deg = float(self.exp_info['Target Visual Angle (deg)'])

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

        self.trials = generate_trial_matrix()

        self.trigger_interface = HardwareTrigger(mode="mock")
        self.logger = GroundTruthLogger()
        self.clock = core.Clock()
        # NOTE: self.kb is initialised in run_experiment() AFTER windows
        # are created — the pyglet keyboard backend needs an active window.
        self.kb: Optional[keyboard.Keyboard] = None

        # Physical stimulus angle constraints (degrees)
        self.initial_angle_deg = 2.0
        self.max_angle_deg = 90.0

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
                           f"_session_{self.exp_info['Session Number']}"
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

        print(f"[Config] Will run {self.total_sessions} session(s) "
              f"starting from Session {self.start_session_num}.")

        # ==============================================================
        # Wait-for-Start: animal adaptation / manual launch gate
        # Fires ONCE before all sessions — auto-cycling handles the rest.
        # Control panel is flipped every iteration to prevent ghosting.
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

                # Fresh trial matrix and clean logger for each session
                self.trials = generate_trial_matrix()
                self.logger.events = []

                print(f"\n{'='*50}")
                print(f"  Session {current_session_num} "
                      f"({session_idx + 1}/{self.total_sessions}) — "
                      f"{len(self.trials)} trials")
                print(f"{'='*50}")

                # ---- Inner Trial Loop ----
                for trial_idx, trial in enumerate(self.trials):
                    if trial_idx > 0:
                        iti_dur = random.uniform(self.iti_min, self.iti_max)
                        print(f"\n--- ITI before Trial {trial_idx + 1}/{len(self.trials)}: "
                              f"{iti_dur:.1f}s ---")
                        self._wait_iti(iti_dur)

                    print(f"Running Trial {trial_idx + 1} | Type: {trial['type']} "
                          f"| l/v: {trial['lv_ratio_ms']}ms "
                          f"| Wind: {trial['wind_dir']} {trial['wind_delay_ms']}ms")
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
        Render the resting state on all active windows:
        control panel mirrors at initial_angle_deg size, plus physical
        windows (if production mode).

        Parameters
        ----------
        extra_ctrl_stims : list, optional
            Additional PsychoPy stimuli to draw on win_control before
            the single flip (e.g. ISI countdown text).  This avoids a
            double-flip flicker.
        """
        # ---- Control panel: reset both mirrors to initial size ----
        self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0
        self.stim_ctrl_right.radius = self.initial_angle_deg / 2.0
        self.stim_ctrl_left.draw()
        self.stim_ctrl_right.draw()
        self.label_left.draw()
        self.label_right.draw()
        if extra_ctrl_stims:
            for stim in extra_ctrl_stims:
                stim.draw()
        self.win_control.flip()

        # ---- Physical windows: reset & flip (production only) ----
        if not self.debug_mode:
            self.stim_left.radius = self.initial_angle_deg / 2.0
            self.stim_right.radius = self.initial_angle_deg / 2.0
            self.stim_left.draw()
            self.win_left.flip()
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
        Maintain the static resting state (gray background + stationary
        initial black disc) during the Inter-Trial Interval while
        remaining responsive to 'escape'.

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
        wind_dir = trial.get('wind_dir', 'none')
        if wind_dir == 'right':
            return (self.stim_ctrl_right,
                    self.win_right, self.stim_right, 'right')
        else:
            # 'left' or 'none' (baseline_visual defaults to left screen)
            return (self.stim_ctrl_left,
                    self.win_left, self.stim_left, 'left')

    # ------------------------------------------------------------------
    # Single trial executor
    # ------------------------------------------------------------------
    def _run_single_trial(self, trial_idx: int, trial: Dict[str, Any]):
        t_start = self.clock.getTime()
        self.logger.log_event("trial_start", t_start, trial_index=trial_idx, **trial)

        trigger_sent = False

        # Resolve active stimulus targets for this trial
        active_ctrl_stim, active_phys_win, active_phys_stim, side_label = \
            self._resolve_active_window(trial)
        print(f"  [Routing] Active screen: {side_label}")

        # ============================================================
        # LOOMING trials  (baseline_visual | looming_wind)
        # ============================================================
        if trial['type'] in ['baseline_visual', 'looming_wind']:
            lv_ratio_sec = trial['lv_ratio_ms'] / 1000.0

            # Precompute critical time-to-collision points
            start_ttc = lv_ratio_sec / math.tan(math.radians(self.initial_angle_deg) / 2)
            end_ttc   = lv_ratio_sec / math.tan(math.radians(self.max_angle_deg) / 2)
            ttc_at_target = lv_ratio_sec / math.tan(math.radians(self.target_angle_deg) / 2)

            # ===========================================================
            # Looming expansion
            # Background is already mid-gray; disc starts at initial size.
            # No fade-in needed — resting state is the static baseline.
            #
            # Per-frame rendering order:
            #   1. Update active_ctrl_stim radius → draw both ctrl stims
            #      + labels → flip win_control.
            #   2. (Production only) Update active_phys_stim radius →
            #      draw → flip active_phys_win.  This flip provides the
            #      authoritative VSync timestamp.
            #   3. (Debug only) Use win_control flip time as timestamp.
            # ===========================================================
            looming_t0 = self.clock.getTime()
            first_frame_logged = False

            # Debug-only: per-trial frame timestamp collector
            if self.debug_mode:
                frame_timestamps: List[float] = []

            while True:
                current_time = self.clock.getTime()
                elapsed = current_time - looming_t0
                ttc = start_ttc - elapsed

                if ttc <= end_ttc:
                    self.logger.log_event("looming_completed", current_time, ttc=ttc)
                    break

                # Threshold trigger (target angle reached)
                if not trigger_sent and ttc <= ttc_at_target:
                    event_str = ("LOOMING_WIND_NODE" if trial['type'] == 'looming_wind'
                                 else "BASELINE_VISUAL_NODE")
                    self.trigger_interface.send_trigger(
                        event_code=event_str,
                        current_time=current_time,
                        wind_delay_ms=trial['wind_delay_ms'],
                        wind_direction=trial['wind_dir']
                    )
                    self.logger.log_event("target_angle_reached", current_time, ttc=ttc)
                    trigger_sent = True

                # Compute current angular size
                theta_rad = 2 * math.atan(lv_ratio_sec / ttc)
                theta_deg = math.degrees(theta_rad)

                # ---- Control panel mirror (always) ----
                active_ctrl_stim.radius = theta_deg / 2.0
                self.stim_ctrl_left.draw()
                self.stim_ctrl_right.draw()
                self.label_left.draw()
                self.label_right.draw()
                ctrl_flip = self.win_control.flip()

                # ---- Physical window (production only) ----
                if not self.debug_mode and active_phys_win is not None:
                    active_phys_stim.radius = theta_deg / 2.0
                    active_phys_stim.draw()
                    flip_time = active_phys_win.flip()
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
        # BASELINE WIND  (pure air-pump control — NO visual change)
        # The resting state (gray bg + stationary initial disc) is
        # maintained throughout; only the hardware trigger fires.
        # Control panel is flipped every frame to stay alive.
        # Physical window (production) is also flipped to stay alive.
        # ============================================================
        elif trial['type'] == 'baseline_wind':
            t0 = self.clock.getTime()

            while True:
                current_time = self.clock.getTime()
                elapsed = current_time - t0

                if elapsed >= 4.0:
                    break

                if elapsed >= 2.0 and not trigger_sent:
                    self.trigger_interface.send_trigger(
                        event_code="BASELINE_WIND_TRIGGER",
                        current_time=current_time,
                        wind_delay_ms=trial['wind_delay_ms'],
                        wind_direction=trial['wind_dir']
                    )
                    self.logger.log_event("wind_triggered", current_time)
                    trigger_sent = True

                # Control panel (static baseline)
                self.stim_ctrl_left.draw()
                self.stim_ctrl_right.draw()
                self.label_left.draw()
                self.label_right.draw()
                self.win_control.flip()

                # Physical window (production only — static baseline)
                if not self.debug_mode and active_phys_win is not None:
                    active_phys_stim.draw()
                    active_phys_win.flip()

                if self.kb.getKeys(['escape']):
                    core.quit()

        # ============================================================
        # POST-TRIAL CLEANUP: force all windows to static baseline.
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
                 initial=1,
                 tip='Number of sessions to run consecutively')
    dlg.addField('Target Visual Angle (deg)',
                 label='Target Visual Angle (deg):',
                 initial=30,
                 tip='Enter the target visual angle')
    dlg.addField('ITI Range (sec)',
                 label='ITI Range (sec):',
                 initial='60-90',
                 tip='Enter the ITI range (e.g. 60-90)')
    dlg.addField('ISI Range (sec)',
                 label='ISI Range (sec):',
                 initial='300-600',
                 tip='Enter the ISI range (e.g. 300-600)')
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

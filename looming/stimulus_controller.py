"""
stimulus_controller.py

A high-precision visual stimulus engine for presenting looming stimuli 
driven by classical l/v physical kinematics. This script is designed as 
a scientific-grade software controller for the BioMoR project.

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

    Hardware topology:
      1 display monitor  +  2 air pumps (left / right).
      The trial parameter ``wind_dir`` controls only the hardware pump
      trigger direction — the visual stimulus is ALWAYS rendered at the
      dead-centre of the single monitor.

    Display modes:
      - Debug Mode:      windowed on OS screen 0.
      - Production Mode: fullscreen on ``Stimulus Screen ID``.
    """

    def __init__(self, exp_info: Dict[str, Any]):
        self.exp_info = exp_info

        # ---- Parse GUI fields ----
        self.target_angle_deg = float(self.exp_info['Target Visual Angle (deg)'])

        iti_range = self.exp_info['ITI Range (sec)'].split('-')
        self.iti_min = float(iti_range[0])
        self.iti_max = float(iti_range[1])

        # Display topology parameters from GUI
        self.debug_mode: bool = bool(self.exp_info.get('Debug Mode (Single Screen)', True))
        self.screen_stimulus: int = int(self.exp_info.get('Stimulus Screen ID', 1))
        self.save_terminal_log: bool = bool(self.exp_info.get('Save Terminal Log (Debug)', False))

        self.trials = generate_trial_matrix()

        self.trigger_interface = HardwareTrigger(mode="mock")
        self.logger = GroundTruthLogger()
        self.clock = core.Clock()

        # Physical stimulus angle constraints (degrees)
        self.initial_angle_deg = 2.0
        self.max_angle_deg = 90.0

        # Populated by run_experiment() — single Window & Circle
        self.win: Optional[visual.Window] = None
        self.stimulus: Optional[visual.Circle] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_experiment(self, output_dir: str = "."):
        """
        Initialises the single rendering Window, steps through the
        randomised paradigm, and handles clean shutdowns.
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

        # --- Initialise the ONE window ---
        if self.debug_mode:
            screen_id = 0
            fullscr = False
            print("[LoomingEngine] Debug Mode ON — windowed on screen 0")
        else:
            screen_id = self.screen_stimulus
            fullscr = True
            print(f"[LoomingEngine] Production Mode — fullscreen on screen "
                  f"{self.screen_stimulus}")

        self.win = visual.Window(
            size=(1920, 1080),
            monitor=my_monitor,
            screen=screen_id,
            color=[0, 0, 0],           # Mid-gray — global static baseline
            colorSpace='rgb',
            units='deg',
            fullscr=fullscr,
            allowGUI=(not fullscr),
            checkTiming=False,         # Skip bright frame-rate test screen
        )
        self.win.recordFrameIntervals=self.debug_mode  # IFI diagnostics (debug only)

        # Create the single stimulus circle (black disc at initial size)
        # pos=(0,0) = dead-centre of screen — never offset.
        self.stimulus = visual.Circle(
            self.win, fillColor=[-1, -1, -1], lineColor=[-1, -1, -1],
            radius=self.initial_angle_deg / 2.0,
            pos=(0, 0)
        )

        # First visible frame = gray background + stationary initial black disc
        self.stimulus.draw()
        self.win.flip()

        print(f"Starting Session {self.exp_info['Session Number']} "
              f"with {len(self.trials)} trials.")

        # ==============================================================
        # Wait-for-Start: animal adaptation / manual launch gate
        # ==============================================================
        print("\n[Ready] Animal adaptation phase. "
              "Press [SPACE] to start the first trial, or [ESCAPE] to abort...")

        # Block until the experimenter presses SPACE or ESCAPE.
        # The stimulus screen keeps displaying the resting state
        # (mid-gray background + stationary initial black disc).
        # No visual.TextStim is drawn — the cricket's visual baseline
        # must remain undisturbed.
        event.clearEvents()  # flush any stale keypresses
        keys = event.waitKeys(keyList=['space', 'escape'])

        if 'escape' in keys:
            print("\n[Aborted] Experiment cancelled during adaptation phase.")
            self.win.close()
            core.quit()

        # SPACE was pressed — proceed with experiment
        print("\n=== Experiment Started ===")

        # ---- Main Trial Loop ----
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

        # ---- Cleanup ----
        self.win.close()

        os.makedirs(output_dir, exist_ok=True)
        prefix = self.exp_info['Subject ID']
        session_id = self.exp_info['Session Number']
        filename = f"{prefix}_session_{session_id}.csv"
        self.logger.save_log(os.path.join(output_dir, filename))

    # ------------------------------------------------------------------
    # ITI handler
    # ------------------------------------------------------------------
    def _wait_iti(self, duration: float):
        """
        Maintain the static resting state (gray background + stationary
        initial black disc) during the Inter-Trial Interval while
        remaining responsive to 'escape'.
        """
        self.logger.log_event("iti_start", self.clock.getTime(), duration=duration)
        t0 = self.clock.getTime()
        while True:
            if (self.clock.getTime() - t0) >= duration:
                break
            self.stimulus.draw()
            self.win.flip()
            if 'escape' in event.getKeys():
                print("Experiment aborted by user during ITI.")
                core.quit()

    # ------------------------------------------------------------------
    # Single trial executor
    # ------------------------------------------------------------------
    def _run_single_trial(self, trial_idx: int, trial: Dict[str, Any]):
        t_start = self.clock.getTime()
        self.logger.log_event("trial_start", t_start, trial_index=trial_idx, **trial)

        trigger_sent = False

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

                # Render the expanding black disc at screen centre
                theta_rad = 2 * math.atan(lv_ratio_sec / ttc)
                theta_deg = math.degrees(theta_rad)
                self.stimulus.radius = theta_deg / 2.0
                self.stimulus.draw()

                flip_time = self.win.flip()
                if self.debug_mode:
                    frame_timestamps.append(flip_time)
                if not first_frame_logged:
                    self.logger.log_event("first_frame", flip_time, initial_angle=theta_deg)
                    first_frame_logged = True

                if 'escape' in event.getKeys():
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

            # --- Restore resting state: initial-size disc on gray ---
            self.stimulus.radius = self.initial_angle_deg / 2.0
            self.stimulus.draw()
            self.win.flip()

        # ============================================================
        # BASELINE WIND  (pure air-pump control — NO visual change)
        # The resting state (gray bg + stationary initial disc) is
        # maintained throughout; only the hardware trigger fires.
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

                self.stimulus.draw()
                self.win.flip()
                if 'escape' in event.getKeys():
                    core.quit()


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

    Returns a dict with keys identical to the former DlgFromDict version
    so that downstream LoomingEngine consumption is unchanged.
    """
    dlg = gui.Dlg(title="BioMoR Looming Paradigms")

    # ---- Section 1: Core Experiment Parameters ----
    dlg.addText("=== Core Experiment Parameters ===")
    dlg.addField('Subject ID',
                 label='✱Subject ID:',
                 initial='cricket_001',
                 tip='Enter the subject ID')
    dlg.addField('Session Number',
                 label='✱Session Number:',
                 initial=1,
                 tip='Enter the session number')
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
    dlg.addField('Stimulus Screen ID',
                 label='Stimulus Screen ID:',
                 initial='1',
                 tip='OS display index for the stimulus screen (e.g. 1)')

    # ---- Section 2: Hardware & Debug Settings ----
    dlg.addText("\n\n")
    dlg.addText("=== Hardware & Debug Settings ===")
    dlg.addField('Debug Mode (Single Screen)',
                 label='Debug Mode (Single Screen):',
                 initial=True,
                 tip='Check for single-window debug; '
                     'uncheck for fullscreen production')
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

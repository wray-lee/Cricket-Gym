import re

def translate():
    with open(r'c:\Users\Wray\Desktop\Projects\Cricket-Gym\looming\stimulus_controller.py', 'r', encoding='utf-8') as f:
        content = f.read()

    replacements = [
        (r'=== 重构说明 \(Refactoring Notes\) ===\n  - 引入"单范式锁定"机制：每个被试 \(Subject\) 仅运行 GUI 中选定的一种范式。\n  - Trial Matrix 重构为纯空间随机化：18 试次 = 9 Left \+ 9 Right，完全打乱。\n  - 稳态视觉基线 \(2° 黑点\)：全部 9 种范式（含 Baseline Wind）的所有\n    空闲状态均显示灰底 \+ 2° 黑点，作为归一化基线并消除 Startle Reflex。\n    \(pattern\.md: "Begin Degree: 2°（as baseline on the same time）"\)',
         r'=== 重构说明 (Refactoring Notes) ===\n  - 引入"单范式锁定"机制：每个被试 (Subject) 仅运行 GUI 中选定的一种范式。\n    (Introduced "Single-Pattern Lock" mechanism: Each subject only runs one pattern selected in GUI.)\n  - Trial Matrix 重构为纯空间随机化：18 试次 = 9 Left + 9 Right，完全打乱。\n    (Trial Matrix refactored to pure spatial randomization: 18 trials = 9 Left + 9 Right, completely shuffled.)\n  - 稳态视觉基线 (2° 黑点)：全部 9 种范式（含 Baseline Wind）的所有\n    空闲状态均显示灰底 + 2° 黑点，作为归一化基线并消除 Startle Reflex。\n    (Steady-state visual baseline (2° black dot): All idle states of all 9 patterns (including Baseline Wind)\n    display a gray background + 2° black dot, serving as a normalization baseline and eliminating Startle Reflex.)\n    (pattern.md: "Begin Degree: 2°（as baseline on the same time）")'),
        
        (r'# 定义 9 种实验范式的完整参数表。\n# 每种范式用一个唯一的 key 标识，在 GUI 下拉菜单中显示完整名称。\n# \'type\' 字段决定 Trial 运行逻辑（与原有 baseline_visual / looming_wind /\n# baseline_wind 三分支完全对应）。',
         r'# 定义 9 种实验范式的完整参数表。\n# (Define the complete parameter table for 9 experimental patterns.)\n# 每种范式用一个唯一的 key 标识，在 GUI 下拉菜单中显示完整名称。\n# (Each pattern is identified by a unique key, displaying its full name in the GUI dropdown menu.)\n# \'type\' 字段决定 Trial 运行逻辑（与原有 baseline_visual / looming_wind /\n# baseline_wind 三分支完全对应）。\n# (The \'type\' field determines Trial execution logic, fully corresponding to the original\n# baseline_visual / looming_wind / baseline_wind three branches.)'),

        (r'# GUI 下拉列表的显示顺序（与 pattern\.md 定义一致）',
         r'# GUI 下拉列表的显示顺序（与 pattern.md 定义一致）\n# (Display order of the GUI dropdown list (consistent with pattern.md definition))'),

        (r'    【重构】单范式试次矩阵生成器。\n\n    根据 GUI 中选定的唯一范式 \(pattern_key\)，生成恰好 18 个试次，\n    并在左右方向上实现绝对空间平衡（9 Left \+ 9 Right），然后完全打乱。\n\n    重构要点：\n    ----------\n    - 原逻辑：在 18 个试次中混合所有 7 种 TTC \+ 2 个视觉基线 \+ 2 个风基线。\n    - 新逻辑：一个 Session 只包含一种范式的 18 个试次，\n      唯一的随机化维度是空间方向（left/right）。\n    - 对于 baseline_visual 类型：wind_dir 设为 \'none\'，\n      但通过 screen_side 字段控制刺激呈现在左屏还是右屏，\n      确保 9 次左屏 \+ 9 次右屏的空间平衡。\n    - 对于 baseline_wind 和 looming_wind 类型：\n      wind_dir 直接控制风的方向和屏幕路由。',
         r'    【重构】单范式试次矩阵生成器。\n    ([Refactoring] Single-pattern trial matrix generator.)\n\n    根据 GUI 中选定的唯一范式 (pattern_key)，生成恰好 18 个试次，\n    并在左右方向上实现绝对空间平衡（9 Left + 9 Right），然后完全打乱。\n    (Generates exactly 18 trials based on the selected unique pattern (pattern_key) in GUI,\n    achieving absolute spatial balance (9 Left + 9 Right), and then completely shuffles them.)\n\n    重构要点：\n    (Refactoring Key Points:)\n    ----------\n    - 原逻辑：在 18 个试次中混合所有 7 种 TTC + 2 个视觉基线 + 2 个风基线。\n      (Original logic: Mixed all 7 TTCs + 2 visual baselines + 2 wind baselines in 18 trials.)\n    - 新逻辑：一个 Session 只包含一种范式的 18 个试次，\n      唯一的随机化维度是空间方向（left/right）。\n      (New logic: A Session only contains 18 trials of one pattern,\n      the only randomization dimension is spatial direction (left/right).)\n    - 对于 baseline_visual 类型：wind_dir 设为 \'none\'，\n      但通过 screen_side 字段控制刺激呈现在左屏还是右屏，\n      确保 9 次左屏 + 9 次右屏的空间平衡。\n      (For baseline_visual type: wind_dir is set to \'none\',\n      but screen_side field controls whether stimulus is presented on left or right screen,\n      ensuring spatial balance of 9 left + 9 right screens.)\n    - 对于 baseline_wind 和 looming_wind 类型：\n      wind_dir 直接控制风的方向和屏幕路由。\n      (For baseline_wind and looming_wind types:\n      wind_dir directly controls wind direction and screen routing.)'),

        (r'        EXPERIMENT_PATTERNS 中定义的范式键名，\n        例如 \'Looming \+ Wind \(TTC -373ms / 30°\)\'。',
         r'        EXPERIMENT_PATTERNS 中定义的范式键名，\n        例如 \'Looming + Wind (TTC -373ms / 30°)\'。\n        (The pattern key name defined in EXPERIMENT_PATTERNS, e.g., \'Looming + Wind (TTC -373ms / 30°)\'.)'),

        (r'        包含 18 个完全打乱的试次字典列表。',
         r'        包含 18 个完全打乱的试次字典列表。\n        (List of dictionaries containing 18 completely shuffled trials.)'),

        (r'    # ------------------------------------------------------------------\n    # 生成 18 个试次：9 Left \+ 9 Right，确保绝对空间平衡\n    # ------------------------------------------------------------------',
         r'    # ------------------------------------------------------------------\n    # 生成 18 个试次：9 Left + 9 Right，确保绝对空间平衡\n    # (Generate 18 trials: 9 Left + 9 Right, ensuring absolute spatial balance)\n    # ------------------------------------------------------------------'),

        (r'            # 纯视觉试次：无风，但通过 screen_side 控制呈现屏幕\n            trial_dict\[\'wind_dir\'\] = \'none\'',
         r'            # 纯视觉试次：无风，但通过 screen_side 控制呈现屏幕\n            # (Pure visual trial: no wind, but control presentation screen via screen_side)\n            trial_dict[\'wind_dir\'] = \'none\''),

        (r'            # baseline_wind 或 looming_wind：wind_dir 同时控制风向和屏幕路由\n            trial_dict\[\'wind_dir\'\] = direction',
         r'            # baseline_wind 或 looming_wind：wind_dir 同时控制风向和屏幕路由\n            # (baseline_wind or looming_wind: wind_dir controls both wind direction and screen routing)\n            trial_dict[\'wind_dir\'] = direction'),

        (r'    # 完全打乱试次顺序',
         r'    # 完全打乱试次顺序\n    # (Completely shuffle the trial order)'),

        (r'    === 重构说明 ===\n    - 新增 self\.pattern_key 属性：标识当前选定范式。\n    - _render_static_baseline\(\) 统一渲染 2° 黑点基线（所有 9 种范式）。\n    - generate_trial_matrix\(\) 现在接收 pattern_key 参数。',
         r'    === 重构说明 ===\n    (Refactoring Notes:)\n    - 新增 self.pattern_key 属性：标识当前选定范式。\n      (Added self.pattern_key attribute: identifies currently selected pattern.)\n    - _render_static_baseline() 统一渲染 2° 黑点基线（所有 9 种范式）。\n      (_render_static_baseline() uniformly renders 2° black dot baseline (all 9 patterns).)\n    - generate_trial_matrix() 现在接收 pattern_key 参数。\n      (generate_trial_matrix() now receives pattern_key parameter.)'),

        (r'        # ==================================================================\n        # 【重构核心】解析 GUI 选中的范式\n        # ------------------------------------------------------------------\n        # pattern_key: 范式键名，用于传递给 generate_trial_matrix\(\)\n        # 所有 9 种范式共享统一的 2° 黑点稳态基线（含 Baseline Wind）。\n        # pattern\.md: "Begin Degree: 2°（as baseline on the same time）"\n        # 2° 黑点作为全局视觉基线，用于后续数据归一化。\n        # ==================================================================',
         r'        # ==================================================================\n        # 【重构核心】解析 GUI 选中的范式\n        # ([Core Refactoring] Parse selected pattern from GUI)\n        # ------------------------------------------------------------------\n        # pattern_key: 范式键名，用于传递给 generate_trial_matrix()\n        # (pattern_key: Pattern key name, used to pass to generate_trial_matrix())\n        # 所有 9 种范式共享统一的 2° 黑点稳态基线（含 Baseline Wind）。\n        # (All 9 patterns share a unified 2° black dot steady-state baseline (including Baseline Wind).)\n        # pattern.md: "Begin Degree: 2°（as baseline on the same time）"\n        # 2° 黑点作为全局视觉基线，用于后续数据归一化。\n        # (2° black dot serves as global visual baseline, used for subsequent data normalization.)\n        # =================================================================='),

        (r'        print\(f"\[LoomingEngine\] 基线策略: 灰底 \+ 2° 黑点 \(通用基线，用于归一化\)"\)\n\n        # 使用选定范式生成试次矩阵\n        self\.trials = generate_trial_matrix\(self\.pattern_key\)',
         r'        print(f"[LoomingEngine] 基线策略: 灰底 + 2° 黑点 (通用基线，用于归一化)")\n\n        # 使用选定范式生成试次矩阵\n        # (Generate trial matrix using the selected pattern)\n        self.trials = generate_trial_matrix(self.pattern_key)'),

        (r'        # 【重构】此阶段已根据范式类型渲染稳态基线：\n        #   - baseline_wind → 纯灰屏幕\n        #   - 其他范式     → 灰底 \+ 2° 黑点（Anti-Startle）\n        # ==============================================================',
         r'        # 【重构】此阶段已根据范式类型渲染稳态基线：\n        # ([Refactoring] Steady-state baseline is rendered according to pattern type in this stage:)\n        #   - baseline_wind → 纯灰屏幕 (Pure gray screen)\n        #   - 其他范式     → 灰底 + 2° 黑点（Anti-Startle） (Other patterns -> Gray background + 2° black dot (Anti-Startle))\n        # =============================================================='),

        (r'                # 【重构】使用选定的单一范式重新生成试次矩阵\n                self\.trials = generate_trial_matrix\(self\.pattern_key\)',
         r'                # 【重构】使用选定的单一范式重新生成试次矩阵\n                # ([Refactoring] Regenerate trial matrix using the selected single pattern)\n                self.trials = generate_trial_matrix(self.pattern_key)'),

        (r'        """\n        【重构】渲染通用稳态基线 — 灰底 \+ 2° 黑点。\n\n        所有 9 种范式（含 Baseline Wind）共享同一基线状态：\n        灰色背景 \+ 屏幕中央 2° 静态黑点。\n\n        设计依据 \(pattern\.md\):\n          "Begin Degree: 2°（as baseline on the same time）"\n        2° 黑点同时满足两个核心需求：\n          1\. 归一化基线：为后续行为数据分析提供统一的视觉参考。\n          2\. Anti-Startle：消除视觉刺激突然出现导致的惊跳反射，\n             确保 Looming 从 2° 无缝膨胀（对 Baseline Wind 则\n             保持 2° 不变，仅触发风刺激）。\n\n        控制面板 \(win_control\) 上同时在左右镜像位置绘制对应的 2° 小圆，\n        为实验者提供实时视觉反馈。\n        """',
         r'        """\n        【重构】渲染通用稳态基线 — 灰底 + 2° 黑点。\n        ([Refactoring] Render universal steady-state baseline — Gray background + 2° black dot.)\n\n        所有 9 种范式（含 Baseline Wind）共享同一基线状态：\n        灰色背景 + 屏幕中央 2° 静态黑点。\n        (All 9 patterns (including Baseline Wind) share same baseline state:\n        Gray background + 2° static black dot in screen center.)\n\n        设计依据 (pattern.md):\n        (Design Basis (pattern.md):)\n          "Begin Degree: 2°（as baseline on the same time）"\n        2° 黑点同时满足两个核心需求：\n        (The 2° black dot satisfies two core requirements:)\n          1. 归一化基线：为后续行为数据分析提供统一的视觉参考。\n             (Normalization baseline: Provides unified visual reference for subsequent behavioral data analysis.)\n          2. Anti-Startle：消除视觉刺激突然出现导致的惊跳反射，\n             确保 Looming 从 2° 无缝膨胀（对 Baseline Wind 则\n             保持 2° 不变，仅触发风刺激）。\n             (Anti-Startle: Eliminates startle reflex caused by sudden appearance of visual stimulus,\n             ensuring Looming expands seamlessly from 2° (For Baseline Wind,\n             keeps 2° unchanged, only triggering wind stimulus).)\n\n        控制面板 (win_control) 上同时在左右镜像位置绘制对应的 2° 小圆，\n        为实验者提供实时视觉反馈。\n        (Corresponding 2° small circles are simultaneously drawn at left/right mirror positions\n        on control panel (win_control), providing real-time visual feedback for experimenter.)\n        """'),

        (r'        # 重置控制面板镜像刺激的半径为初始 2° 基线值\n        # （Looming 试次结束后半径可能被放大到 180°）\n        self\.stim_ctrl_left\.radius = self\.initial_angle_deg / 2\.0',
         r'        # 重置控制面板镜像刺激的半径为初始 2° 基线值\n        # (Reset control panel mirror stimulus radius to initial 2° baseline value)\n        # （Looming 试次结束后半径可能被放大到 180°）\n        # (Radius might be enlarged to 180° after Looming trial ends)\n        self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0'),

        (r'        # 控制面板：绘制两个 2° 小圆 \+ 标签\n        self\.stim_ctrl_left\.draw\(\)',
         r'        # 控制面板：绘制两个 2° 小圆 + 标签\n        # (Control panel: Draw two 2° small circles + labels)\n        self.stim_ctrl_left.draw()'),

        (r'        # 物理屏幕：绘制中央 2° 黑点 \(production only\)\n        if not self\.debug_mode:',
         r'        # 物理屏幕：绘制中央 2° 黑点 (production only)\n        # (Physical screen: Draw central 2° black dot (production only))\n        if not self.debug_mode:'),

        (r'        """\n        ITI 期间渲染通用稳态基线（灰底 \+ 2° 黑点）。\n\n        所有范式均使用统一的 2° 黑点基线。\n        """',
         r'        """\n        ITI 期间渲染通用稳态基线（灰底 + 2° 黑点）。\n        (Render universal steady-state baseline (Gray background + 2° black dot) during ITI.)\n\n        所有范式均使用统一的 2° 黑点基线。\n        (All patterns use unified 2° black dot baseline.)\n        """'),

        (r'            pattern=self\.pattern_key,  # 【新增】记录当前范式\n            \*\*\{k: v for k, v in trial\.items\(\)',
         r'            pattern=self.pattern_key,  # 【新增】记录当前范式 ([New] Log current pattern)\n            **{k: v for k, v in trial.items()'),

        (r'        # 【重构说明】Looming 起始时，黑点已经以 2° 大小停留在屏幕上\n        # （由 _render_static_baseline 持续渲染）。Looming 开始后，\n        # 引擎从 initial_angle_deg=2° 开始逐帧膨胀，实现无缝衔接。\n        # ============================================================',
         r'        # 【重构说明】Looming 起始时，黑点已经以 2° 大小停留在屏幕上\n        # ([Refactoring Notes] At start of Looming, black dot is already resting on screen at 2° size)\n        # （由 _render_static_baseline 持续渲染）。Looming 开始后，\n        # (continuously rendered by _render_static_baseline. After Looming starts,)\n        # 引擎从 initial_angle_deg=2° 开始逐帧膨胀，实现无缝衔接。\n        # (Engine expands frame-by-frame from initial_angle_deg=2°, achieving seamless transition.)\n        # ============================================================'),

        (r'            # 【修正】确定非活动侧刺激对象，用于 Looming 期间维持 2° 基线\n            if side_label == \'left\':',
         r'            # 【修正】确定非活动侧刺激对象，用于 Looming 期间维持 2° 基线\n            # ([Fix] Determine inactive side stimulus object, used to maintain 2° baseline during Looming)\n            if side_label == \'left\':'),

        (r'            # 非活动物理窗口设为非阻塞，避免双重 VSync 卡顿\n            if _inactive_phys_win is not None:',
         r'            # 非活动物理窗口设为非阻塞，避免双重 VSync 卡顿\n            # (Set inactive physical window to non-blocking, avoiding double VSync stutter)\n            if _inactive_phys_win is not None:'),

        (r'                    # 【修正】刷新非活动侧物理屏幕 2° 基线（非阻塞 flip）\n                    if _inactive_phys_stim is not None and _inactive_phys_win is not None:',
         r'                    # 【修正】刷新非活动侧物理屏幕 2° 基线（非阻塞 flip）\n                    # ([Fix] Refresh inactive side physical screen 2° baseline (non-blocking flip))\n                    if _inactive_phys_stim is not None and _inactive_phys_win is not None:'),

        (r'            # 恢复非活动物理窗口的 VSync 阻塞设置\n            if _inactive_phys_win is not None:',
         r'            # 恢复非活动物理窗口的 VSync 阻塞设置\n            # (Restore VSync blocking setting for inactive physical window)\n            if _inactive_phys_win is not None:'),

        (r'        # BASELINE WIND  \(pure air-pump control — 视觉保持 2° 基线\)\n        # 屏幕持续显示 2° 黑点（通用基线），不做任何视觉变化。\n        # 仅在随机延迟后触发风刺激。2° 黑点确保归一化一致性。\n        # Control panel is flipped every frame to stay alive\.',
         r'        # BASELINE WIND  (pure air-pump control — 视觉保持 2° 基线)\n        # (pure air-pump control — Vision maintains 2° baseline)\n        # 屏幕持续显示 2° 黑点（通用基线），不做任何视觉变化。\n        # (Screen continuously displays 2° black dot (universal baseline), without visual changes.)\n        # 仅在随机延迟后触发风刺激。2° 黑点确保归一化一致性。\n        # (Only triggers wind stimulus after random delay. 2° black dot ensures normalization consistency.)\n        # Control panel is flipped every frame to stay alive.'),

        (r'                # 【修正】Baseline Wind 试次内也渲染 2° 黑点基线\n                # 保持与所有其他范式一致的视觉基线，用于归一化\n                self\.stim_ctrl_left\.radius = self\.initial_angle_deg / 2\.0',
         r'                # 【修正】Baseline Wind 试次内也渲染 2° 黑点基线\n                # ([Fix] Baseline Wind trial internally also renders 2° black dot baseline)\n                # 保持与所有其他范式一致的视觉基线，用于归一化\n                # (Maintains consistent visual baseline with all other patterns, used for normalization)\n                self.stim_ctrl_left.radius = self.initial_angle_deg / 2.0'),

        (r'        # POST-TRIAL CLEANUP: force all windows to static baseline\.\n        # 所有范式统一恢复 2° 黑点基线\n        # ============================================================',
         r'        # POST-TRIAL CLEANUP: force all windows to static baseline.\n        # 所有范式统一恢复 2° 黑点基线\n        # (All patterns uniformly restore 2° black dot baseline)\n        # ============================================================'),

        (r'    === 重构说明 ===\n    新增 Experiment Pattern 下拉选择框，包含 9 种实验范式：\n    实验者在 GUI 中为当前被试选定一种范式后，整个 Session 仅运行该范式的 18 个试次。\n    这实现了"单范式锁定"机制 \(Single-Pattern per Subject\)。',
         r'    === 重构说明 ===\n    (Refactoring Notes:)\n    新增 Experiment Pattern 下拉选择框，包含 9 种实验范式：\n    (Added Experiment Pattern dropdown selection box, containing 9 experimental patterns:)\n    实验者在 GUI 中为当前被试选定一种范式后，整个 Session 仅运行该范式的 18 个试次。\n    (After the experimenter selects one pattern for current subject in GUI, the whole Session only runs 18 trials of that pattern.)\n    这实现了"单范式锁定"机制 (Single-Pattern per Subject)。\n    (This implements "Single-Pattern Lock" mechanism (Single-Pattern per Subject).)'),

        (r'    # ==================================================================\n    # 【重构核心】Experiment Pattern 下拉选择框\n    # ------------------------------------------------------------------\n    # 使用 choices= 参数创建下拉菜单（PsychoPy Dlg 原生支持）。\n    # 下拉列表包含 pattern\.md 中定义的全部 9 种范式：\n    #   1\. Baseline Visual \(仅视觉，无风\)\n    #   2\. Baseline Wind \(仅风，随机延迟\)\n    #   3-8\. Looming \+ Wind \(6 种负 TTC / 对应角度\)\n    #   9\. Looming \+ Wind \(TTC \+200ms\)\n    # 默认选中第一项 \(Baseline Visual\)，实验者可从下拉菜单中选择。\n    # ==================================================================',
         r'    # ==================================================================\n    # 【重构核心】Experiment Pattern 下拉选择框\n    # ([Core Refactoring] Experiment Pattern dropdown selection box)\n    # ------------------------------------------------------------------\n    # 使用 choices= 参数创建下拉菜单（PsychoPy Dlg 原生支持）。\n    # (Uses choices= parameter to create dropdown menu (natively supported by PsychoPy Dlg).)\n    # 下拉列表包含 pattern.md 中定义的全部 9 种范式：\n    # (Dropdown list contains all 9 patterns defined in pattern.md:)\n    #   1. Baseline Visual (仅视觉，无风) (Pure visual, no wind)\n    #   2. Baseline Wind (仅风，随机延迟) (Pure wind, random delay)\n    #   3-8. Looming + Wind (6 种负 TTC / 对应角度) (6 negative TTCs / corresponding angles)\n    #   9. Looming + Wind (TTC +200ms)\n    # 默认选中第一项 (Baseline Visual)，实验者可从下拉菜单中选择。\n    # (Defaults to selecting first item (Baseline Visual), experimenter can select from dropdown menu.)\n    # =================================================================='),

        (r'tip=\'选择当前被试的实验范式。整个 Session 将仅运行所选范式的 18 个试次。\'\)',
         r'tip=\'选择当前被试的实验范式。整个 Session 将仅运行所选范式的 18 个试次。 \'\n                     \'(Select the experimental pattern for the current subject. The entire Session will only run 18 trials of the selected pattern.)\')')
    ]

    for pat, repl in replacements:
        content, num_subs = re.subn(pat, repl, content)
        print(f"Matched and replaced {num_subs} times for pattern starts with {pat[:30]}")

    with open(r'c:\Users\Wray\Desktop\Projects\Cricket-Gym\looming\stimulus_controller.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    translate()

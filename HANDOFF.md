# 塔罗斯助手接续说明

更新日期：2026-10-21。GitHub 同步的是提交的项目文件，不会同步本次对话；本文件用于向新任务或云端任务交代上下文。接续时先读本文件、`README.md` 和 `data/rules_reference.json`。

## 当前状态与必须保留的约束

- 当前是 Windows 简体中文桌面**校验版**，使用 Python 3.12 x64、PySide6、本地 OCR；提供画面读取、手动校正、双策略、保存与回放，不自动点击游戏、不上传截图或局面。
- **平均收益**最大化最终绮彩数量期望，相同收益优先较少操作；**冲顶级**寻找至少有指定概率达到的最高绮彩数量，默认最低达成率 20%，再比较达成率及平均数量。切换策略应保留棋盘并使旧任务结果失效。
- 未知概率不能当作等概率，不能编造掉率或终局数值。未知效果、隐藏概率、未来巡监和未展示奖励必须显式保留为未知；规则与分支不完整时，不能把启发分数、截断搜索或合成模型结果宣称为真实终局收益、完整分布或全局最优。
- 尚未取得用户真实游戏截图及完整对局；真实图标样本、校准、动画/遮挡/缩放适配、规则与概率来源核验、连续对局回放仍待完成。不能声称覆盖完整赛季卡池或已完成断网实机整局验证。
- 识别器目前不能读取特殊标记，须逐格核对；当前巡监候选也需填写实际选项。概率表缺少可靠统计时保持为空，不以示例概率代替游戏数据。

## 关键路径

- `run.py`、`talos_assistant/app.py`：入口与界面；`domain.py`、`engine.py`、`solver.py`：状态、规则转移与策略搜索。
- `talos_assistant/capture.py`、`vision.py`：窗口采集、OCR、图标识别及画面稳定判定；`storage.py`：会话、设置与过期结果过滤。
- `data/rules_reference.json`：已支持规则、来源和未实现范围；`tests/`：六个测试模块；`benchmarks/solver_benchmark.py`：合成场景基准。
- `tools/fetch_models.py`、`models/provenance.json`：模型下载与来源；`THIRD_PARTY.md`、`third_party_licenses/`：第三方声明。
- `TalosAssistant.spec`、`tools/package_release.py`：Windows 构建与便携包；`user_data/`：本地用户数据，不应提交或装入发布包。

## 已有验证证据

- `artifacts/selfcheck-release.json` 记录 `passed=true`、`network_attempts=0`，模型齐全、CPU OCR 通过，策略切换保留状态。它使用合成中文图片与离屏界面，关闭真实窗口采集，并阻止 Python socket 连接；这不等同于操作系统断网或真实游戏整局验证。
- 本地 pytest 缓存记录 96 个用例。缓存不是测试通过日志；接续者应运行下方命令确认当前代码，不能据此直接声称 96 项全部通过。
- `artifacts/solver-benchmark.json` 明确使用合成概率模型，包含精确、估计及条件性结果；不能作为真实掉率或所有局面求解完整性的证据。
- 现有干净 ZIP：`dist/TalosAssistant-Windows-x64.zip`，校验文件：`dist/TalosAssistant-Windows-x64.sha256`；解压副本：`dist/portable/TalosAssistant/`。本次只读核对 ZIP 目录共 354 项，包含可执行文件，未含 `user_data`、`icuuc.dll`、`icudt78.dll` 或 `icuuc.dll.unused`。
- `artifacts/`、模型二进制及 `dist/` 被 `.gitignore` 排除；新克隆源码不会自动获得这些本地验证产物。

## Windows 开发、验证与打包

`requirements-lock.txt` 已记录 Windows 环境的精确依赖版本。它包含 Windows 相关包，不能盲目假定在 Linux 上可安装或运行。以下命令在项目根目录执行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e ".[dev]" -c requirements-lock.txt
.venv\Scripts\python.exe tools\fetch_models.py
.venv\Scripts\python.exe run.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe benchmarks\solver_benchmark.py
.venv\Scripts\python.exe tools\smoke_app.py
```

模型只在开发/打包时下载，应用运行时不自动下载。三个 ONNX 文件固定模型版本并验证预设 SHA256；`keys.txt` 目前从上游分支下载，没有预设 SHA256，尚不是完全固定的构建输入。

```powershell
.venv\Scripts\python.exe -m PyInstaller --noconfirm TalosAssistant.spec
dist\TalosAssistant\TalosAssistant.exe --self-test artifacts\selfcheck-packaged.json
.venv\Scripts\python.exe tools\package_release.py
```

必须保留 `.spec` 中排除 `icuuc.dll` 与 `icudt78.dll` 的修复：构建机 PATH 中的 Poppler ICU 可能被误收集，与 Qt 所需 Windows 系统 ICU 导出符号不兼容，导致程序无法启动。发布脚本也排除这些冲突 DLL 及用户数据，并要求打包自检通过且报告不早于 EXE。

当前 `dist/portable/` 已存在，发布脚本会拒绝再次向该目录解压。重新发布前应明确安排新的验证目录或保留旧副本后调整目录，不要直接照抄命令覆盖验证材料。便携版启动日志位于其 `user_data/startup.log`。

## 云端与本机的工作边界

云端可继续代码审查、规则建模、纯逻辑修复及测试；在兼容的 Python 环境安装 pytest 后，可先在源码根目录执行：

```bash
python -m pytest -q tests/test_domain.py tests/test_engine.py tests/test_solver.py tests/test_storage.py
python benchmarks/solver_benchmark.py
```

上述是云端优先验证范围，本次未在 Linux 实测。图像、OCR 和离屏 Qt 测试还需另行配置依赖、模型、字体及平台插件。真实 Windows 游戏窗口枚举与采集、显示缩放和交互、Windows EXE 打包与启动、完整对局识别准确率必须回到 Windows 实机验证。

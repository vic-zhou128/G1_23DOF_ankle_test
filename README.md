# G1 23DoF 脚踝 SDK 反馈语义测试

本仓库测试 G1 23DoF 在 `mode_machine=4`、`mode_pr=0` 时，SDK 脚踝槽位的命令与反馈。程序固定使用 PR 模式，不包含 AB 切换、RL、接触网络、InEKF 实机接管或 DCM 控制。现有数据支持按 PR 槽位使用接口，但**没有独立测量脚相对小腿的姿态，尚不能最终证明反馈一定是虚拟 PR 角，而非固件转换后的其他坐标**。

| SDK 槽位 | PR 名称 | AB 别名 |
| --- | --- | --- |
| 4 | 左踝 Pitch | B |
| 5 | 左踝 Roll | A |
| 10 | 右踝 Pitch | B |
| 11 | 右踝 Roll | A |

槽位、模式及低层命令格式参考[宇树 C++ 脚踝例程](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp)与[Python 底层例程](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)。

## 文件说明

| 路径 | 用途 |
| --- | --- |
| `ankle_test/g1_ankle_pr_test.py` | 实机只读预检、原位接管、顺序脚踝回零、单轴测试与退出。默认只读。 |
| `ankle_test/core.py` | 槽位、五次平滑轨迹、URDF 限位和速度告警。 |
| `ankle_test/hardware_io.py` | DDS、CRC、MotionSwitcher、LowState/LowCmd 原始记录。 |
| `ankle_test/analyze_run.py` | 离线统计 q/dq、跟踪、耦合和重复性，绘图。 |
| `ankle_test/test_core.py`、`test_entry.py`、`test_analysis.py` | 离线单元测试，不发送 LowCmd；`test_entry.py` 需要 SDK。 |
| `models/g1_description/g1_23dof_rev_1_0.urdf` | 本轮使用的官方 23DoF URDF；只解析关节限位，不需要 mesh 文件。 |
| `reports/ankle_pr_qualitative_08/` | 本轮四轴实机测试的精简报告和图。 |
| `ankle_test/run_inekf_sensitivity.py`、`test_sensitivity.py`、`reports/inekf_sensitivity_20261123/` | 后续 InEKF 离线敏感性代码与精简结果；脚本依赖原论文工程，单独克隆本仓库不能重跑。 |

URDF 与[宇树原文件](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_23dof_rev_1_0.urdf)逐字节相同，SHA-256：`7a16cc8fc1cd053ba8ee7b03057585959336c9aee2bad720bb4d2fa980034efb`。其 BSD-3-Clause 声明见 `THIRD_PARTY_LICENSES/unitree_ros_LICENSE`。

## 安装和只读预检

在 Ubuntu 主机上安装[宇树 Python SDK2](https://github.com/unitreerobotics/unitree_sdk2_python)及其 CycloneDDS 依赖，安装步骤以官方 README 为准。`eno1` 换成连接机器人的网卡。先在离线环境验证代码：

```bash
git clone https://github.com/vic-zhou128/G1_23DOF_ankle_test.git
cd G1_23DOF_ankle_test
python -m pip install numpy matplotlib pyyaml
python -m unittest -q ankle_test.test_core ankle_test.test_analysis
python -m unittest -q ankle_test.test_entry  # 需要已安装宇树 SDK2
python ankle_test/g1_ankle_pr_test.py --help
```

只读预检不释放官方服务，也不建立 LowCmd publisher：

```bash
python ankle_test/g1_ankle_pr_test.py \
  --network eno1 \
  --run-name ankle_readonly_01 \
  --urdf models/g1_description/g1_23dof_rev_1_0.urdf \
  --output-dir outputs/hardware/ankle_test \
  --observe-duration 60
```

控制台可输入 `status`、`mark video_sync`、`exit`。检查预检通过，反馈为 `mode_machine=4`、`mode_pr=0`，状态稳定，且没有其他用户程序同时发布 `rt/lowcmd`。每次运行换一个新的 `--run-name`。

## 吊装单轴测试

**以下命令会接管真实机器人。** 机器人须可靠吊装、双脚离地，周围和吊带运动范围无干涉；一人观察并掌握遥控器 `LT+B`。启动前官方 `ai` 服务须持有机器人。看到 `zero LowCmd verified` 后才能继续；若显示 `no official MotionSwitcher service owns the robot`，先按本节末尾方法恢复官方服务，不要绕过检查。

```bash
python ankle_test/g1_ankle_pr_test.py \
  --network eno1 \
  --run-name ankle_pr_new_01 \
  --urdf models/g1_description/g1_23dof_rev_1_0.urdf \
  --output-dir outputs/hardware/ankle_test \
  --enable-lowcmd \
  --gain-ramp-duration 3 \
  --zero-min-duration 8 \
  --zero-max-target-speed-deg-s 3 \
  --zero-speed-tolerance-deg-s 5 \
  --pitch-amplitude-deg 5 \
  --roll-amplitude-deg 3 \
  --ramp-duration 3 \
  --hold-duration 2 \
  --cycles 2
```

在**程序控制台**中，每条命令单独输入一行。先接管并保持捕获姿态：

```text
status
mark before_arm
arm ARM PR HOLD
```

看到 `[hold] captured posture ready` 并检查吊带、各关节后，输入 `prepare ANKLE ZERO`。程序仅按 4→5→10→11 顺序将四个踝槽位移至 SDK `q=0`；其余有效关节保持捕获角度，整机不会回零。看到 `[zero] ankle posture reached`、目视确认四踝后，输入：

```text
mark zero_pose_video
confirm ZERO HOLD
run left-pitch
run left-roll
run right-pitch
run right-roll
status
exit
```

等一轴完成并回到 `zero_hold`，再输入下一条 `run`。每轴两轮“正方向→零点→负方向→零点”。`mark <标签>` 可对齐视频；`observe <轴> pitch|roll|mixed|unclear` 只记录真实观察，不驱动机器人。`return` 请求平滑回零；`stop`、`exit` 和遥控器 `LT+B` 进入退出流程。反馈速度超过默认 0.25 rad/s 只记诊断告警，**不是厂家安全上限，也不自动停机**；温度、故障码、CRC、状态超时、限位和跟踪误差仍受检查。现场急停始终由操作员负责。

退出后程序不自动恢复官方高层服务。运行官方 SDK 仓库的恢复例程：

```bash
python /path/to/unitree_sdk2_python/example/motionSwitcher/motion_switcher_example.py eno1
```

按 Enter，核对输出 `ret: (0, None)`。这不是脚踝运动命令。

## 原始日志与离线分析

每轮写入 `outputs/hardware/ankle_test/<run-name>/`：元数据、事件、摘要和分块 NPZ。日志记录 LowState 的 `mode_pr/mode_machine`、`q/dq/ddq/tau_est/motorstate/temperature`，以及 LowCmd 的 `q/dq/tau/kp/kd`、CRC 和时间戳。每个电机的 `temperature` 有两个通道。

```bash
python ankle_test/analyze_run.py \
  outputs/hardware/ankle_test/ankle_pr_new_01 \
  --derivative-window-ms 20 \
  --output-dir reports/ankle_pr_new_01
```

`--output-dir` 必须尚不存在。分析会生成 `report.json`、逐样本 `samples.csv` 和四槽位曲线图；不传输出目录则只打印 JSON。只有精简报告时，无法重新计算逐样本统计。

## 本次结果解读

`reports/ankle_pr_qualitative_08/` 是四轴成功执行的一轮吊装测试：

可直接打开[四轴反馈图](reports/ankle_pr_qualitative_08/four_axis_response.pdf)和[机器可读报告](reports/ankle_pr_qualitative_08/report.json)。

- `report.json`：每轴目标与反馈幅度、同侧另一槽位的变化、端点误差、q/dq 一致性；其结论为 `verdict=not_determined`。
- `audit_details.json`：原始日志质量审计，包括约 100.6 万帧 LowState、40.8 万帧本程序命令、模式、CRC、故障码和温度。
- `four_axis_response.pdf/png`：四轴总览；`slot_4/5/10/11.png`：各槽位曲线。图中的角度为 SDK 数值，不是外部测得的脚相对小腿角度。

| 轴 | 命令峰峰值 | 反馈峰峰值 | 同侧另一槽位峰峰值 |
| --- | ---: | ---: | ---: |
| 左 Pitch（4） | 10.00° | 5.51° | 1.32° |
| 左 Roll（5） | 6.00° | 2.96° | 2.33° |
| 右 Pitch（10） | 10.00° | 2.77° | 1.50° |
| 右 Roll（11） | 6.00° | 0.93° | 0.37° |

四轴命令均有记录；右 Roll 已发送，但反馈幅度较小。审计中模式始终为 `4/0`，LowState CRC 错误、LowCmd 发送失败和活动电机故障样本均为 0。日志没有同步的操作员观察记录，因此不能定量对照脚底姿态。`dq` 与 `q` 的窗口导数总体一致，但不是独立速度标定；四踝 `ddq` 未提供可用的非零信息。吊装 `tau_est` 不能证明承重力矩或接触力精度。

最终确认 PR/AB 反馈坐标，需要同步测量**脚相对小腿**姿态，并把 SDK 实际反馈 `q` 代入 URDF 后与测量值对比；优先复核右 Roll 的方向和角度分辨率。仅比较 LowCmd 与 LowState 不足以完成判定。

`reports/inekf_sensitivity_20261123/` 为后续固定轨迹的 InEKF 离线输入敏感性分析。`manifest.json` 记录情形与输入身份，`hardware_profile.json` 记录吊装数据统计，`summary.json/csv` 给出 64 次回放指标，`report.txt` 和 `sensitivity.pdf/png` 为阅读版。64/64 次回放完成；例如规则接触、无推扰轨迹的 DCM RMSE：基线 2.23 cm，右 Pitch 反号 10.02 cm，左侧 Pitch/Roll 交换 8.94 cm。这说明错误踝坐标映射可能影响 InEKF，**不能反过来证明实机 PR/AB 语义，也不是独立步态验证**。重跑需原论文工程中的 `g1_estimator.py`、P2 配置和轨迹、Hartley InEKF 二进制等依赖。

## 发布数据范围

本仓库只含代码和精简结果。原始硬件 NPZ 约 2.6 GB，逐样本 `samples.csv` 约 171 MB，InEKF 派生回放约 951 MB，未纳入普通 Git。公开版 JSON 将原主机路径统一写为 `<original-project>/...`；占位符不是可执行路径，数值和输入文件 SHA-256 保持原样。论文复核若需完整数据，应另行归档并记录文件哈希、固件/SDK 版本和视频同步依据。

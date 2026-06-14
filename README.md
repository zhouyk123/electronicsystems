# electronicsystems

电子系统课程项目代码。项目主要面向 Raspberry Pi 小车：通过摄像头识别红、黄、绿色色块，结合 GPIO 电机控制、编码器计数和 PID 闭环，让小车完成搜索色块、靠近色块、绕行色块和最终前进的任务流程。

当前主程序是 `metrics.py`。`metrics_legacy.py` 是旧版实现，`pid_control.py` 是独立的速度 PID 调试脚本，其余 `test_*.py` 文件用于分模块调试或辅助验证。

## 功能概览

- 摄像头采集：使用 OpenCV 从默认摄像头读取 640x480、30 FPS 图像。
- 颜色识别：在 HSV 空间中识别红色、黄色、绿色目标，并根据轮廓面积和质心计算目标水平位置。
- 目标搜索：未发现目标时，小车原地右转搜索，直到检测到指定颜色或达到最大搜索次数。
- 视觉跟随：检测到目标后，根据目标中心相对画面中心的偏差调整左右轮占空比，朝目标前进。
- 编码器计数：使用左右轮编码器脉冲统计移动距离和转向角度。
- 速度闭环：通过 `WheelSpeedPID` 对直行、转弯和绕圈动作做速度控制。
- 动作组合：将基础动作组合成靠近色块、左绕行、右绕行、顺/逆时针绕圈和完整任务流程。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `metrics.py` | 主程序，包含图像处理、GPIO 初始化、电机控制、PID 控制和完整任务流程 |
| `metrics_legacy.py` | 旧版主程序，保留用于对照 |
| `pid_control.py` | 独立的电机速度 PID 调试脚本 |
| `test_visual_center_area.py` | 实时显示颜色 mask、轮廓、中心线和面积 |
| `test_detect_color.py` | 单独测试指定颜色的搜索逻辑 |
| `test_forward_color.py` | 单独测试识别目标后的前进跟随逻辑 |
| `test_motion_actions.py` | 单独测试直行、左转、右转和绕圈等底层动作 |
| `test_one_decomposed_action.py` | 单独执行一个组合动作，默认 dry-run，不触碰真实硬件 |
| `test_decomposed_actions.py` | 组合动作的单元测试；当前部分断言仍按旧接口编写，使用前需要和 `metrics.py` 同步 |
| `test_turn_pid.py` | 转向 PID 实测脚本 |

## 运行环境

建议在 Raspberry Pi 上运行真实硬件流程。

需要的 Python 依赖：

```bash
pip install opencv-python numpy matplotlib
```

Raspberry Pi GPIO 依赖通常使用：

```bash
pip install RPi.GPIO
```

如果是在没有 Raspberry Pi GPIO 的电脑上调试，只能运行带 fake GPIO 的 dry-run 或部分单元测试；真实运动、摄像头跟随和 GPIO 控制必须在树莓派硬件上运行。

## 硬件连接

`metrics.py` 当前使用 BCM 编号，初始化时配置如下：

| 信号 | BCM 引脚 | 作用 |
| --- | ---: | --- |
| `EA` | 13 | 左/右电机 PWM，使能端之一 |
| `I2` | 19 | 电机方向控制 |
| `I1` | 26 | 电机方向控制 |
| `B2A` | 6 | 编码器输入之一 |
| `EB` | 16 | 左/右电机 PWM，使能端之一 |
| `I4` | 20 | 电机方向控制 |
| `I3` | 21 | 电机方向控制 |
| `B1A` | 25 | 编码器输入之一 |

电机左右映射由 `MODE_` 决定，方向由 `MODE_L` 和 `MODE_R` 决定。如果小车前进方向、电机左右或编码器左右不符合预期，优先检查这三个参数以及接线。

## 主程序运行

在 Raspberry Pi 上连接好电机、编码器和摄像头后执行：

```bash
python3 metrics.py
```

程序启动后会执行以下流程：

1. 初始化 GPIO、PWM、编码器中断和速度控制线程。
2. 启动摄像头采集线程。
3. 初始化前进方向修正 PID。
4. 等待摄像头稳定。
5. 执行 `start_job()`。

`start_job()` 当前任务顺序为：

```text
靠近红色 -> 左侧绕过
靠近黄色 -> 逆时针绕圈
靠近绿色 -> 右侧绕过
最终直行
```

颜色顺序由 `COLOR_SEQ = ["red", "yellow", "green"]` 控制。

## 关键参数

常用调参项集中在 `metrics.py` 文件开头。

### 速度和 PID

| 参数 | 说明 |
| --- | --- |
| `SPEED_TARGET` | 直行目标轮速 |
| `TURN_SPEED_TARGET` | 原地转向目标轮速 |
| `SPEED_PID` | 电机速度闭环 PID 参数 |
| `FORWARD_PID` | 视觉跟随时的方向修正 PID 参数 |
| `FORWARD_DUTY` | 视觉跟随前进的基础占空比 |
| `STRAIGHT_DUTY` | 编码器直行动作的基础占空比 |
| `RATIO` | 左右轮直行补偿比例 |
| `ANGLE_FACTOR` | 转角到编码器圈数的换算系数 |

### 颜色识别和距离判断

| 参数 | 说明 |
| --- | --- |
| `low_red1/high_red1`、`low_red2/high_red2` | 红色 HSV 阈值，红色跨 HSV hue 边界所以分两段 |
| `low_yellow/high_yellow` | 黄色 HSV 阈值 |
| `low_green/high_green` | 绿色 HSV 阈值 |
| `LEAST_AREA_FOLLOW` | 视觉跟随的最小有效面积 |
| `LEAST_AREA_FIND_CUBE` | 认为找到目标色块的最小面积 |
| `AREA_FOR_TURN` | 认为已经接近目标、需要停车/绕行的面积 |
| `MAX_TURN_COUNT` | 搜索目标时最多转动次数 |
| `SEARCH_TURN_ANGLE` | 每次搜索转动角度 |

光照、摄像头角度和色卡材质会明显影响 HSV 阈值。正式运行前建议先用 `test_visual_center_area.py` 查看 mask 和面积，再修改阈值。

## 调试方式

### 颜色识别可视化

```bash
python3 test_visual_center_area.py red
python3 test_visual_center_area.py yellow --print-interval 0.2
python3 test_visual_center_area.py green --camera 0
```

窗口中会显示原图叠加轮廓、画面中心线、目标中心线和二值 mask。按 `q` 或 `Esc` 退出。

### 测试颜色搜索

```bash
python3 test_detect_color.py red
python3 test_detect_color.py green --scan-angle 15 --turn-timeout 3
```

该脚本会调用 `metrics.detected_color()`，并给单次转向加超时保护，避免编码器异常时一直阻塞。

### 测试视觉跟随

```bash
python3 test_forward_color.py red --timeout 20
python3 test_forward_color.py yellow --duty 18 --stop-area 70000
python3 test_forward_color.py green --follow-area 1200
```

该脚本会输出目标中心、面积、平均 HSV、PID 修正量和左右轮速度，适合调 `FORWARD_DUTY`、`FORWARD_PID`、面积阈值和 HSV 阈值。

### 测试底层运动

```bash
python3 test_motion_actions.py straight --duty 20 --dist 1 --timeout 5
python3 test_motion_actions.py left --duty 20 --angle 90 --timeout 5
python3 test_motion_actions.py right --duty 20 --angle 90 --timeout 5
python3 test_motion_actions.py circle --v-target 1.5 --dv 0.3 --dist 1 --timeout 5
```

输出会包含目标脉冲数、左右编码器计数、左右轮速度和是否触发结束条件。

### dry-run 组合动作

默认 dry-run 不会触碰 GPIO、电机和摄像头，只打印组合动作会调用的基础动作：

```bash
python3 test_one_decomposed_action.py _move_forward --dist 2
python3 test_one_decomposed_action.py bypass_left
python3 test_one_decomposed_action.py circle_clockwise
```

在 Raspberry Pi 上真实执行时增加 `--real`：

```bash
python3 test_one_decomposed_action.py _move_forward --dist 1 --real
python3 test_one_decomposed_action.py approach_color --color red --real
```

## `metrics.py` 结构

`metrics.py` 可以按下面几层理解：

1. 全局参数：颜色顺序、电机方向、速度、PID、面积阈值、搜索次数和控制周期。
2. 摄像头线程：`picShoot()` 持续读取摄像头并更新全局 `frame`。
3. 编码器和速度线程：`speed_callback()` 统计脉冲，`speedShoot()` 周期性计算左右轮速度并更新 PWM。
4. 图像处理：`getImg_Mask()` 生成颜色 mask，`get_Cube_center_area()` 计算目标中心和面积。
5. 底层动作：`go_straight()`、`turn_left()`、`turn_right()`、`circle()`、`stop()`、`brake()`。
6. 视觉动作：`detected_color()` 搜索目标，`forward_color()` 跟随目标前进，`forward()` 根据中心偏差修正方向。
7. 组合动作：`approach_color()`、`bypass_left()`、`bypass_right()`、`circle_clockwise()`、`circle_anticlockwise()`。
8. 完整任务：`start_job()` 按颜色顺序组合动作。

## 注意事项

- `metrics.py` 在导入时会创建 `cv2.VideoCapture(0)`，部分测试脚本会主动释放或替换摄像头对象。
- 真实运行前请确认小车悬空或处于安全测试区域，尤其是首次调 PID、方向和编码器参数时。
- 如果动作一直不结束，通常优先检查编码器信号、`SPEED_PULSE_PER_REV`、`ANGLE_FACTOR` 和 `threshold` 是否合理。
- 如果目标识别不稳定，优先使用 `test_visual_center_area.py` 调整 HSV 阈值和面积阈值。
- `test_decomposed_actions.py` 当前与 `metrics.py` 的部分组合动作接口不一致，作为旧测试参考使用前需要更新。

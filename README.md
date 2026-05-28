# Optimization_rocket

这是一个基于 CasADi 的火箭轨迹优化与故障分析项目，核心目标是对三段式火箭入轨过程进行建模、求解、回放验证和可达性分析。

项目中既包含标准工况的最优控制求解，也包含故障工况下的容错分析、热力图扫描和结果检查工具。整体结构比较清晰，core 目录里放通用模型和工具，根目录下的多个 .py 文件则是不同任务的入口脚本。

## 目录结构

- biaozhun_opt.py：标准工况优化，可选是否对程序角变化速率、平滑性做限制，进行对比
- fault_opt.py：固定二级时长的故障工况优化。
- fault_opt_T4.py：二级时长可延长的故障工况求解。
- dynamics_replay.py：对已求得的 npz 结果进行高分辨率动力学回放验证。
- kedaxing.py：一级时长可延长的可达性分析，寻找每个故障时刻下可承受的最大推力衰减比例。
- kedaxing_guding.py：一级时长固定的可达性分析。
- relitu.py：故障参数扫描并绘制热力图，展示不同故障条件下额外需要的 T4 补偿时长。
- inspect_npz.py：查看或对比 npz 文件内容的小工具。
- core/：通用物理模型、优化约束和可视化函数。
- results/：各脚本生成的优化结果和图像。

## 各脚本说明

### 标准工况

biaozhun_opt.py 用于求解标准三段式入轨问题。它会读取标准工况的初始猜测文件，建立优化问题，求解后输出优化结果到 results 目录。这个脚本是后续故障分析的基准来源。

### 故障工况

fault_opt.py 用于在给定故障时刻和推力衰减条件下求解入轨问题。脚本支持两种故障模式：一种是推力下降但一级工作时间不变，另一种是推力和秒耗同比下降并延长一级工作时间。

fault_opt_T4.py 与 fault_opt.py 类似，但它把第四段时长 T4 作为优化变量，表示 T4 可延长的故障工况。

### 可达性分析

kedaxing.py 和 kedaxing_guding.py 都是用于找可达性边界的脚本，核心任务是计算在不同故障时刻下系统还能承受多大的推力损失，区别在于故障模式中的一级时长是否可延长。

relitu.py 会扫描故障时刻 te 和推力衰减比例 kappa 的组合，记录每组条件下求得的 T4 及其相对标准工况的额外补偿量 Delta_T4，并绘制热力图。这个脚本既可以先跑优化再保存 relitu.npz，也可以直接读取已有的 relitu.npz 来单独画图。

### 验证与检查

dynamics_replay.py 用于把优化结果拿出来重新积分验证，通常会采用更小的步长对轨迹进行回放，以检查原优化结果在更高分辨率下是否仍然一致。

inspect_npz.py 是一个查看 npz 数据结构的工具。它可以打印文件里的键、数组维度、数据类型和简单统计信息，也支持两个 npz 文件之间的对比，适合用来检查优化结果是否合理。

## 核心模块说明

### core/env_models.py

定义 EarthEnv，集中管理地球环境、发射场参数、目标轨道参数、大气模型、重力模型、坐标变换和轨道六根数计算。它同时支持 NumPy 和 CasADi 两种后端，是整个项目的物理参数中心。

### core/rocket_stage.py

定义 Rocket，封装单级火箭动力学。给定状态、控制和环境对象后，可以直接返回状态导数，供 RK4 离散和最优控制约束使用。

### core/utils.py

提供一些通用工具函数，包括轨迹插值和 RK4 单步积分。优化脚本在构造初猜和离散动力学约束时会频繁用到这里的函数。

### core/ocp_blocks.py

封装优化约束的通用拼装函数，例如控制角上下界、程序角变化率限制、RK4 段约束、质量下界和 IPOPT 求解器配置。这个模块适合把重复的最优控制拼装逻辑抽出来复用。

### core/visual.py

负责把 npz 中保存的优化结果画成一组更完整的可视化图，包括速度、质量、高度、控制角、分量轨迹、经纬高以及轨道图等。

## 典型工作流

1. 先运行 biaozhun_opt.py，生成标准工况初始解。
2. 再运行 fault_opt.py 或 fault_opt_T4.py，分析故障后的可行性和补偿需求。
3. 运行 kedaxing.py & kedaxing_guding & relitu.py 得到可达性边界和补偿规律。
4. 用 dynamics_replay.py 对关键结果做高精度回放验证。
5. 用 inspect_npz.py 快速检查某个 npz 文件的结构和内容。

## 输出文件

项目中的结果默认保存在 results 目录下，常见文件包括：

- biaozhundandao.npz：标准工况初始猜测数据。
- biaozhundandao_unlimited.npz：标准工况优化结果。
- fault_case.npz：故障工况优化结果。
- fault_case_T4.npz：优化第四段时长后的故障结果。
- fault_replay_case.npz：动力学回放验证结果。
- kedaxing.png：可达性分析图片。
- kedaxing_guding.png：可达性分析图片。
- relitu.png：热力图图片。

## 备注

- 这个项目的结果文件是分段保存的，常见字段包括 X1、X3、X4、U1、U3、U4、t1、t3、t4。
- core 里的模块是整个项目共享的基础设施，入口脚本主要负责组合这些模块并设置具体任务参数。
- 如果后续要继续扩展新工况，优先考虑复用 core 中的环境、动力学和工具函数。

## 2026.04.26新增
- 新增new_kedaxing: 在原可达性基础上删掉了目标里的平滑项，仅仅优化kappa，这样得到的才是硬边界，才能和fault_opt里的情况对应起来，这样的可达域会变大一点点。新的npz和图像存为了kedaxing111，可在result中查看，以后写论文的时候，把其他几个也跑一下，尽量避免平滑项的干扰。

## 2026.05.28 新增说明

- `results/biaozhundandao_new.npz` 是在当前标准工况设置下新生成的标准入轨优化结果。它和旧的 `results/biaozhundandao.npz` 使用相同任务条件，但不是逐点完全相同的轨迹；旧文件可作为初始猜测，新文件是重新优化后的另一个可行解。由于该问题是非线性最优控制问题，且末端轨道约束允许有限容差，不同初猜、平滑权重、约束开关或求解器收敛路径都可能得到略有差别但同样满足约束的局部解。
- 新旧标准结果对比时，重点看末端六根数是否满足容差、T4 时长、控制角是否贴边界、轨迹回放误差是否可接受，而不是要求 `X/U` 数组逐点一致。
- 角度周期误差的处理方式已从 `ca.fmod(angle_diff + 180.0, 360.0) - 180.0` 改为 `atan2(sin, cos)` 形式。原来的 `fmod` 写法在负角度跨越 0/360 度边界时可能不能正确得到最短角差，例如 `1 deg - 359 deg = -358 deg` 不一定会被折算成 `+2 deg`。
- 当前统一使用 `EarthEnv.wrap_angle_deg(angle_deg)` 计算角度误差：

```python
@staticmethod
def wrap_angle_deg(angle_deg):
    angle_rad = angle_deg * np.pi / 180.0
    return ca.atan2(ca.sin(angle_rad), ca.cos(angle_rad)) * 180.0 / np.pi
```

终端轨道约束中对应写法为：

```python
err_O = env.wrap_angle_deg(O_fin - O_t)
err_w = env.wrap_angle_deg(w_fin - w_t)
err_f = env.wrap_angle_deg(f_fin - f_t)
```

这样得到的是 `[-180 deg, 180 deg]` 附近的最短有符号角差，更适合处理升交点赤经、近地点幅角和真近点角这类周期角。

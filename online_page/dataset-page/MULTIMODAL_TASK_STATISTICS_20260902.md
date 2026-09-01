# Motion Data Atlas：多模态任务统计

审计日期：2026-09-02  
数据范围：HumanML3D、SONIC、MotionX 当前服务器目录与 Motion Data Atlas 统一索引。

## 1. 统计原则

- `annotation samples`：标注文件或标注记录的唯一样本数。
- `text units`：真正的文本监督单元，例如 caption 行、temporal event；它不等于样本数。
- `paired samples`：该任务要求的模态均存在、可直接用于训练的唯一样本数。
- `motion-backed duration`：可配 motion 的 `sum(frames / fps)`。HumanML3D 使用 20 FPS；SONIC、MotionX 使用 30 FPS。
- 同一个 motion 会进入多个任务，下面各行不能相加成“总训练样本”。

## 2. 任务矩阵

| 任务 | 标注 / 媒体样本 | 文本监督单元 | 可配 motion | 可配 video | 完整任务配对 | 对齐时长 |
|---|---:|---:|---:|---:|---:|---:|
| HumanML3D motion–text（短文本） | 29,232 caption files | 87,384 caption lines；其中 87,372 条可配 | 29,228 | 0 | 29,228 motion–text | 57.1529 h motion-backed |
| SONIC motion–text（时序事件） | 142,220 metadata / temporal samples | 352,703 events | 142,216 | 0 | 142,216 motion–text | 287.4880 h motion-backed |
| MotionX video–motion | 115,990 source videos | — | 64,246 | 64,246 | 64,246 video–motion | 177.1308 h motion-backed |
| MotionX video–text（clip caption） | 64,249 caption records | 64,249 composite descriptions | 64,246 可选配 | 64,249 | 64,249 video–text | 177.1308 h，仅 64,246 条 motion-backed |
| MotionX motion–text（逐帧） | 64,219 frame-caption files | 配对部分覆盖 19,129,928 motion frames | 64,216 | 64,219 | 64,216 motion–text；同时也是完整 video–motion–text | 177.1290 h motion-backed |
| MotionX video–motion–text（Complex CoT） | 49,408 CoT outputs | 49,408 complete step2 generations | 49,405 | 49,408 | 49,405 complete triplets | 136.1625 h motion-backed |

## 3. 分任务说明

### HumanML3D motion–text

- 现存 motion：29,228。
- caption files：29,232；多出的 4 个 annotation-only ID 为 `009707`、`011059`、`M009707`、`M011059`。
- caption 文本总行数：87,384；可与 motion 对齐 87,372 行。
- 镜像口径：caption 层 14,616；现存 motion / 有效配对层 14,614。

### SONIC temporal motion–text

- `142,220` 是 annotation sample 数，不是现存 motion 文件数；现存 motion 为 142,216。
- temporal event 总数：352,703；每个样本中位数 2 个 event。
- `71,088` 是 annotation 层 `is_mirror=True` 的数量；现存可配镜像 motion 为 71,086；镜像 event 为 176,298。
- 时长存在三种合法口径，必须注明：
  - 现存配对 motion 帧时长：287.4880 h。
  - temporal clip span：全标注 284.2698 h；仅配对样本 284.2593 h。
  - event duration 逐事件求和：全标注 284.1225 h；仅配对样本 284.1121 h。

### MotionX video–text

- `descriptions.json`、`overall_action_overview.json`、`key_action_summary.json` 各有 64,249 条，三者 ID 集完全相同。
- 三个文件是同一批 clip 的完整描述、overview 和 key summary 三种文本视图，不能相加为 192,747 个样本。
- 64,249 条 caption 均有视频，因此 video–text pair 是 64,249。
- 其中 64,246 条有当前 NPY motion；177.1308 h 是这 64,246 条的 motion-backed duration，不是 64,249 个 MP4 容器的全量时长。

### MotionX frame motion–text

- frame-caption files：64,219；可配 motion：64,216；三者齐全的 video–motion–frame-text：64,216。
- 3 个 caption-only ID 均有视频但无当前 NPY motion。
- 另有 30 个 motion 没有 frame caption，合计 203 帧。
- MotionX 没有可用的 `is_mirror` 或 `_M` 显式字段；应记为“镜像信息不可判定”，而不是“确认没有镜像”。

### MotionX Complex CoT

- sample list、sample directory 和 `step2_generation.json` 均为 49,408，生成任务本身完整，failed=0。
- 49,408 个视频全部存在；当前 NPY motion 可配 49,405，因此完整 video–motion–text triplet 为 49,405。
- Complex CoT 全部是 64,249 条 clip descriptions 的子集；其中 25 条没有当前 frame-caption 文件。

## 4. 共享缺失 ID

MotionX clip caption、frame caption 与 Complex CoT 共同出现的 3 个“有 video/text、无当前 NPY motion”样本：

- `1119212259`
- `24103020185`
- `24103022586`

## 5. 页面展示约定

页面首要数字显示“标注 / 媒体记录、完整任务配对、motion-backed 对齐时长”；说明文字补充 text units、镜像口径与缺失项。Video–text 的 64,249 个 pair 与 64,246 个 motion-backed duration 样本必须分开表达。

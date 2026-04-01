# nnAudio 迁移说明（中文）

## 1. 文档目的

这份文档用于说明本次 `nnAudio` 迁移/维护工作中：

1. 改了哪些文件
2. 分别解决了什么问题
3. 如何在本地运行验证
4. 正常情况下应该看到什么结果

本说明主要围绕老师明确提出的任务整理，不把额外的工程建议当作必须交付项。

## 2. 本次工作的范围

老师明确提出的任务包括：

1. 基于实验室 fork 的仓库继续维护
2. 更新到兼容当前 Torch / 相关依赖
3. 处理 issue `#132`（TorchScript）
4. 处理 issue `#136`（log-STFT / 非均匀频率轴 inverse）
5. 增加引用论文的提醒

本次已经围绕以上任务完成实现和验证。

## 3. 当前验证环境

本地验证环境如下：

- Python: `3.11.10`
- Torch: `2.10.0+cpu`
- NumPy: `2.4.3`
- SciPy: `1.17.1`
- librosa: `0.11.0`
- nnAudio 包版本: `0.3.4`

说明：

- 这次工作把旧项目拉到了上述现代依赖环境下跑通。
- 当前没有把包版本号改成 `2.0`，也没有改成新的发布版本号；功能上是“开始维护新版”，不是“已经正式发版 2.0”。

## 4. 改了哪些地方，分别解决了什么问题

### 4.1 `Installation/nnAudio/features/stft.py`

这个文件是本次改动最多的地方，主要解决了两个问题：

#### 问题 A：issue #132，TorchScript 失败

原问题：

- `torch.jit.script(STFT(...))` 会失败
- `torch.jit.script(iSTFT(...))` 也会失败

原因包括：

1. `forward()` 里动态写模块属性
2. `forward()` 里动态实例化 padding 模块
3. `iSTFT` 的 `None` 分支写法不利于 TorchScript 编译

对应修改：

1. 去掉了 `forward()` 中的动态属性写入，改为局部变量
2. 把动态 padding 模块改成函数式 padding
3. 给 TorchScript 相关路径补了更明确的参数处理

结果：

- `STFT` 可以被 `torch.jit.script(...)` 编译
- `iSTFT` 可以被 `torch.jit.script(...)` 编译

#### 问题 B：issue #136，非均匀频率轴 inverse 会静默返回坏结果

原问题：

- `freq_scale='linear'` 或 `freq_scale='log'` 时，inverse 会返回明显质量很差的结果
- 但代码不会报错，容易误导使用者

对应修改：

1. 增加了 inverse 支持边界检查
2. 明确规定只有 `freq_scale='no'` 的标准均匀频率轴才支持可靠 inverse
3. 对 `linear` / `log` / `log2` 频率轴，inverse 现在会明确抛出异常
4. 同时在初始化时给出 warning，提示这类配置是 analysis-only

结果：

- `freq_scale='no'`：inverse 仍然正常
- `freq_scale='linear' / 'log' / 'log2'`：不再静默返回坏音频，而是明确拒绝执行 inverse

### 4.2 `Installation/nnAudio/utils.py`

这个文件主要是配合 issue `#132` 的 TorchScript 修复。

对应修改：

1. 给 `torch_window_sumsquare(...)` 和 `overlap_add(...)` 增加了更明确的类型标注
2. 修正了 `fold(..., stride=...)` 的参数形式，使其更适配 TorchScript / 当前 Torch 行为

结果：

- `iSTFT` 所依赖的 helper 函数可以被 TorchScript 正常处理

### 4.3 `Installation/tests/test_stft.py`

这个文件新增了回归测试。

新增测试覆盖：

1. `STFT` 的 TorchScript 编译
2. `iSTFT` 的 TorchScript 编译
3. 非均匀频率轴 inverse 的拒绝行为

结果：

- 这些行为现在不是“手工验证一次”，而是进入测试集，后续修改时更容易防止回归

### 4.4 `Installation/nnAudio/features/cfp.py`

这个文件修复的是现代 SciPy 兼容问题。

原问题：

- `scipy.signal.blackmanharris` 在当前 SciPy 版本下路径不再兼容

对应修改：

- 改为使用 `scipy.signal.windows.blackmanharris`

结果：

- CFP 相关测试恢复通过

### 4.5 `Installation/nnAudio/features/vqt.py`

这个文件修复的是 VQT 与 CQT 的兼容/一致性问题。

原问题：

- `VQT(gamma=0)` 理论上应退化成 CQT，但实际和 `CQT1992v2` 存在明显差异

对应修改：

- 当 `gamma == 0` 时，显式走 `CQT1992v2` 路径

结果：

- `VQT(gamma=0)` 与 CQT 行为对齐
- 相关测试恢复通过

### 4.6 `Installation/nnAudio/__init__.py`

这个文件实现了老师提到的引用提醒。

新增内容：

1. `__citation__`
2. `cite()`
3. `show_citation()`
4. `CitationReminderWarning`
5. import 时的 citation reminder
6. 环境变量 `NNAUDIO_DISABLE_CITATION_REMINDER=1` 用于关闭提醒

结果：

- `import nnAudio` 时会提示引用论文
- 用户也可以程序化读取 citation 信息

### 4.7 `Installation/tests/test_package.py`

这个文件是新增测试文件，用来验证 citation 相关行为。

覆盖内容：

1. citation 文本是否存在且包含 DOI
2. import 时 warning 是否出现
3. suppress 环境变量是否生效

### 4.8 文档补充

为了避免 issue `#136` 的行为继续误导用户，补充更新了：

1. `README.md`
2. `Sphinx/source/intro.rst`
3. `Sphinx/source/index.rst`

这里的目的不是扩 scope，而是把已经修改过的行为边界说明清楚。

## 5. 当前我们解决了什么

到目前为止，已经解决的问题包括：

1. 旧代码在当前 Torch 下的核心兼容性问题
2. issue `#132`：`STFT` / `iSTFT` 无法 TorchScript 编译
3. issue `#136`：非均匀频率轴 inverse 静默返回错误结果
4. 现代 SciPy 下 CFP 初始化失败
5. `VQT(gamma=0)` 与 CQT 行为不一致
6. 缺少引用论文提醒

## 6. 如何运行

### 6.1 激活环境

```bash
cd /junyi/nnAudio
source .venv/bin/activate
```

### 6.2 运行全量测试

```bash
pytest Installation/tests -q
```

### 6.3 仅验证 TorchScript / STFT 相关行为

```bash
pytest Installation/tests/test_stft.py -q
```

### 6.4 验证 citation 提醒

```bash
python - <<'PY'
import nnAudio
print(nnAudio.__version__)
print(nnAudio.__citation__)
PY
```

### 6.5 如果不想看到 citation 提醒

```bash
NNAUDIO_DISABLE_CITATION_REMINDER=1 python - <<'PY'
import nnAudio
print(nnAudio.__version__)
PY
```

## 7. 正常情况下应该看到什么结果

### 7.1 全量测试

运行：

```bash
pytest Installation/tests -q
```

期望结果：

- 当前应看到 `57 passed`
- 允许存在若干 warning
- 不应出现 failed tests

### 7.2 TorchScript

`STFT` / `iSTFT` 现在应能被 `torch.jit.script(...)` 编译，不应再出现原来 issue `#132` 的报错。

### 7.3 标准 inverse

对于：

- `freq_scale='no'`

期望结果：

- inverse 正常
- 重建误差非常小

### 7.4 非均匀频率轴 inverse

对于：

- `freq_scale='linear'`
- `freq_scale='log'`
- `freq_scale='log2'`

期望结果：

- inverse 会明确抛出异常
- 这是预期行为，不是 bug

原因：

- 当前实现下这类 inverse 本来就不可靠
- 明确失败比悄悄返回坏结果更安全

### 7.5 citation 提醒

运行：

```bash
python - <<'PY'
import nnAudio
PY
```

期望结果：

- 会看到一条 `CitationReminderWarning`
- 提醒用户如果使用 nnAudio，请引用论文

## 8. 当前还存在但不阻塞的问题

虽然测试已经全过，但目前还有一些 warning 没有清理：

1. CFP 里有 `divide by zero` warning
2. `torch.stft(return_complex=False)` 有未来弃用 warning
3. `torch.jit.script` 本身在新 Torch 中也有 deprecation warning

这些不是当前老师任务的阻塞项，但如果后续继续维护，可以作为下一阶段清理项。

## 9. 当前结论

按老师明确提出的任务来看：

1. 当前 Torch / 相关依赖兼容：已在本地现代环境中验证通过
2. issue `#132`：已完成
3. issue `#136`：已完成（通过安全失败模式处理）
4. citation 提醒：已完成（通过 import-time reminder 实现）

当前代码状态可以认为已经完成老师本轮提出的核心任务。

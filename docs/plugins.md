# 编写插件 · Writing a plugin

插件就是一个文件夹：里面有你的脚本，以及（可选的）一个描述它的 `stellaria-plugin.json`。
在 **设置 → 插件 → 添加文件夹** 里加入后，混音节点的“处理器”下拉框里就能选它。
A plugin is a folder with your script and (optionally) a `stellaria-plugin.json` describing it.
Add it in **Settings → Plugins → Add folder**; mix plugins then show up as processors on the
canvas's mix node.

> 插件可以运行任意代码。只添加你信任的文件夹。
> Plugins can run arbitrary code. Only add folders you trust.

API details: [`api.md` → Plugins](api.md#plugins).

---

## 中文

### 两种运行方式

| `runtime` | 适合 | 怎么运行 |
|---|---|---|
| `command` | 已有的脚本/程序，不想改代码 | 按 `command` 列表启动（`{python} {folder}/xxx.py {backing} {vocal}` 这类），从工作目录里按 `outputs` 找结果 |
| `python` | 新写的插件 | 入口文件里写 `process(job)`，可以拿到参数、汇报进度、写日志 |

每次运行：

* 默认在一个**全新的临时工作目录**（`"cwd": "temp"`）里运行，脚本创建/清空 `input/`、`output/`、
  `temporary/` 都不会碰到别的东西；运行结束后整个目录会被删除。需要在插件文件夹里运行时写
  `"cwd": "folder"`（此时只认本次运行新写出的结果文件）。
* 输入是 WAV 副本，文件名可读（`<工程> - <声音>.wav`），放在工作目录**之外**。
* 标准输出/错误输出逐行进入任务日志；打印一行 `STELLARIA_PROGRESS 0.5` 就能更新进度（0..1）。
* 退出码非 0 → `plugin.failed`（错误详情是最后 60 行日志）；超时 → `plugin.timeout`；
  正常结束但找不到音频 → `plugin.no_output`。取消任务会结束整个进程树。
* 结果可以是 WAV/FLAC/MP3 等，应用会转成 WAV 收进工程。
* 环境变量：`STELLARIA_PLUGIN_ID`、`STELLARIA_PLUGIN_FOLDER`、`STELLARIA_OUTPUT`、`STELLARIA_OUTDIR`、
  `STELLARIA_PARAMS_JSON`，以及 `STELLARIA_SEED`（混音节点出多个版本时每个版本不同，想做随机变化可以用它）；输出为 UTF-8；没有控制台窗口，也没有标准输入（`input()` 会直接报错退出）。

### `stellaria-plugin.json`

```jsonc
{
  "id": "my-mixer",                       // 小写字母/数字/-/_，最长 64
  "name": "我的混音",
  "version": "1.0",
  "kind": "mix",                          // mix：人声(+伴奏) → 成品；effect：一段音频 → 一段音频
  "runtime": "command",                   // command | python
  "command": ["{python}", "{folder}/mix.py", "{backing}", "{vocal}", "{output}"],
  "entry": null,                          // runtime=python 时：含 process(job) 的文件
  "cwd": "temp",                          // temp（默认）| folder
  "outputs": [],                          // 结果文件的通配符（相对工作目录，第一个匹配的生效）
  "maxVocals": 1,                         // 能同时接几条人声（默认 1）
  "params": [                             // 混音节点上显示的参数（ParamSpec）
    {"id": "wet", "label": "混响量", "type": "number", "default": 0.3, "min": 0, "max": 1, "step": 0.05}
  ],
  "settings": [                           // 设置页里填一次的东西，比如路径
    {"id": "vst3", "label": "VST3 文件夹", "type": "folder", "default": null}
  ],
  "requirements": ["pedalboard", "librosa", "soundfile"],
  "timeoutSec": 1200
}
```

（JSON 里不能写注释，上面的注释只是说明。）

`command` 里可用的占位符：

| 占位符 | 含义 |
|---|---|
| `{python}` | 这个插件用的 Python 解释器 |
| `{folder}` | 插件文件夹 |
| `{backing}` / `{vocal}` / `{vocals}` | 伴奏、第一条人声、全部人声（`{vocals}` 必须单独作为一个参数，展开成多个参数）—— 仅 mix |
| `{input}` | 输入音频 —— 仅 effect |
| `{output}` | 应用指定的结果路径（写到这里就不需要 `outputs`） |
| `{outdir}` | 应用指定的结果文件夹（里面任意音频文件都算结果） |
| `{params_json}` | 一个 JSON 文件：`{params, settings, inputs, output, outdir, plugin}` |
| `{setting:<id>}` | 某个设置的值（没填时插件显示“需要设置”） |
| `{{` / `}}` | 字面的 `{` / `}` |

参数类型 `type`：`number`、`boolean`、`choice`（需要 `options: [{value, label}]`）、`text`、`folder`、`file`。

### `python` 运行方式模板

```python
# entry.py   （manifest: "runtime": "python", "entry": "entry.py"）
import soundfile as sf

def process(job):
    vocal = job.inputs["vocals"][0]          # Path；effect 插件是 job.inputs["input"]
    backing = job.inputs["backing"]          # Path 或 None
    job.log(f"参数: {job.params}  设置: {job.settings}")
    job.progress(0.1)
    ...                                      # 处理
    sf.write(job.output, mix, sr)            # 写到 job.output（WAV/FLAC 均可）
    job.progress(1.0)
    # 也可以把结果写到别处，然后 return 那个路径
```

`print()` 的内容会进入日志。入口文件同目录下的其他模块可以正常 `import`。宿主脚本只用标准库，
需要 Python ≥ 3.8。

### Python 解释器

* **内置（默认）**：应用为每个插件建一个独立环境 `<数据目录>/plugins/envs/<id>`，直接复用应用自带的
  numpy / scipy / librosa / soundfile / torch。缺少的包在插件页点“安装依赖”，用和附加组件相同的镜像安装，
  只装应用里没有的那部分。
* **自定义**：选你自己的 `python.exe`（或 conda/venv 环境文件夹），比如已经装好 pedalboard、
  调好 VST3 的 conda 环境。应用**不会**往你的解释器里装东西，除非你在插件页明确点了“安装到这个解释器”。

### 示例：把朋友的自动混音脚本接进来

假设文件夹里是 `auto_mixing.py`（入口，运行方式 `python auto_mixing.py <伴奏> <人声>`）、`setting.py`
以及其他模块，脚本会在当前目录下创建并清空 `input/`、`output/`、`temporary/`，最后写出
`output/mixdown_*.flac`。

1. 打开 `setting.py`，把 `recieve_mode` 设为 `1`（“从命令行接收路径”模式；`0` 是测试模式，会去读
   `input/` 文件夹，而临时工作目录里它是空的）。
2. 确认 `setting.py` 里的 VST3 插件路径都是**绝对路径**（例如 `C:\\Program Files\\Common Files\\VST3\\…`）。
   脚本在临时目录里运行，相对路径会找不到。
3. `random_mode` 可以保持 `False`。设为 `True` 时会输出多个文件，应用只取第一个。
4. 设置 → 插件 → 添加文件夹，选这个文件夹。因为没有 `stellaria-plugin.json`，会进入向导，按下面填：

   | 向导字段 | 值 |
   |---|---|
   | 名称 `name` | `Sodium 自动混音` |
   | 类型 `kind` | `mix` |
   | 入口 `entry` | `auto_mixing.py` |
   | 参数 `args` | `{backing}`、`{vocal}` |
   | 结果 `outputs` | `output/*.flac`、`output/*.wav` |
   | 工作目录 `cwd` | `temp` |
   | 依赖 `requirements` | `pedalboard`、`librosa`、`soundfile`、`scipy`、`numpy` |

   向导会在应用数据目录里保存一个清单，**不会**改动你的文件夹。等价的 `stellaria-plugin.json`
   （也可以自己放进文件夹里，就不用向导了）：

   ```json
   {
     "id": "sodium-automix", "name": "Sodium 自动混音", "version": "1.0",
     "kind": "mix", "runtime": "command",
     "command": ["{python}", "{folder}/auto_mixing.py", "{backing}", "{vocal}"],
     "cwd": "temp", "outputs": ["output/*.flac", "output/*.wav"], "maxVocals": 1,
     "requirements": ["pedalboard", "librosa", "soundfile", "scipy", "numpy"],
     "timeoutSec": 1200
   }
   ```

5. 解释器：如果朋友的 conda 环境里已经配好了，选“自定义”并指向那个环境的 `python.exe`；
   否则保持“内置”，点“安装依赖”（会装 pedalboard，其余应用自带）。
6. 点“试运行”：应用用自带的 8 秒测试伴奏 + 人声跑一遍，可以直接试听结果；失败时日志里能看到脚本的输出。

注意：传给脚本的路径是系统原生格式（Windows 上是 `\`），所以按 `\\` 切分路径的脚本也能用。
这个脚本按文件名里有没有 “instrument” 区分伴奏和人声（伴奏它会自己改名为 `*_instrument.wav`），
所以工程名、声音名里别带 “instrument”。

### 常见错误

| 代码 | 含义 |
|---|---|
| `plugin.no_manifest` | 文件夹里没有清单 → 用向导 |
| `plugin.bad_manifest` | 清单有问题（`reason` 说明是哪一项） |
| `plugin.python_missing` | 解释器不存在或不可用 |
| `plugin.packages_missing` | 缺少依赖 → 安装依赖，或换一个解释器 |
| `plugin.setting_missing` | 命令里用到的设置还没填 |
| `plugin.timeout` | 超过 `timeoutSec` |
| `plugin.failed` | 退出码非 0 / `process` 抛了异常（详情是最后 60 行日志） |
| `plugin.no_output` | 正常结束，但没找到结果音频（检查 `outputs` 和 `cwd`） |

---

## English

**Runtimes.** `command` wraps an existing script or program unchanged: the app starts the
`command` list with tokens filled in and picks the result up from `{output}`, the `outputs` globs
(relative to the working folder, first match wins) or `{outdir}`. `python` calls
`process(job)` in an entry file and gives you params, settings, progress and logging (template
above: `job.inputs`, `job.output`, `job.outdir`, `job.params`, `job.settings`, `job.progress(x)`,
`job.log(text)`; returning a path uses that file as the result). The host script is stdlib-only
and needs Python ≥ 3.8.

**Each run** gets a fresh temporary working folder (`"cwd": "temp"`, deleted afterwards), so
scripts that create and clear `input/`, `output/`, `temporary/` cannot touch anything else;
`"cwd": "folder"` runs inside the plugin folder and then only files written during that run count.
Inputs are WAV copies with readable names, placed outside the working folder. stdout/stderr go to
the job log line by line; a line `STELLARIA_PROGRESS 0.5` sets progress. Env vars: `STELLARIA_PLUGIN_ID`,
`STELLARIA_PLUGIN_FOLDER`, `STELLARIA_OUTPUT`, `STELLARIA_OUTDIR`, `STELLARIA_PARAMS_JSON` and
`STELLARIA_SEED` (differs per variant when the mix node asks for several). No console window, no
stdin (`input()` fails at once), UTF-8 output, and cancelling kills the whole process tree.
Non-zero exit → `plugin.failed` (detail: last 60 log lines), too slow → `plugin.timeout`, no audio
found → `plugin.no_output`. Results may be WAV/FLAC/MP3/…; the app converts them to WAV.

**Manifest.** See the annotated example and the token table above (`{python}`, `{folder}`,
`{backing}`, `{vocal}`, `{vocals}` (a whole argument, expands to one argument per vocal), `{input}`,
`{output}`, `{outdir}`, `{params_json}`, `{setting:<id>}`, `{{`/`}}`). `params` are shown on the
mix node, `settings` once in Settings (e.g. a VST3 folder). Without a manifest, the wizard
writes one into the app's data folder and never into yours.

**Interpreter.** *Built-in* (default): a per-plugin environment under
`<data>/plugins/envs/<id>` that reuses the app's numpy/scipy/librosa/soundfile/torch; "Install
requirements" installs only what the app does not already have, from the same mirrors as add-ons.
*Custom*: point at your own `python.exe` or environment folder (e.g. a conda env with pedalboard
and your VST3s set up). The app never installs into it unless you explicitly ask.

**Friend's auto-mixing script** (`python auto_mixing.py <backing> <vocal>`, writes
`output/mixdown_*.flac`, imports `setting.py`): set `recieve_mode = 1` in `setting.py` (mode 0 reads
an `input/` folder, which is empty in the temporary working folder), make the VST3 paths in
`setting.py` absolute, keep `random_mode = False` (with `True` only the first output is used), then
add the folder and fill the wizard with: name `Sodium 自动混音`, kind `mix`, entry `auto_mixing.py`,
args `{backing}` `{vocal}`, outputs `output/*.flac` `output/*.wav`, cwd `temp`, requirements
`pedalboard librosa soundfile scipy numpy` (the JSON above is the equivalent manifest). Pick the
conda env that already works as a custom interpreter, or keep the built-in one and install
requirements. "Test run" mixes the bundled 8-second test song so you can listen right away.
Paths are passed in native form (backslashes on Windows), so scripts that split paths on `\\`
work.

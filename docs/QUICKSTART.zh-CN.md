# AudioSeek 本地使用

AudioSeek 可以把英语会议或讲座转成文字，按问题检索相关段落，再播放对应的录音片段。

## 安装与启动

建议使用 Python 3.12，在项目文件夹中打开终端：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m audioseek --open-browser
```

第一次启动需要下载约 240 MB 的模型文件。终端出现网址后，打开 `http://127.0.0.1:7860`。使用期间请保持终端运行。

终端会显示四个准备阶段，下载时显示文件大小和百分比。单个文件失败会自动重试，校验通过的文件会保留。也可以先单独下载模型：

```powershell
.venv\Scripts\python.exe -m audioseek --download-models
```

模型准备好后，该命令会直接退出，再运行正常启动命令即可。若设置了 `--data-dir`，下载和启动时应使用同一个目录。

如果端口被占用，换一个端口：

```powershell
.venv\Scripts\python.exe -m audioseek --port 7861 --open-browser
```

## 操作

1. 上传英语录音，等待转录和索引完成。单个文件须同时满足不超过 120 分钟、512 MiB。
2. 输入英语问题或主题，选择 BM25、SBERT 或 Hybrid，点击 **Find passages**。
3. 阅读相关段落，点击 **Listen to result** 播放。部分结果包含多个可选片段。
4. 可下载文字转录，或用 **Clear upload** 清理当前会话的应用工作文件。

播放器的零点是片段开头；上方显示的时间是原始录音中的位置。结果和时间来自自动识别与检索，仍应通过阅读和听音判断。

录音及派生文件保存在运行应用的电脑中，默认目录为 `~/.cache/audioseek`。清理按钮不删除原始录音和已导出的文件，也不立即删除 Gradio 自己管理的临时副本。停止后重试只支持当前会话；刷新页面或重启应用后不保证恢复。

## 常见问题

- **页面无法连接**：确认终端仍在运行，并已出现启动网址；首次模型下载完成前页面不会就绪。
- **模型下载失败**：确认可以访问 Hugging Face，然后重启应用。不要手动替换为其他版本模型。
- **转录很慢**：CPU 处理长录音需要时间。**Stop processing** 会等当前 15 分钟块结束后停止，之后可在当前会话点击 **Retry processing**。
- **问题无关却出现结果**：应用返回的是排序候选段落，没有自动判定“录音一定包含答案”。请阅读并播放核实。
- **中文录音**：当前版本没有验证中文支持，请使用英语录音和英语查询。


[![License](https://img.shields.io/badge/License-View%20License-blue.svg)](https://github.com/GuijiAI/HeyGem.ai/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![Linux](https://img.shields.io/badge/OS-Linux-brightgreen.svg)

**[中文](#chinese-version)** | **[English](README_en.md)**

---

<a name="chinese-version"></a>

# HeyGem-Linux-Python-Hack

## 项目简介

[HeyGem-Linux-Python-Hack] 是一个基于 Python 的数字人项目，它从 [HeyGem.ai](https://github.com/GuijiAI/HeyGem.ai) 中提取出来，它能够直接在 Linux 系统上运行，摆脱了对 Docker 和 Windows 系统的依赖。我们的目标是提供一个更易于部署和使用的数字人解决方案。

**如果你觉得这个项目对你有帮助，欢迎给我们 Star！**  
**如果运行过程中遇到问题，欢迎提交 Issues！**

## 主要特性

* 无需 Docker: 直接在 Linux 系统上运行，简化部署流程。
* 无需 Windows: 完全基于 Linux 开发和测试。
* Python 驱动: 使用 Python 语言开发，易于理解和扩展。
* 开发者友好: 易于使用和扩展。

## 开始使用

### 安装
本项目支持且仅支持 Linux & python3.8 环境  
请确保你的 Linux 系统上已经安装了 Python 3.8。然后，使用 pip 安装项目依赖项：

```bash
pip install -r requirements.txt
```

### 使用
把项目克隆到本地
```bash
git clone https://github.com/Holasyb918/HeyGem-Linux-Python-Hack
cd HeyGem-Linux-Python-Hack
bash download.sh
```
开始使用  
在开始前，打开 run.py 并修改 audio_url 和 video_url 和 code，请注意，**url 是本地文件，且仅支持相对路径**.
```bash
python run.py 
```

## Contributing  
欢迎贡献！

## License
参考 heyGem.ai 的协议.

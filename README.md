# visual-understanding

![PyPI version](https://img.shields.io/pypi/v/visual-understanding)
![Python versions](https://img.shields.io/pypi/pyversions/visual-understanding)
![License](https://img.shields.io/pypi/l/visual-understanding)

多提供商视觉理解工具——**MCP 服务器 + CLI 双模式**。通过统一的接口调用
智谱 GLM-V、OpenAI GPT-4o、Anthropic Claude 或任何 OpenAI 兼容端点，
完成图片/视频/文档的多模态理解与目标定位。

- 🚀 **零安装**：`uvx visual-understanding <command>` 直接运行（已发布 [PyPI](https://pypi.org/project/visual-understanding/)）
- 🔌 **双模式**：MCP 服务器（`serve`）+ CLI（`analyze`/`ground`/`list-providers`）
- 🌐 **多提供商**：Zhipu / OpenAI / Anthropic / 任意 OpenAI 兼容端点，YAML 配置即插即用
- 👁️ **核心能力**：图片/视频/文档理解 + 目标定位（bounding box）

## 功能

| 能力 | 说明 | 支持的输入 |
|------|------|-----------|
| **多模态理解** (`vision_analyze`) | 图片描述、OCR、视觉问答、文档解读、多图对比 | 图片 URL/路径/base64、视频 URL、文档 URL |
| **目标定位** (`vision_ground`) | 定位图像中的目标，输出归一化坐标，可选画框可视化 | 图片 URL/路径/base64 |
| **提供商查询** (`list_providers`) | 查看已配置的提供商、模型、能力 | — |

## 快速开始

### 安装

**方式一：uvx 运行（推荐，零安装）** —— 已发布到 PyPI：

```bash
# 直接运行，无需安装（uv 自动缓存）
uvx visual-understanding list-providers

# 如果默认镜像（如清华 TUNA）尚未同步新包，可临时指定官方索引：
uvx --default-index https://pypi.org/simple visual-understanding list-providers
```

**方式二：本地安装**：

```bash
cd mcp-servers/visual-understanding
pip install -e .
```

### 配置 API Key

至少设置一个提供商的 API Key（环境变量）：

```bash
# 智谱（推荐——支持原生定位、视频、文件）
export ZHIPU_API_KEY="your_key"       # https://bigmodel.cn/usercenter/proj-mgmt/apikeys

# OpenAI
export OPENAI_API_KEY="your_key"      # https://platform.openai.com/api-keys

# Anthropic
export ANTHROPIC_API_KEY="your_key"   # https://console.anthropic.com/settings/keys
```

### 验证安装

```bash
visual-understanding list-providers
```

## 模式一：MCP 服务器

在 ZCode / Claude Desktop / Cursor 等 MCP 客户端中注册：

**ZCode** (`~/.zcode/cli/config.json`):

```json
{
  "mcpServers": {
    "visual-understanding": {
      "command": "uvx",
      "args": ["visual-understanding", "serve"]
    }
  }
}
```

> 💡 如果默认 PyPI 镜像（清华/中科大等）尚未同步最新版本，加上官方索引参数：
>
> ```json
> { "command": "uvx", "args": ["--default-index", "https://pypi.org/simple", "visual-understanding", "serve"] }
> ```

### 🔑 在 MCP 配置里直接注入环境变量

MCP 客户端支持给子进程注入 `env`——API key 可以不设系统变量，直接写在客户端配置里：

```json
{
  "mcpServers": {
    "visual-understanding": {
      "command": "uvx",
      "args": ["visual-understanding", "serve"],
      "env": {
        "ZHIPU_API_KEY": "你的智谱key",
        "OPENCODE_API_KEY": "你的opencode key"
      }
    }
  }
}
```

**Claude Desktop** 同样支持 `env` 字段（`claude_desktop_config.json`）。

### 📁 配置文件直接写 API Key（可选）

除了环境变量，`config.yaml` 里也支持 `api_key` 字段直接配置（**优先级高于** `api_key_env`）：

```yaml
providers:
  dashscope:
    type: openai_compat
    api_key: "sk-直接写在配置里"      # 不设系统变量也能用
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    chat_models: [qwen-vl-max]
    default_chat_model: qwen-vl-max
```

> ⚠️ 使用 `api_key` 直配时，务必把配置文件加入 `.gitignore` / 不要提交到版本库。

### 🔍 配置诊断

不确定配置是否正确？运行 doctor 检查：配置文件位置、每个提供商的 key 来源、端点连通性：

```bash
visual-understanding doctor
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "visual-understanding": {
      "command": "visual-understanding",
      "args": ["serve"]
    }
  }
}
```

注册后即可在对话中直接使用 `vision_analyze`、`vision_ground`、`list_providers` 工具。

## 模式二：CLI / Skill

```bash
# 描述图片
visual-understanding analyze --images photo.jpg

# OCR 文字提取
visual-understanding analyze --images scan.png --prompt "Extract all text"

# 视觉问答
visual-understanding analyze --images photo.jpg --prompt "What color is the car?"

# 目标定位 + 画框
visual-understanding ground --image photo.jpg --prompt "all people" --visualize --save-path result.png

# 使用特定提供商/模型
visual-understanding analyze --images photo.jpg --provider openai --model gpt-4o
```

> Agent 可通过 `SKILL.md` 中的指引调用 CLI。两种模式共享同一套核心逻辑。

## 提供商配置

### 内置默认

不创建配置文件时，内置六个提供商：

| 提供商 | 类型 | 模型 | 视频 | 文件 | 原生定位 |
|--------|------|------|------|------|---------|
| `zhipu` | OpenAI 兼容 | GLM-V 系列 | ✅ | ✅ | ✅ |
| `openai` | OpenAI 兼容 | GPT-4o 系列 | ❌ | ❌ | ❌ |
| `anthropic` | Anthropic | Claude 系列 | ❌ | ❌ | ❌ |
| `dashscope` | OpenAI 兼容 | 通义千问 VL 系列（阿里百炼） | ❌ | ❌ | ❌ |
| `siliconflow` | OpenAI 兼容 | 开源 VLM（硅基流动，国内直连） | ❌ | ❌ | ❌ |
| `openrouter` | OpenAI 兼容 | 聚合多模型 | ❌ | ❌ | ❌ |

内置预设只是"开箱即用"——真正强大的是自定义能力：**任何 OpenAI 兼容的视觉模型服务都能添加**（vLLM、Ollama、Azure、Together、本地模型等）。

### 自定义配置

```bash
# 方式一：环境变量指定路径
export VISUAL_UNDERSTANDING_CONFIG=/path/to/config.yaml

# 方式二：默认路径
mkdir -p ~/.config/visual-understanding
cp config.example.yaml ~/.config/visual-understanding/config.yaml
```

配置文件格式见 `config.example.yaml`。

### 添加自定义 OpenAI 兼容端点

任何 OpenAI 兼容的视觉模型服务都可以添加（vLLM、Ollama、Together、Azure 等）：

```yaml
providers:
  my-vlm:
    type: openai_compat
    api_key_env: MY_API_KEY              # 环境变量名
    base_url: http://localhost:8080/v1   # API 地址
    chat_models: [qwen-vl-max]
    default_chat_model: qwen-vl-max
    max_images: 10
```

然后：

```bash
export MY_API_KEY="your_key_or_dummy"
visual-understanding analyze --images photo.jpg --provider my-vlm
```

## 架构

```
                    ┌─────────────┐
                    │   config    │  YAML / env vars
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ ops.py   │ │ ops.py   │ │ ops.py   │   ← 共享业务逻辑
        │ do_analyze│ │ do_ground│ │ do_list  │
        └────┬─────┘ └────┬─────┘ └──────────┘
             │             │
     ┌───────┴──────┐     │
     ▼              ▼     ▼
 ┌────────┐   ┌────────┐ ┌──────────┐
 │server.py│  │ cli.py │ │grounding │
 │ (MCP)  │   │ (CLI)  │ │  .py     │
 └───┬────┘   └────────┘ └──────────┘
     │
     ▼
 ┌──────────────────────────────────┐
 │         providers/               │
 │  ┌────────────┐ ┌────────────┐  │
 │  │openai_compat│ │ anthropic  │  │
 │  │(OpenAI/智谱)│ │ (Claude)   │  │
 │  └────────────┘ └────────────┘  │
 └──────────────────────────────────┘
```

- **`config.py`** — Pydantic 配置模型 + YAML 加载（三级查找）
- **`media.py`** — 输入解析（URL/路径/base64 归一化 + SSRF 防护）
- **`providers/`** — 提供商抽象 + 实现（OpenAI 兼容、Anthropic）
- **`grounding.py`** — 定位 prompt 构造、坐标解析、Pillow 画框
- **`ops.py`** — 共享操作逻辑（MCP 工具与 CLI 子命令的唯一调用入口）
- **`server.py`** — FastMCP 服务器（3 个 MCP 工具）
- **`cli.py`** — CLI 入口（4 个子命令：analyze / ground / list-providers / serve）

## 安全设计

- **API 密钥**始终通过环境变量名引用（`api_key_env`），配置文件中不出现明文密钥
- **`base_url`** 仅在配置中指定，工具参数不接受覆盖（防止密钥泄露到恶意端点）
- **URL 输入**仅允许 http/https 公网地址，拒绝 localhost/内网 IP（防 SSRF）
- **`.gitignore`** 排除 `config.yaml`、`.env`

## 技术栈

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (FastMCP v1.x)
- `httpx` 异步 HTTP
- `pydantic` 配置校验
- `pyyaml` 配置文件
- `pillow` grounding 画框可视化

## License

MIT

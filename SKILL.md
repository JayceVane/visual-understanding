---
name: visual-understanding
description: |
  Multi-provider visual understanding tool for images, videos, and documents.
  Supports captioning, OCR, visual Q&A, document analysis, and object grounding
  (bounding-box localisation) through configurable providers (Zhipu GLM-V,
  OpenAI GPT-4o, Anthropic Claude, or any OpenAI-compatible endpoint).
  Use when the user wants to describe, analyze, extract text from, or locate
  objects in visual content. Invoke via CLI: `visual-understanding <subcommand>`.
when_to_use: >
  当用户提供图片/视频/文档并要求描述、解读、提取文字、视觉问答、对比分析、
  目标定位时触发。触发词：看图说话、图片描述、描述这张图、OCR、文字提取、
  视频摘要、文档解读、目标检测、定位、bounding box、caption、describe image、
  extract text、ground、locate、分析图片、识别图片内容。
skillType: "cli"
metadata:
  emoji: "👁️"
  openclaw:
    requires:
      env: [ZHIPU_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY]
      bins: []
    primaryEnv: ZHIPU_API_KEY
    homepage: https://github.com/JayceVane/visual-understanding
---

# Visual Understanding Skill

Multi-provider visual understanding — caption, OCR, Q&A, and object grounding
through Zhipu GLM-V, OpenAI GPT-4o, Anthropic Claude, or any OpenAI-compatible
vision endpoint. All capabilities are accessible via a single CLI.

## When to Use

- Describe, caption, or summarize image/video/document content
- Extract text from images (OCR) or scanned documents
- Answer questions about visual content
- Compare multiple images
- Locate objects in an image with bounding boxes
- User mentions "看图说话", "图片描述", "OCR", "目标检测", "定位", "describe", "caption", "ground"

## Prerequisites

### 环境检查（第一步）

调用前先确认 CLI 可用，按以下顺序探测：

```bash
# 方式一：入口点命令（pip 安装后）
visual-understanding --help

# 方式二：模块调用（Python 环境已安装该包）
python -m visual_understanding --help

# 方式三：uvx 零安装（若前两种不可用）
uvx visual-understanding --help
# 镜像未同步时加官方索引：
uvx --default-index https://pypi.org/simple visual-understanding --help
```

若三种方式都不可用，先安装：

```bash
pip install visual-understanding
# 或从源码：pip install -e /path/to/visual-understanding
```

> ⚠️ 若 `visual-understanding` 命令不在 PATH（Windows 常见），用 `python -m visual_understanding` 代替。

### API Key Setup (Required)

At least one provider's API key must be set as an environment variable.

**Zhipu (recommended — supports native grounding + video + files):**

```bash
export ZHIPU_API_KEY="your_key"
# Get key: https://bigmodel.cn/usercenter/proj-mgmt/apikeys
```

**OpenAI:**

```bash
export OPENAI_API_KEY="your_key"
# Get key: https://platform.openai.com/api-keys
```

**Anthropic:**

```bash
export ANTHROPIC_API_KEY="your_key"
# Get key: https://console.anthropic.com/settings/keys
```

**OpenCode Go** (subscription, https://opencode.ai/go):

```bash
export OPENCODE_API_KEY="your_key"
# Get key: 登录 https://opencode.ai/auth → Create API Key
# 注意：opencode 客户端 auth.json 里的 key 不一定能用于 API 直连，
# 必须从 opencode.ai/auth 网站创建/获取
```

**或者：配置文件直接写 key（MCP 场景推荐）**

`~/.config/visual-understanding/config.yaml` 里直接配置，无需系统环境变量：

```yaml
providers:
  zhipu:
    type: openai_compat
    api_key: "sk-直接写在配置里"    # 优先级高于 api_key_env
    base_url: https://open.bigmodel.cn/api/paas/v4
    chat_models: [glm-5v-turbo]
    default_chat_model: glm-5v-turbo
```

MCP 客户端配置里也能注入 env：

```json
{ "command": "uvx", "args": ["visual-understanding", "serve"],
  "env": { "ZHIPU_API_KEY": "你的key" } }
```

**通用三方 OpenAI 端点**（URL + key 全从 env 注入，自动注册 `custom` 提供商）：

```json
{ "command": "uvx", "args": ["visual-understanding", "serve"],
  "env": {
    "VISUAL_UNDERSTANDING_BASE_URL": "https://三方服务.com/v1",
    "VISUAL_UNDERSTANDING_API_KEY": "sk-xxx",
    "VISUAL_UNDERSTANDING_MODEL": "qwen-vl-max"
  } }
```

- `VISUAL_UNDERSTANDING_BASE_URL` 必填（触发注册）；API key 可选——不设则无认证直连（适合本地 vLLM/Ollama）
- 调用时 `--provider custom`

### 配置诊断

配置有问题先跑 doctor（检查配置文件位置、key 来源、端点连通性）：

```bash
visual-understanding doctor
```

### Provider Configuration (Optional)

配置文件位置（按优先级自动查找，找到即用）：

1. **`$VISUAL_UNDERSTANDING_CONFIG`** 环境变量指定的路径
2. **`~/.config/visual-understanding/config.yaml`**（推荐，本机默认）
3. 内置默认配置（无配置文件时：zhipu + openai + anthropic）

本机已安装配置文件：`C:\Users\JayceVane\.config\visual-understanding\config.yaml`
（含 zhipu + opencode-go 两个提供商，详见文件内注释）

API key 通过 `~/.bashrc` 注入环境变量：
- `OPENCODE_API_KEY` 从 `~/.local/share/opencode/auth.json` 动态提取（无需手动维护）
- `ZHIPU_API_KEY` 需用户手动填写（获取：https://bigmodel.cn/usercenter/proj-mgmt/apikeys）

> ⚠️ 若命令报 "API key not set"：先检查 `echo $ZHIPU_API_KEY`（或对应变量），
> 为空则编辑 `~/.bashrc` 填入 key 后新开终端，或当前终端手动 `export`。

要修改提供商/模型，编辑配置文件后重启调用即可，无需重启任何服务。

## How to Use

### Check Available Providers

```bash
visual-understanding list-providers
```

### Analyze an Image (Caption / OCR / Q&A)

```bash
# Describe an image
visual-understanding analyze --images "https://example.com/photo.jpg"

# Analyze a local file
visual-understanding analyze --images /path/to/photo.png

# Custom prompt (OCR, Q&A, etc.)
visual-understanding analyze --images document.jpg --prompt "Extract all text from this image"
visual-understanding analyze --images photo.jpg --prompt "What color is the car?"

# Multiple images (comparison)
visual-understanding analyze --images img1.jpg img2.png --prompt "Compare these two images"

# Use a specific provider/model
visual-understanding analyze --images photo.jpg --provider openai --model gpt-4o
```

### Analyze a Video

```bash
visual-understanding analyze --videos "https://example.com/clip.mp4" --prompt "Summarize this video"
```

> ⚠️ Videos and files only support URLs (not local paths). Provider must support
> video/file input (Zhipu GLM-V does; OpenAI/Anthropic do not).

### Analyze a Document

```bash
visual-understanding analyze --files "https://example.com/report.pdf" --prompt "Summarize this document"
```

### Locate Objects (Grounding)

```bash
# Get bounding box coordinates (normalized 0-1000)
visual-understanding ground --image photo.jpg --prompt "all people wearing red hats"

# With detection JSON format (includes labels)
visual-understanding ground --image photo.jpg --prompt "cars and pedestrians" --format detection_json

# Visualize boxes on the image
visual-understanding ground --image photo.jpg --prompt "all faces" --visualize --save-path result.png

# Use a specific provider
visual-understanding ground --image photo.jpg --prompt "animals" --provider zhipu --model glm-5v-turbo
```

### Save Results

```bash
visual-understanding analyze --images photo.jpg --output result.json
```

## CLI Reference

### `analyze`

```
visual-understanding analyze (--images IMG [IMG...] | --videos VID [VID...] | --files FILE [FILE...]) [OPTIONS]
```

| Parameter         | Required | Description                                              |
| ----------------- | -------- | ------------------------------------------------------- |
| `--images`, `-i`  | One of   | Image URLs, local paths, or `base64:` strings           |
| `--videos`, `-v`  | One of   | Video URLs (mp4/mkv/mov) — provider must support video  |
| `--files`, `-f`   | One of   | Document URLs (pdf/docx/txt/xlsx/pptx)                  |
| `--prompt`, `-p`  | No       | Instruction for the model (default: "Describe this image") |
| `--provider`      | No       | Provider name (default: config default)                 |
| `--model`, `-m`   | No       | Model name (default: provider default)                  |
| `--temperature`   | No       | Sampling temperature (default: 0.8)                     |
| `--max-tokens`    | No       | Max output tokens (default: 2048)                       |
| `--output`, `-o`  | No       | Save JSON result to file                                |

### `ground`

```
visual-understanding ground --image IMG --prompt TEXT [OPTIONS]
```

| Parameter          | Required | Description                                              |
| ------------------ | -------- | ------------------------------------------------------- |
| `--image`, `-i`    | Yes      | Image URL, local path, or `base64:` string              |
| `--prompt`, `-p`   | Yes      | What to locate (e.g. "all people wearing hats")         |
| `--provider`       | No       | Provider name                                            |
| `--model`, `-m`    | No       | Model name                                               |
| `--format`         | No       | `bbox_2d` (default) or `detection_json`                 |
| `--visualize`      | No       | Draw bounding boxes on the image                         |
| `--box-color`      | No       | Box color (default: red)                                 |
| `--box-thickness`  | No       | Box line thickness (default: 3)                          |
| `--save-path`      | No       | Save visualised image to this path                       |
| `--output`, `-o`   | No       | Save JSON result to file                                 |

## Response Format

### analyze

```json
{
  "success": true,
  "text": "A landscape photo showing a mountain range at sunset...",
  "usage": {"prompt_tokens": 128, "completion_tokens": 256, "total_tokens": 384},
  "provider": "zhipu",
  "model": "glm-5v-turbo"
}
```

### ground

```json
{
  "success": true,
  "coordinates": [[100, 200, 300, 400], [500, 600, 700, 800]],
  "labels": ["person-1", "person-2"],
  "raw_text": "[{\"label\": \"person-1\", \"bbox_2d\": [100, 200, 300, 400]}, ...]",
  "provider": "zhipu",
  "model": "glm-5v-turbo",
  "native_grounding": true,
  "visualization_saved_path": "/tmp/visual_understanding_ground.png"
}
```

Coordinates are normalised to 0-1000: `x = round(x_pixel / W * 1000)`.
To convert to pixels: `x_pixel = round(x / 1000 * image_width)`.

## Output Display Rules (Mandatory)

After running any command, **show the full JSON output to the user**. Do not
summarize, truncate, or only say "done".

- For `analyze`: show the full `text` field — this is the model's response.
- For `ground`: show `coordinates` and `labels` if present.
- If `visualization_saved_path` is present, tell the user where the visualised
  image was saved.
- If `error` is present, show the error message and guide the user.

## Error Handling

- **API key not configured**: Show the error, guide user to set the environment variable.
- **Authentication failed (401/403)**: API key invalid/expired → reconfigure.
- **Rate limit (429)**: Quota exhausted → inform user to wait.
- **Provider does not support video/files**: Use Zhipu GLM-V or switch to images.
- **No coordinates found in grounding**: Model may not support grounding natively.
  Use a provider with `native_grounding: true` (e.g. Zhipu GLM-V).

## Adding Custom Providers

Any OpenAI-compatible endpoint works. Add to your config:

```yaml
providers:
  my-vlm:
    type: openai_compat
    api_key_env: MY_API_KEY
    base_url: http://localhost:8080/v1
    chat_models: [my-vlm-model]
    default_chat_model: my-vlm-model
```

Then use `--provider my-vlm`.

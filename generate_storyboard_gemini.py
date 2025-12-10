import os
import json
import time
import base64
import hmac
import hashlib
from typing import Dict, Any, Optional

import requests
import streamlit as st

# ========================
# 智谱 BigModel 配置
# ========================
APP_TITLE = "智谱分镜 + 文案生成助手"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.6"  # 纯文本生成足够用了（如你有更适合的模型名也可改）


# ========================
# JWT（可选）生成：不依赖 pyjwt
# 如果你的 key 是 {id}.{secret} 格式且直接 Bearer 不行，可切换 JWT 模式
# ========================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def make_jwt_from_id_secret(api_key: str, exp_seconds: int = 60) -> str:
    if "." not in api_key:
        raise ValueError("JWT 模式需要 api_key 为 {id}.{secret} 格式。")
    kid, secret = api_key.split(".", 1)

    header = {"alg": "HS256", "sign_type": "SIGN"}
    now_ms = int(time.time() * 1000)
    payload = {
        "api_key": kid,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }

    header_b64 = _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def build_auth_header(raw_key: str, auth_mode: str) -> str:
    raw_key = (raw_key or "").strip()
    if not raw_key:
        raise ValueError("请先填写 ZHIPU_API_KEY（智谱 API Key）。")

    if auth_mode == "直接 API Key（推荐）":
        return f"Bearer {raw_key}"

    if auth_mode == "JWT（id.secret）":
        token = make_jwt_from_id_secret(raw_key)
        return f"Bearer {token}"

    return f"Bearer {raw_key}"


# ========================
# Prompt 构建
# ========================
def build_prompt(brand: str, product: str, duration_sec: int, style: str) -> str:
    return f"""
你是一位资深短视频导演和广告文案，擅长为抖音 / 小红书 / 视频号设计高转化竖版广告。

请为下面的产品设计一个时长约 {duration_sec} 秒的竖版短视频广告分镜，包含每个镜头的文案和用于 AI 出图的英文提示词。

品牌：{brand}
产品：{product}
整体风格：{style}

要求：
1. 输出必须是标准 JSON（不要任何多余解释、注释或 Markdown），顶层结构严格如下：
{{
  "brand": "...",
  "product": "...",
  "duration_sec": {duration_sec},
  "style": "...",
  "scenes": [
    {{
      "id": "S01",
      "time_range": "0.0-2.0",
      "shot_desc": "中文，描述画面，适合给导演看的分镜描述",
      "camera": "中文，镜头机位与运动（如：手持中景推近、航拍俯视摇镜等）",
      "action": "中文，人物动作与关键行为",
      "mood": "中文，情绪氛围（如温馨、紧张、治愈、烟火气）",
      "voiceover": "中文旁白/口播文案，口语化、有销售力，适合配音直接念",
      "image_prompt_en": "英文提示词，用于生成这一镜头的 AI 静帧画面，包含人物、环境、光线、镜头、画质等细节"
    }}
  ]
}}

2. 注意：
- time_range 从 0.0 秒开始，后一镜头开始时间紧接前一镜头结束时间，总时长控制在 {duration_sec} 秒左右。
- voiceover 尽量自然口语化，像真实主播口播，不要新闻播音腔。
- image_prompt_en 尽量详细、摄影感强，可包含人物外观/服装/道具/场景/光线/构图/镜头/画幅 9:16/画质等。

3. 必须只输出 JSON（不要三引号、不要 Markdown 代码块、不要解释）。
""".strip()


# ========================
# 智谱调用 + JSON 解析兜底
# ========================
def call_bigmodel_json(
    base_url: str,
    api_key: str,
    auth_mode: str,
    model: str,
    prompt: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    timeout_sec: int = 90,
) -> Dict[str, Any]:
    auth = build_auth_header(api_key, auth_mode)

    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)

    if resp.status_code != 200:
        try:
            err = resp.json()
            raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(err, ensure_ascii=False)}")
        except Exception:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # 第一轮：直接 json.loads
    try:
        return json.loads(content)
    except Exception:
        pass

    # 第二轮：尝试从文本里截取 JSON（防止模型夹杂少量文字）
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = content[start : end + 1]
        return json.loads(candidate)

    # 仍失败：把原文抛出，便于你排查
    raise ValueError(f"模型未返回可解析 JSON，原始输出如下：\n{content}")


def extract_voiceover(data: Dict[str, Any]) -> str:
    scenes = data.get("scenes", [])
    lines = []
    for scene in scenes:
        sid = scene.get("id", "")
        t = scene.get("time_range", "")
        vo = scene.get("voiceover", "")
        if vo:
            lines.append(f"[{sid} | {t}] {vo}")
    return "\n".join(lines)


# ================= Streamlit UI =================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🎬 智谱分镜 + 文案生成助手")

with st.sidebar:
    st.subheader("🔑 智谱 API Key")
    st.caption("建议用环境变量：ZHIPU_API_KEY；也可在此手动输入。")

    api_key_input = st.text_input(
        "ZHIPU_API_KEY",
        type="password",
        value=os.getenv("ZHIPU_API_KEY", ""),
        help="从 open.bigmodel.cn 获取",
    )

    auth_mode = st.selectbox(
        "鉴权方式",
        ["直接 API Key（推荐）", "JWT（id.secret）"],
        index=0,
        help="若直接方式 401 且你的 key 为 id.secret 形式，可选 JWT。",
    )

    base_url = st.text_input("接口地址", value=DEFAULT_BASE_URL)
    model = st.text_input("模型", value=DEFAULT_MODEL)

    st.divider()
    temperature = st.slider("temperature", 0.0, 1.5, 0.6, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.01)
    max_tokens = st.slider("max_tokens", 512, 8192, 4096, 256)

col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("品牌（必填）", value="邵警秘卤")
    product = st.text_input("产品（必填）", value="卤鸭脖+卤鸭翅 夜宵套餐")
    duration_sec = st.number_input("视频时长（秒）", min_value=5, max_value=120, value=15, step=1)

with col2:
    style = st.text_area(
        "整体风格（中文描述）",
        value="烟火气、夜宵档、适合抖音的真实街边风格，有点幽默",
        height=100,
    )

if st.button("✨ 生成分镜 & 文案", type="primary"):
    if not brand or not product:
        st.error("请先填写品牌和产品名称")
    else:
        with st.spinner("正在调用智谱生成分镜，请稍等..."):
            try:
                prompt = build_prompt(brand, product, int(duration_sec), style)
                data = call_bigmodel_json(
                    base_url=base_url.strip(),
                    api_key=api_key_input.strip(),
                    auth_mode=auth_mode,
                    model=model.strip(),
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                st.error(f"调用智谱出错：{e}")
            else:
                st.success("生成完成！")

                st.subheader("📜 分镜 JSON")
                st.json(data)

                voice_script = extract_voiceover(data)
                st.subheader("🎙 旁白脚本")
                st.text_area("可复制给配音用", value=voice_script, height=220)

                st.download_button(
                    "下载 storyboard.json",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name="storyboard.json",
                    mime="application/json",
                )
                st.download_button(
                    "下载 voiceover_script.txt",
                    data=voice_script,
                    file_name="voiceover_script.txt",
                    mime="text/plain",
                )

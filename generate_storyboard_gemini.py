import os
import json
from typing import Dict, Any

import streamlit as st
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


def build_prompt(brand: str, product: str, duration_sec: int, style: str) -> str:
    return f"""
你是一位资深短视频导演和广告文案，擅长为抖音 / 小红书 / 视频号设计高转化竖版广告。

请为下面的产品设计一个时长约 {duration_sec} 秒的竖版短视频广告分镜，包含每个镜头的文案和用于 AI 出图的英文提示词。

品牌：{brand}
产品：{product}
整体风格：{style}

要求：
1. 输出必须是标准 JSON（不要任何多余解释、注释或 Markdown），顶层结构：
{{
  "brand": "...",
  "product": "...",
  "duration_sec": 15,
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
- time_range 从 0.0 秒开始，后一镜头的开始时间紧接前一镜头结束时间，总时长控制在 {duration_sec} 秒左右。
- voiceover 尽量自然口语化，像一个真实主播在讲，而不是新闻播音腔。
- image_prompt_en 要尽量详细、摄影感强，可以包含：
  - 人物外形 / 年龄 / 性别 / 国籍
  - 服装 / 道具
  - 场景（室内/夜市/街头/厨房等）
  - 光线（soft light, cinematic lighting, warm tone 等）
  - 构图和镜头（close-up, medium shot, wide shot, 9:16 等）
  - 画质（8k, ultra detailed, high dynamic range）
"""


def generate_storyboard(
    brand: str,
    product: str,
    duration_sec: int,
    style: str,
) -> Dict[str, Any]:
    prompt = build_prompt(brand, product, duration_sec, style)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    text = response.text
    data = json.loads(text)
    return data


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

st.set_page_config(page_title="Gemini 分镜生成小工具", layout="wide")
st.title("🎬 Gemini 分镜 + 文案生成助手")

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
        with st.spinner("正在调用 Gemini 生成分镜，请稍等..."):
            try:
                data = generate_storyboard(brand, product, duration_sec, style)
            except Exception as e:
                st.error(f"调用 Gemini 出错：{e}")
            else:
                st.success("生成完成！")

                # 左侧展示 JSON 分镜
                st.subheader("📜 分镜 JSON")
                st.json(data)

                # 右侧展示旁白脚本
                voice_script = extract_voiceover(data)
                st.subheader("🎙 旁白脚本")
                st.text_area("可复制给配音用", value=voice_script, height=200)

                # 下载按钮
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

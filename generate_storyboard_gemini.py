import os
import json
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

# 从环境变量读取 API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("环境变量 GEMINI_API_KEY 未设置")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Gemini Storyboard API")


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


def generate_storyboard(brand: str, product: str, duration_sec: int, style: str) -> Dict[str, Any]:
    prompt = build_prompt(brand, product, duration_sec, style)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
        },
    )
    text = response.text
    data = json.loads(text)
    return data


class StoryboardRequest(BaseModel):
    brand: str
    product: str
    duration_sec: int = 15
    style: str = "生活感、烟火气、真实、有点幽默"


@app.post("/generate_storyboard")
def generate_storyboard_endpoint(req: StoryboardRequest):
    data = generate_storyboard(
        brand=req.brand,
        product=req.product,
        duration_sec=req.duration_sec,
        style=req.style,
    )
    return data

import os
import json
from typing import Dict, Any

from google import genai

# 从环境变量读取 Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 GEMINI_API_KEY 再运行本脚本。")

# 初始化 Gemini 客户端
client = genai.Client(api_key=API_KEY)


def build_prompt(brand: str, product: str, duration_sec: int, style: str) -> str:
    """
    构造给 Gemini 的提示词，让它输出【分镜 + 文案 + 出图提示词】的 JSON。
    这个模板你后续可以根据自己口味继续调。
    """
    prompt = f"""
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
    return prompt


def generate_storyboard(
    brand: str,
    product: str,
    duration_sec: int = 15,
    style: str = "生活感、烟火气、真实、有点幽默"
) -> Dict[str, Any]:
    """
    调用 Gemini 生成分镜 JSON。
    """
    prompt = build_prompt(brand, product, duration_sec, style)

    response = client.models.generate_content(
        model="gemini-2.0-flash",  # 免费测试非常够用
        contents=prompt,
        config={
            # 让它尽量按 JSON 格式输出
            "response_mime_type": "application/json",
        },
    )

    # response.text 是一个 JSON 字符串
    text = response.text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 如果不小心不符合 JSON，简单做一次容错（你也可以在这里加正则清洗）
        raise ValueError(f"Gemini 返回的内容不是合法 JSON：\n{text}")

    return data


def save_storyboard(data: Dict[str, Any], output_path: str = "storyboard.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 分镜 JSON 已保存到: {output_path}")


def save_voiceover_script(data: Dict[str, Any], output_path: str = "voiceover_script.txt"):
    """
    提取所有镜头的旁白 voiceover，汇总成一个口播文案文件，方便配音。
    """
    lines = []
    scenes = data.get("scenes", [])
    for scene in scenes:
        sid = scene.get("id", "")
        time_range = scene.get("time_range", "")
        vo = scene.get("voiceover", "")
        if vo:
            lines.append(f"[{sid} | {time_range}] {vo}")

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ 旁白文案已保存到: {output_path}")


if __name__ == "__main__":
    # 这里你可以先随便填一个产品测试
    brand = "邵警秘卤"
    product = "卤鸭脖+卤鸭翅 夜宵套餐"
    duration_sec = 15
    style = "烟火气、夜宵档、适合抖音的真实街边风格"

    storyboard = generate_storyboard(
        brand=brand,
        product=product,
        duration_sec=duration_sec,
        style=style
    )

    save_storyboard(storyboard, "storyboard.json")
    save_voiceover_script(storyboard, "voiceover_script.txt")

    print("\n📌 简要预览：")
    print(json.dumps(storyboard.get("scenes", [])[:2], ensure_ascii=False, indent=2))

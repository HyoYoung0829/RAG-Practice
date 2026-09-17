import base64
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image


class MultiModal:
    def __init__(self, llm, system_prompt, user_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    # 이미지와 질문을 메시지로 변환
    def _create_messages(self, image_filepath):
        image_path = Path(image_filepath)
        with Image.open(image_path) as image:
            image_format = image.format
            image.verify()
        mime_types = {"JPEG": "image/jpeg", "PNG": "image/png"}
        if image_format not in mime_types:
            raise ValueError("JPG 또는 PNG 이미지를 사용해 주세요.")

        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_url = f"data:{mime_types[image_format]};base64,{image_data}"
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            ),
        ]

    # 이미지 기반 답변 스트리밍
    def stream(self, image_filepath):
        return self.llm.stream(self._create_messages(image_filepath))

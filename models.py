from __future__ import annotations

import os
from types import SimpleNamespace

import dotenv
from openai import OpenAI
from transformers import AutoModelForImageTextToText, AutoProcessor


dotenv.load_dotenv()


class Qwen3_5:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        system_prompt: str | None = None,
    ):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name_or_path).to(device)
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str, logits_processor=None, enable_thinking: bool = False) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            logits_processor=logits_processor,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()


class BackboneLLM:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        provider: str = "auto",
        temperature: float = 0,
        reasoning_effort: str | None = None,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.provider = self.infer_provider(model, provider)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        if self.provider == "deepseek":
            self.client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                timeout=300,
            )
        elif self.provider == "openai":
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=300)
        elif self.provider == "gemini":
            from google import genai

            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def infer_provider(model: str, provider: str) -> str:
        if provider and provider != "auto":
            return provider
        name = (model or "").lower()
        if name.startswith("deepseek"):
            return "deepseek"
        if name.startswith("gemini"):
            return "gemini"
        return "openai"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        model_name = model or self.model
        temp = self.temperature if temperature is None else temperature
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if self.provider == "gemini":
            return self._generate_gemini(
                system_prompt,
                user_prompt,
                model_name,
                temp,
                max_output_tokens,
            )
        return self._generate_openai_compatible(
            system_prompt,
            user_prompt,
            model_name,
            thinking,
            temp,
            effort,
        )

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        enable_thinking: bool,
        temperature: float,
        reasoning_effort: str | None,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self.provider == "deepseek":
            kwargs["temperature"] = temperature
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if enable_thinking else "disabled"}
            }
        elif self.provider == "openai":
            kwargs["temperature"] = temperature
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (response.text or "").strip()

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self.generate("", prompt))


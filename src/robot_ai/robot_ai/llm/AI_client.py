import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional
import threading
import requests
from .memory_manager import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openrouter_client")


#API_KEY_PATH = BASE_DIR / "config" / "api_keys.json"

BASE_DIR = Path(__file__).resolve().parent

PROMPT_PATH = str(BASE_DIR / "prompts" / "prompt.txt")
MEMORY_PROMPT_PATH = str(BASE_DIR / "prompts" / "memoryprompt.txt")



def _load_api_key() -> str:
    try:
        with open(API_KEY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("openrouter_api_key", "").strip()
        if not key:
            raise ValueError("openrouter_api_key is empty in api_keys.json")
        return key
    except FileNotFoundError:
        raise RuntimeError(f"api_keys.json not found at: {API_KEY_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenRouter API key: {e}")


# TODO: split these Ai's in three levels of intelligence, and use the 2nd level Ai as the main one to decide what to use for each task.
TEXT_MODELS: list[str] = [
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "arcee-ai/trinity-large-preview:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "minimax/minimax-m2.5:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3-27b-it:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3-12b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3n-e2b-it:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
]

VISION_MODELS: list[str] = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3n-e2b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

API_URL               = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS    = 4096
DEFAULT_TEMPERATURE   = 0.7
REQUEST_TIMEOUT       = 60   # seconds per request
MAX_RETRIES_PER_MODEL = 2    # attempts before moving to next model
RETRY_DELAY           = 3    # seconds between retries
RATE_LIMIT_COOLDOWN   = 200   # seconds before retrying a rate-limited model

_rate_limited: dict[str, float] = {}


def load_system_prompt(PROMPT_PATH = PROMPT_PATH) -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ prompt.txt couldn't be loaded: {e}")
        return "You are Skye, a helpful AI assistant."



class OpenRouterClient:

    def __init__(self) -> None:
        self.api_key  = "REDACTED"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/mark-xxv",
            "X-Title":       "SKYE",
        }

        # Current chat stored in memory for less delay
        self.chat = []

    def _is_rate_limited(self, model: str) -> bool:
        ts = _rate_limited.get(model)
        if ts is None:
            return False
        if time.time() - ts > RATE_LIMIT_COOLDOWN:
            del _rate_limited[model]
            return False
        return True

    def _mark_rate_limited(self, model: str) -> None:
        _rate_limited[model] = time.time()
        logger.warning(
            f"[OpenRouter] Rate limited: {model} — "
            f"cooling down for {RATE_LIMIT_COOLDOWN}s"
        )

    def _call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        payload: dict = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                resp = requests.post(
                    API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 429:
                    self._mark_rate_limited(model)
                    return None

                if resp.status_code == 200:
                    data    = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                    )
                    return content.strip() if content else None

                logger.warning(
                    f"[OpenRouter] {model} → HTTP {resp.status_code} "
                    f"(attempt {attempt}/{MAX_RETRIES_PER_MODEL})"
                )

            except requests.exceptions.Timeout:
                logger.warning(
                    f"[OpenRouter] {model} → Timeout "
                    f"(attempt {attempt}/{MAX_RETRIES_PER_MODEL})"
                )
            except Exception as e:
                logger.error(f"[OpenRouter] {model} → Unexpected error: {e}")

            if attempt < MAX_RETRIES_PER_MODEL:
                time.sleep(RETRY_DELAY)

        return None

    def _call_with_fallback(
        self,
        pool: list[str],
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        response_format: Optional[dict] = None,
    ) -> str:
        if model and not self._is_rate_limited(model):
            result = self._call(model, messages, max_tokens, temperature, response_format)
            if result:
                return result
            logger.info(
                f"[OpenRouter] Requested model failed, "
                f"falling back to pool: {model}"
            )

        for m in pool:
            if self._is_rate_limited(m):
                continue
            #logger.info(f"[OpenRouter] Trying: {m}")
            result = self._call(m, messages, max_tokens, temperature, response_format)
            if result:
                #logger.info(f"[OpenRouter] ✓ Success: {m}")
                print(m)
                return result

        raise RuntimeError(
            "[OpenRouter] All models failed or are rate-limited. "
            "Check your API key and network connection."
        )

    def _background_memory_saver(self):
        pass

    def chat(
        self,
        prompt: str,
        system: str = (
            "You are Skye, an AI assistant inspired by Wall-E. "
            "Be concise, helpful, and precise."
        ),
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        return self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature
        )

    def chat_json(
        self,
        prompt: str,
        system: str = load_system_prompt(),
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        addtional_ai_context: dict = {}
    ) -> dict:
        messages = [
            {"role": "system", "content": system + str(addtional_ai_context)},
            {"role": "user",   "content": prompt},
        ]
        raw = self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature=0.2
        )

        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("`").strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(
                f"[OpenRouter] JSON parse failed: {e}\n"
                f"Raw response (first 300 chars): {raw[:300]}"
            )
            raise ValueError(
                f"Model returned unparseable JSON: {e}\n"
                f"Raw output: {raw[:200]}"
            )

    def vision(
        self,
        prompt: str,
        image_b64: str,
        mime: str = "image/png",
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._call_with_fallback(
            VISION_MODELS, messages, model, max_tokens, temperature=0.2
        )

    def vision_from_file(
        self,
        prompt: str,
        image_path: str,
        system: str = "Analyze the image and describe what you see clearly and concisely.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        path = Path(image_path)
        mime_map = {
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif":  "image/gif",
        }
        mime = mime_map.get(path.suffix.lower(), "image/png")

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        return self.vision(prompt, image_b64, mime, system, model, max_tokens)

    def multi_turn(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
    
        return self._call_with_fallback(
            TEXT_MODELS, messages, model, max_tokens, temperature
        )

    def available_models(self) -> dict:
        return {
            "text_models":   TEXT_MODELS,
            "vision_models": VISION_MODELS,
            "rate_limited":  list(_rate_limited.keys()),
            "total_text":    len(TEXT_MODELS),
            "total_vision":  len(VISION_MODELS),
        }

    
    def chat_main(self,
        prompt: str,
        system: str = load_system_prompt(),
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        addtional_ai_context: dict = {}
        ) -> dict:

        if prompt == "exit":
            current = str(build_current_memories()) + str(self.chat)
            memeory_saver = self.chat_json(prompt=current, system=load_system_prompt(MEMORY_PROMPT_PATH))
            long_term_memory(memeory_saver)
            print(memeory_saver)
            return
        
        
        ai_context = str(addtional_ai_context) + ", " + "current chat: " + str(self.chat)

        # call AI normally with build in function
        result = self.chat_json(prompt, system, model, max_tokens, ai_context)

        self.chat.append({"User": prompt, "Skye": result})

        # Save chat memory in .json
        threading.Thread(target=short_term_memory, args=(prompt, result)).start()

        return result
        

"""class AI_client(Node):
    def __init__(self):
        super().__init__('Robot_AI_Client')

        self.ai_response_publisher = self.create_publisher(String, "/ai/response", 10)
        self.ai_request_subscriber = self.create_subscription(String, "ai/request", ai_request_callback, 10)

        self.client = OpenRouterClient()

    
    def ai_request_callback(self, user_input):
        prompt = user_input.data

        self.ai_chat(prompt)


    def ai_chat(self, prompt):
        try:
            response = client.chat_main(
                prompt=user_input,
                system=load_system_prompt()
            )

            msg = String()
            msg.data = "AI: " + str(response)
            
            self.ai_response_publisher.publish(msg)

        except Exception as e:
            msg = String()
            msg.data = "Error: " + str(e)

            self.ai_response_publisher.publish(msg)"""
            



def main():
    rclpy.init()
    node = AI_client()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

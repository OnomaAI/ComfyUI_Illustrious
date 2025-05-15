import os
from dotenv import load_dotenv
from .autonode import node_wrapper, get_node_names_mappings, validate
import base64
import io
import numpy as np
import torch
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from urllib3.exceptions import InsecureRequestWarning


load_dotenv()
cur_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(cur_dir, ".env")):
    load_dotenv(os.path.join(cur_dir, ".env"))

ILLUSTRIOUS_API_KEY = os.getenv("ILLUSTRIOUS_API_KEY")
if not ILLUSTRIOUS_API_KEY:
    raise EnvironmentError("Environment variable 'ILLUSTRIOUS_API_KEY' is not set. Please add it to your .env file.")

fundamental_classes = []
fundamental_node = node_wrapper(fundamental_classes)

def fetch_model_names() -> list[str]:
    url = "https://api.v1.illustrious-xl.ai/api/model/list"
    headers = {
        "Content-Type": "application/json",
        "x-illustrious-api-key": ILLUSTRIOUS_API_KEY,
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    models = resp.json() 
    return [(m["id"], m["name"]) for m in models]

MODEL_INFO = fetch_model_names()
MODEL_NAMES = [m[1] for m in MODEL_INFO]
MODEL_IDS = [m[0] for m in MODEL_INFO]


@fundamental_node
class IllustriousGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAMES,),
                "width": ("INT", {"min": 256, "max": 2048, "default": 1024}),
                "height": ("INT", {"min": 256, "max": 2048, "default": 1024}),
                "prompt": ("STRING", {"multiline": True, "default": "1 girl"}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "worst quality, low quality, lowers, low details, "
                            "bad quality, poorly drawn, bad anatomy, multiple views, "
                            "bad hands, blurry, artist sign, weibo username"
                        ),
                    },
                ),
                "steps": ("INT", {"min": 1, "max": 100, "default": 28}),
                "cfg": ("FLOAT", {"min": 1.0, "max": 20.0, "default": 7.5}),
                "sampler": (['euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp','dpm_2', 'dpm_2_ancestral', 'dpmpp_2s_ancestral', 'dpmpp_2s_ancestral_cfg_pp'],),
                "scheduler": (['normal', 'karras', 'exponential', 'sgm_uniform', 'simple','ddim_uniform', 'beta'],),
                "seed": ("INT", {"min": -1, "max": 2**32 - 1, "default": -1}),
                "n_requests": ("INT", {"min": 1, "max": 10, "default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING")
    RETURN_NAMES = ("images","usage","balance")
    FUNCTION = "generate"
    custom_name = "Illustrious-Generator"
    CATEGORY = "Illustrious"

    @staticmethod
    def _build_headers() -> dict:
        return {
            "User-Agent": "MyCustomAgent/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "sec-ch-ua": 'Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            "sec-ch-ua-mobile": "?0",
            "x-illustrious-api-key": ILLUSTRIOUS_API_KEY,
        }

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    @staticmethod
    def _b64_to_tensor(data_url_or_b64: str) -> torch.Tensor:
        raw = base64.b64decode(data_url_or_b64)
        pil = Image.open(io.BytesIO(raw))
        return IllustriousGenerate._pil_to_tensor(pil)
    

    def generate(
        self,
        model_name,
        width,
        height,
        prompt,
        negative_prompt,
        steps,
        cfg,
        sampler,
        scheduler,
        seed,
        n_requests,
    ):
        base_params = {
            "modelId": MODEL_IDS[MODEL_NAMES.index(model_name)],
            "steps": steps,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negativePrompt": negative_prompt,
            "cfgScale": cfg,
            "samplerName": sampler,
            "scheduler": scheduler,
        }
        

        headers = self._build_headers()
        warnings.simplefilter("ignore", InsecureRequestWarning)
        def _call(_worker_idx):
            p = base_params.copy()
            p["seed"] = random.randint(0, 2**32 - 1) if seed == -1 else seed
            p["seed"] = p["seed"] + _worker_idx
            print(f"IllustriousGenerate: {base_params}")
            try:
                url = "https://api.v1.illustrious-xl.ai/api/text-to-image/generate"
                r = requests.post(url, headers=headers, json=p, verify=False, timeout=600)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                message = str(r.json().get("message")) + f" (status code: {r.status_code})"
                raise RuntimeError(f"IllustriousGenerate error: {message}")

        with ThreadPoolExecutor(max_workers=n_requests) as pool:
            results = list(pool.map(_call, range(n_requests)))  

        tensors = []
        usages = 0
        balances = []
        for result in results:
            if not result:
                tensors.append(torch.zeros((height, width, 3), dtype=torch.float32))
            else:
                tensor = result.get("images")[0]
                usage = result.get("stardustUsage")
                balance = result.get("stardustBalance")
                tensors.append(self._b64_to_tensor(tensor))
                usages+=usage
                balances.append(int(balance))
            balance = str(min(balances))
        batch = torch.stack(tensors, dim=0)
        print(f'usage: {usages}')
        print(f'balance: {balance}')
        return (batch,usages,balance)
    

@fundamental_node
class IllustriousTagBooster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "tag_length": (["short", "long"], {"default": "short"}),
                "ban_tags": ("STRING", {"multiline": True, "default": ""}),
                "format": ("STRING", {"multiline": True, "default": "<|special|>, <|characters|>, <|quality|>, <|meta|>, <|rating|>"}),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.5}),
                "top_p": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.9}),
                "top_k": ("INT", {"min": 0, "max": 100, "default": 100}),
                "seed": ("INT", {"min": -1, "max": 2**32 - 1, "default": -1}),
                "mode": (["merge input string", "just prompt"], {"default": "just prompt"}),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("tags","usage","balance")
    FUNCTION = "boost_tags"
    custom_name = "Illustrious-TagBooster"
    CATEGORY = "Illustrious"

    def boost_tags(
        self,
        text,
        tag_length,
        ban_tags,
        format,
        temperature,
        top_p,
        top_k,
        seed,
        mode,
    ):
        payload = {
            "text": text,
            "tag_length": tag_length,
            "ban_tags": ban_tags,
            "format": format,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": random.randint(0, 2**32 - 1) if seed == -1 else seed,
        }
        headers = {
            "Content-Type": "application/json",
            "x-illustrious-api-key": ILLUSTRIOUS_API_KEY,
        }
        url = "https://api.v1.illustrious-xl.ai/api/text-enhance/tag-booster"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            tags = data.get("output", "")
            usage = data.get("stardustUsage", 0)
            balance = data.get("stardustBalance", 0)
            if mode == "merge input string":
                tags = text + ", " + tags
            print(f'tags: {tags}')
            print(f'tag booster usage: {usage}')
            print(f'balance: {balance}')
            return (tags, usage, balance)
        except Exception as e:
            message = str(r.json().get("message")) + f" (status code: {r.status_code})"
            raise RuntimeError(f"IllustrioustagBooster error: {message}")

@fundamental_node
class IllustriousMoodEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["merge input string", "just prompt"], {"default": "just prompt"}),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("prompt","usage","balance")
    FUNCTION = "enhance_mood"
    custom_name = "Illustrious-MoodEnhancer"
    CATEGORY = "Illustrious"

    def enhance_mood(self, text, mode):
        payload = {"text": text}
        headers = {
            "Content-Type": "application/json",
            "x-illustrious-api-key": ILLUSTRIOUS_API_KEY,
        }
        url = "https://api.v1.illustrious-xl.ai/api/text-enhance/mood-enhancer"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            prompt = data.get("output", "")
            usage = data.get("stardustUsage", 0)
            balance = data.get("stardustBalance", 0)
            if mode == "merge input string":
                prompt = text + ", " + prompt
            print(f'prompt: {prompt}')
            print(f'mood enhancer usage: {usage}')
            print(f'balance: {balance}')
            return (prompt, usage, balance)
        except Exception as e:
            message = str(r.json().get("message")) + f" (status code: {r.status_code})"
            raise RuntimeError(f"IllustriousMoodEnhancer error: {message}")

CLASS_MAPPINGS = get_node_names_mappings(fundamental_classes)
CLASS_MAPPINGS, CLASS_NAMES = get_node_names_mappings(fundamental_classes)
validate(fundamental_classes)

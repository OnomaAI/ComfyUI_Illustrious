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
MODELS = ["ILXL v1.0", "ILXL v1.1", "ILXL v2.0 Base", "ILXL v2.0 Refined","ILXL v3.0 Creative+", "ILXL v3.0 Creative", "ILXL v3.0 Expressive+", "ILXL v3.0 Expressive","ILXL v3.5 Creative", "ILXL v3.5 Expressive"]

@fundamental_node
class IllustriousGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # API key is handled via environment variable
                "model_name": (MODELS,),
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
    custom_name = "Illustrious-XL-API-Generator"
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
        """
        PIL.Image (RGB) → torch.Tensor [H,W,C], float32, 0-1
        """
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    @staticmethod
    def _b64_to_tensor(data_url_or_b64: str) -> torch.Tensor:
        """Handle both bare base64 strings and full data URLs."""
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
            "modelId": MODELS.index(model_name) + 1,
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
        def _call(_):
            p = base_params.copy()
            p["seed"] = random.randint(0, 2**32 - 1) if seed == -1 else seed

            try:
                url = "https://api.v1.illustrious-xl.ai/api/text-to-image/generate"
                r = requests.post(url, headers=headers, json=p, verify=False, timeout=120)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"IllustriousGenerate error: {e}")
                return []

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

CLASS_MAPPINGS = get_node_names_mappings(fundamental_classes)
CLASS_MAPPINGS, CLASS_NAMES = get_node_names_mappings(fundamental_classes)
validate(fundamental_classes)

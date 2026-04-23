from pathlib import Path
import torch
from PIL import Image
from .base import BaseVLM

class DeepSeekVL2(BaseVLM):
    def __init__(
        self,
        model_name: str = "deepseekvl2_small",
        model_id: str = "deepseek-ai/deepseek-vl2-small",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model_name=model_name, device=device,
                         model_id=model_id, **kwargs)
        self.model_id = model_id

    def load_model(self, model_id: str, **kwargs):
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

        self.logger.info(f"Loading {model_id}...")
        self.processor = DeepseekVLV2Processor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def predict(self, image_path: str | Path, question: str) -> str:
        self.ensure_loaded()
        from deepseek_vl2.utils.io import load_pil_images

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [str(image_path)],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )

        return self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        ).strip()

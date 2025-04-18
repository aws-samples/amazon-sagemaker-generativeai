from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from djl_python import Input, Output
from typing import Optional
import torch
import json
import os


@dataclass
class Config:
    # models can optionally be passed in directly
    model = None
    model_name: Optional[str] = 'Qwen/Qwen2-0.5B-Chat' 
    # interrogator settings
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Qwen1_5():
    def __init__(self, config: Config, properties):
        self.history = None
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.load_model(properties)
        self.caption_offloaded = True

    def load_model(self, properties):
        if self.config.model_name is not None:
            print(f'model name: {self.config.model_name}')
            # Note: The default behavior now has injection attack prevention off.
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name,
                                                           trust_remote_code=True)
            # use bf16
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name,
                                                              device_map=self.device,
                                                              trust_remote_code=True,
                                                              torch_dtype="auto").eval()
        else:
            raise ValueError('Please make sure to set model_name')

    def generate_text(self, prompt: [str], params: Optional[dict] = {}) -> str:
        assert self.model is not None, "No model loaded."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


with open('./model_name.json', 'rb') as openfile:
    json_object = json.load(openfile)

model_name = json_object.pop('model_name')
config = None
_service = None


def handle(inputs: Input) -> Optional[Output]:
    global config, _service
    if not _service:
        config = Config()
        config.model_name = model_name
        _service = Qwen1_5(config, inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()

    if 'prompt' in data:
        prompt = data.pop("prompt")
    else:
        return None

    params = data["parameters"] if 'parameters' in data else {}
    generated_text = _service.generate_text(prompt, params)

    return Output().add(generated_text)
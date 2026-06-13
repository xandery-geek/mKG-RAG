import copy
import torch
import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag.llm.utils import locate_json_string_body_from_string
from rag.exceptions import APIConnectionError, RateLimitError, APITimeoutError


def hf_format_message(prompt, system_prompt=None, image=None, history_messages=[]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)

    if image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})
    
    return messages


def hf_load_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        raise ValueError("Image should be a path to an image file.")
    return image


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_processor = AutoProcessor.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, use_fast=True
    )

    if hf_processor.tokenizer.pad_token is None:
        hf_processor.tokenizer.pad_token = hf_processor.tokenizer.eos_token

    if "72B" in model_name:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 CPU offloading
        )
    else:
        quantization_config = None

    if "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            quantization_config=quantization_config
        )
    elif "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            quantization_config=quantization_config
        )
    elif "Llama" in model_name:
        from transformers import MllamaForConditionalGeneration
        hf_model = MllamaForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, 
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    
    hf_model.eval()

    return hf_model, hf_processor


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def hf_model_if_cache(
    model,
    prompt,
    image=None,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    model_name = model
    hf_model, hf_processor = initialize_hf_model(model_name)
    messages = hf_format_message(prompt, system_prompt, image, history_messages)

    input_prompt = ""
    try:
        input_prompt = hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    image = hf_load_image(image) if image else None

    input_ids = hf_processor(
        images=image,
        text=input_prompt,
        return_tensors="pt", 
        padding=True,
        truncation=True
    ).to(hf_model.device)
    
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    max_tokens = kwargs.pop("max_tokens", 512)

    with torch.no_grad():
        outputs = hf_model.generate(
            **input_ids, max_new_tokens=max_tokens, num_return_sequences=1, early_stopping=True
        )

    outputs_ids_trimmed = outputs[0][len(inputs["input_ids"][0]) :]
    response_text = hf_processor.decode(outputs_ids_trimmed, skip_special_tokens=True).strip()

    return response_text


async def hf_model_complete(
    prompt, image=None, system_prompt=None, history_messages=[], llm_model_name=None, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await hf_model_if_cache(
        llm_model_name,
        prompt,
        image=image,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


async def hf_text_embed(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    encoded_texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        if hasattr(embed_model, "get_text_features"):
            # for models like CLIP
            embeddings = embed_model.get_text_features(**encoded_texts)
        else:
            outputs = embed_model(**encoded_texts)
            # mean pooling on all token embeddings 
            # [batch_size, token_length, hidden_size] -> [batch_size, hidden_size]
            embeddings = mean_pooling(outputs, encoded_texts['attention_mask'])

        embeddings = F.normalize(embeddings, p=2, dim=1)
    if embeddings.dtype != torch.float32:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()


async def hf_vision_embed(images, processor, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    inputs = processor(images=images, return_tensors="pt")
    inputs = inputs.pixel_values.to(device)

    with torch.no_grad():
        if hasattr(embed_model, "get_image_features"):
            # for models like CLIP
            embeddings = embed_model.get_image_features(inputs)
        else:
            # get the embeddings of the cls token from the last layer
            outputs = embed_model(inputs)
            embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

    if embeddings.dtype != torch.float32:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()
    

async def hf_mm_embed(data, vis_processor, txt_processor, embed_model, text_type) -> np.ndarray:
    device = next(embed_model.parameters()).device
    
    if not isinstance(data, list):
        data = [data]

    images, texts = list(zip(*data))
    images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]
    
    image_input = [vis_processor(image) for image in images]
    text_input = [txt_processor(text) for text in texts]

    image_input = torch.stack(image_input).to(device)
    sample = {"image": image_input, "text_input": text_input}

    with torch.no_grad():
        embeddings = embed_model.extract_features(sample, text_type=text_type)
        embeddings = F.normalize(embeddings.mean(dim=1), dim=1)

    if embeddings.dtype != torch.float32:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()
    

if __name__ == "__main__":
    import asyncio
    llm_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"

    async def main():
        prompt = "What is the location of this image?"
        image = ".cache/examples/img/Cahir_Castle.jpg"
        response = await hf_model_complete(prompt, image=image, llm_model_name=llm_model_name)
        print(response)

    asyncio.run(main())

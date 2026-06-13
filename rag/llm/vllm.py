import torch
import vllm

from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag.llm.hf import hf_format_message, hf_load_image
from rag.llm.utils import locate_json_string_body_from_string
from rag.exceptions import APIConnectionError, RateLimitError, APITimeoutError


@lru_cache(maxsize=1)
def initialize_vllm_model(model_name, max_model_len, max_num_seqs):
    model = vllm.LLM(
        model_name,
        max_num_batched_tokens=max_model_len * max_num_seqs,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        task="generate",
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


@lru_cache(maxsize=1)
def initialize_vllm_sampling_params(temperature=0.6, top_p=0.9, num_beams=1, max_tokens=512):
    return vllm.SamplingParams(
        temperature=temperature,
        top_p=top_p,
        best_of=num_beams,
        max_tokens=max_tokens,
        skip_special_tokens=True
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def vllm_model_if_cache(
    model,
    prompt,
    image=None,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    max_model_len = kwargs.pop("llm_model_max_token_size", 32768)
    max_num_seqs = kwargs.pop("llm_model_max_async", 10)

    vllm_model, vllm_tokenizer = initialize_vllm_model(model, max_model_len=max_model_len, max_num_seqs=max_num_seqs)

    messages = hf_format_message(prompt, system_prompt, image, history_messages)
    text_prompt = vllm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = hf_load_image(image) if image else None

    vllm_prompts = [{
        "prompt": text_prompt,
        "multi_modal_data": {"image": image} if image else None,
    }]

    max_tokens = kwargs.pop("max_tokens", 512)
    with torch.no_grad():
        outputs = vllm_model.generate(
            vllm_prompts,
            sampling_params = initialize_vllm_sampling_params(max_tokens=max_tokens),
            use_tqdm=False,
        )

    response = outputs[0].outputs[0].text
    return response


async def vllm_model_complete(
    prompt, image=None, system_prompt=None, history_messages=[], llm_model_name=None, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await vllm_model_if_cache(
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


if __name__ == "__main__":
    import asyncio
    llm_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"

    async def main():
        prompt = "What is the location of this image?"
        image = ".cache/examples/img/Cahir_Castle.jpg"
        response = await vllm_model_complete(prompt, image=image, llm_model_name=llm_model_name)
        print(response)

    asyncio.run(main())

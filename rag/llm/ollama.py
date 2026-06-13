import sys
import ollama
import numpy as np
from typing import Union

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag.exceptions import APIConnectionError, RateLimitError, APITimeoutError


def get_headers(api_key, api_version="1.0.0"):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"MMRAG/{api_version}",
    }
    if api_key:
        headers["Authorization"] = api_key
    return headers


def ollama_format_message(prompt, system_prompt=None, image=None, history_messages=[]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)

    if image:
        messages.append({"role": "user", "content": prompt, "images": [image]})
    else:
        messages.append({"role": "user", "content": prompt})
    
    return messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    image=None,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    api_key = kwargs.pop("api_key", None)
    headers = get_headers(api_key)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    
    messages = ollama_format_message(prompt, system_prompt, image, history_messages)
    response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    
    if stream:
        """cannot cache stream response and process reasoning"""
        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        model_response = response["message"]["content"]

        """
        If the model also wraps its thoughts in a specific tag,
        this information is not needed for the final
        response and can simply be trimmed.
        """

        return model_response


async def ollama_model_complete(
    prompt, image=None, system_prompt=None, history_messages=[], llm_model_name=None, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"

    return await _ollama_model_if_cache(
        llm_model_name,
        prompt,
        image=image,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    headers = get_headers(api_key)
    kwargs["headers"] = headers
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embed(model=embed_model, input=texts)
    return np.float32(data["embeddings"])


if __name__ == "__main__":
    import asyncio
    
    host = "http://localhost:11434"
    llm_model_name = "llama3.2-vision:latest"

    async def main():
        prompt = "What is the location of this image?"
        image = ".cache/examples/img/Cahir_Castle.jpg"
        response = await ollama_model_complete(prompt, image=image, llm_model_name=llm_model_name, host=host)
        print(response)

    asyncio.run(main())

import torch
import vllm
from model.generator.base import AnswerGenerator


class AnswerGeneratorForVLLM(AnswerGenerator):        
    def __init__(self, model_path, device, **kwargs):
        super().__init__(device, **kwargs)

        self.model_path = model_path
        self.max_model_len = kwargs.get("max_model_len", 65536)
        self.max_num_seqs = kwargs.get("max_num_seqs", 8)
        self.hf_overrides = kwargs.get("hf_overrides", None)
        self.stop_tokens = kwargs.get("stop_tokens", None)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        
        self.prompt_template = kwargs.get("prompt_template", None)
        
        if self.prompt_template is None:
            pass

        self.model, self.tokenizer = self._load_model()
        self.sampling_params = self._load_sampling_params()
    
    def _get_default_prompt_template(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages
    
    def _apply_prompt_template(self, prompt, prompt_template=None):
        if prompt_template is None:
            messages = self._get_default_prompt_template(prompt)
        else:
            assert isinstance(prompt_template, list), "prompt_template must be a list of messages"
            messages = []
            for message in prompt_template:
                messages.append(
                    {key: val.format(prompt=prompt) if isinstance(val, str) else val 
                     for key, val in message.items()}
                )
        
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted_prompt

    def _load_model(self):
        llm = vllm.LLM(
            self.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            max_num_batched_tokens=self.max_model_len * self.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            hf_overrides=self.hf_overrides,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
            task="generate",
        )
        tokenizer = llm.get_tokenizer()
        return llm, tokenizer
    
    def _load_sampling_params(self):
        if self.stop_tokens is not None:
            stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens]
            self.stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
        else:
            self.stop_token_ids = None

        sampling_params = vllm.SamplingParams(
            temperature=self.temperature, 
            top_p=self.top_p,
            best_of=self.num_beams, 
            max_tokens=self.max_new_tokens,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=True,
        )
        return sampling_params

    @torch.no_grad()
    def batch_mm_forward(
        self,
        prompts,
        images,
        **kwargs
    ):
        if self.prompt_template is None:
            prompts = [self._apply_prompt_template(prompt) for prompt in prompts]
        elif isinstance(self.prompt_template, str):
            prompts = [self.prompt_template.format(prompt=prompt) for prompt in prompts]
        elif isinstance(self.prompt_template, list):
            prompts = [self._apply_prompt_template(prompt, self.prompt_template) for prompt in prompts]
        else:
            raise ValueError(f"Unsupported prompt_template type: {type(self.prompt_template)}")
        
        vllm_prompts = []
        for prompt, image in zip(prompts, images, strict=True):
            vllm_prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
        
        # Generate responses for the batch
        outputs = self.model.generate(vllm_prompts, sampling_params=self.sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]
        
        return responses

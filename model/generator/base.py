import torch
from abc import abstractmethod, ABC


class AnswerGenerator(ABC):
    """Question generator."""

    def __init__(self, device, **kwargs):
        self.device = device

        self.temperature = kwargs.get("temperature", 0)
        self.top_p = kwargs.get("top_p", 1)
        self.top_k = kwargs.get("top_k", 1)
        self.num_beams = kwargs.get("num_beams", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)

        self.system_prompt = "You are a helpful assistant that answers questions based on the provided image."
        self.context_prompt = "Given the following context: "
        self.vqa_prompt = "Please keep your answer brief without additional explanations. \nAnswer: "
        self.captioning_prompt = "Please describe the image in one or two sentences. \nDecription: "

    @abstractmethod
    def _load_model(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method to load the model.")

    def prompt_for_vqa(self, question, context=None):
        if context is not None:
            context = f"{self.context_prompt}{context}\n"
        else:
            context = ""
        prompt = f"{context}{question}\n{self.vqa_prompt}"
        return prompt
    
    def prompt_for_captioning(self):
        return self.captioning_prompt

    @torch.no_grad()
    def mm_forword(self, prmopt, image, **kwargs):
        """Multimodal forward pass with single image."""
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @torch.no_grad()
    def batch_mm_forward(self, prompts, images, **kwargs):
        """Multimodal forward pass with batch of images."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def vqa_forword(
        self,
        questions,
        images,
        **kwargs
    ):
        """Answer the question.

        Args:
            question: The question to answer.
            image: The image to use.
            kwargs: Additional arguments.
        """
        contexts = kwargs.get("contexts", [None] * len(images))
        prompts = [self.prompt_for_vqa(q, c) for q, c in zip(questions, contexts)]
        return self.batch_mm_forward(prompts, images, **kwargs)

    def captioning_forword(
        self,
        images,
        **kwargs
    ):
        """Generate a caption for the image.

        Args:
            image: The image to use.
            kwargs: Additional arguments.
        """
        if not isinstance(images, list):
            images = [images]

        prompts = [self.prompt_for_captioning()] * len(images)
        return self.batch_mm_forward(prompts, images, **kwargs)

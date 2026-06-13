import torch
from abc import ABC, abstractmethod
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor


def load_processor_and_model(model_path, device='cpu', dtype=torch.float16):
    print(f"Loading model from {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    return processor, model


def load_tokenizer_and_model(model_path, device='cpu', dtype=torch.float16):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    return tokenizer, model


def load_blip2_model(model_path, device='cpu', dtype=torch.float16):
    from lavis.models import load_model_and_preprocess

    print(f"Loading model from {model_path}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_path, model_type="qm_retriever", is_eval=True, device=device)

    model.to(dtype)
    vis_processor = vis_processors["eval"]
    txt_processor = txt_processors["eval"]
    return model, vis_processor, txt_processor


class BaseEmbedding(ABC):
    def __init__(self, model_path, device='cpu', dtype=torch.float16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
    
    @abstractmethod
    def embed(self, *args, **kwargs):
        raise NotImplementedError("The embed method must be implemented in the subclass.")


class ClipEmbedding(BaseEmbedding):
    def __init__(self, model_path, device='cpu'):
        super().__init__(model_path, device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_path, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.embedding_dim = self.model.config.projection_dim

    def vision_embed(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.pixel_values.to(self.device, dtype=self.dtype)
        image_embedding = self.model.get_image_features(inputs)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        return image_embedding
    
    def text_embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        text_embedding = self.model.get_text_features(input_ids, attention_mask=attention_mask)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return text_embedding
    
    @torch.no_grad()
    def embed(self, data, modality='image', **kwargs):
        if modality == 'image':
            return self.vision_embed(data)
        elif modality == 'text':
            return self.text_embed(data)
        else:
            raise ValueError("Invalid mode. Use 'image' or 'text'.")


class EvaClipEmbedding(BaseEmbedding):
    def __init__(self, model_path, device='cpu'):
        super().__init__(model_path, device)
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.dtype, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.embedding_dim = self.model.config.projection_dim

    def vision_embed(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.pixel_values.to(self.device, dtype=self.dtype)
        image_embedding = self.model.encode_image(inputs)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        return image_embedding
    
    def text_embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        text_embedding = self.model.encode_text(input_ids, attention_mask=attention_mask)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return text_embedding
    
    @torch.no_grad()
    def embed(self, data, modality='image', **kwargs):
        if modality == 'image':
            return self.vision_embed(data)
        elif modality == 'text':
            return self.text_embed(data)
        else:
            raise ValueError("Invalid mode. Use 'image' or 'text'.")


class NomicVisionEmbedding(BaseEmbedding):
    def __init__(self, model_path, device='cpu'):
        super().__init__(model_path, device)
        self.processor, self.model = load_processor_and_model(self.model_path, self.device, self.dtype)
        self.embedding_dim = self.model.config.n_embd
    
    @torch.no_grad()
    def embed(self, images, **kwargs):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.pixel_values.to(self.device, dtype=self.dtype)

        image_embedding = self.model(inputs).last_hidden_state[:, 0]
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        
        return image_embedding


class NomicTextEmbedding(BaseEmbedding):
    def __init__(self, model_path, device='cpu', dtype=torch.float16):
        super().__init__(model_path, device, dtype)
        self.tokenizer, self.model = load_tokenizer_and_model(self.model_path, self.device, self.dtype)
        self.embedding_dim = self.model.config.n_embd
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def embed(self, texts, **kwargs):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        text_embedding = self.mean_pooling(outputs, inputs["attention_mask"])
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        
        return text_embedding


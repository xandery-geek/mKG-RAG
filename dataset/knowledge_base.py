import os
from tools.markdown import convert_to_markdown
from tools.utils import jsonl_generator


class KnowledgeBase:
    def __init__(self, kb_path):
        """
        Load knowledge base
        Args:
            kb_path: Path to the knowledge base file, str or list of str
        """
        self.kb_path = kb_path
        self._knowledge_base = None
    
    @property
    def knowledge_base(self):
        if self._knowledge_base is None:
            self._knowledge_base = self._load_knowledge_base(self.kb_path)
        return self._knowledge_base
        
    def _load_knowledge_base(self, kb_path):
        """
        Load knowledge base from a list of jsonl files
        """
        if isinstance(kb_path, str):
            kb_path = [kb_path]

        kb = {}
        for path in kb_path:
            if not os.path.exists(path):
                raise ValueError(f"Knowledge base file {path} does not exist")
            
            print(f"Loading knowledge base from {path}")
            josnl_data = jsonl_generator(path)
            for data in josnl_data:
                url = list(data.keys())[0]
                kb[url] = data[url]
        return kb
    
    def get_title(self, doc_key):
        if doc_key not in self.knowledge_base:
            print(f"Document {doc_key} not found in the knowledge base")
            return ""
        
        return self.knowledge_base[doc_key].get("title", "")

    def get_article(self, doc_key, **kwargs):
        doc = self.knowledge_base.get(doc_key, None)
        if doc is None:
            print(f"Document {doc_key} not found in the knowledge base")
            return ""

        return convert_to_markdown({doc_key: doc}, **kwargs)

    def get_section(self, doc_key, section_id, load_images=False):
        doc = self.knowledge_base.get(doc_key, None)
        if doc is None:
            print(f"Document {doc_key} not found in the knowledge base")
            return ""
        
        section_texts = doc["section_texts"]
        context = section_texts[int(section_id)]
        
        if load_images:
            image_urls = doc["image_urls"]
            image_sec_indices = doc["image_section_indices"]

            images = []
            for image_sec_idx, image_url in zip(image_sec_indices, image_urls):
                if int(section_id) == image_sec_idx:
                    images.append(image_url)
            return context, images
        else:
            return context

    def get_all_items(self):
        return self.knowledge_base.items()
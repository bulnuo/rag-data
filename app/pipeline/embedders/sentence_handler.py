
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch


class SentenceHandler():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model_name = '../../evaluators/sentence-transformers/all-MiniLM-L6-v2'    # if a local model
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_name)
        self.model = SentenceTransformer(self.model_name).to(self.device)
        self.tokenizer.eos_token = '[SEP]'
        self.tokenizer.pad_token = '[PAD]'
        self.embedding_input_size = 512
        self.vector_size = 384

        print(f"embedder_handlers.py: Intialized Sentence Embedder: {self.model_name}, {self.get_vector_size()}, {self.get_embedding_input_size()}")

    def embed_texts(self, dataset):
        print(f"embedder_handlers.py: Starting sentence embedding for {len(dataset)} characters...")
        embeddings = self.get_model().encode(dataset)
        return embeddings

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_embedding_input_size(self):
        return self.embedding_input_size

    def get_vector_size(self):
        return self.vector_size
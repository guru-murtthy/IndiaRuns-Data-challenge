import os
from sentence_transformers import SentenceTransformer

class CandidateEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', local_dir='sentence_transformer_model'):
        self.model_name = model_name
        self.local_dir = local_dir
        self.model = None

    def load_model(self):
        if os.path.exists(self.local_dir):
            self.model = SentenceTransformer(self.local_dir)
        else:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, texts, batch_size=256, show_progress_bar=False):
        if self.model is None:
            self.load_model()
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar, normalize_embeddings=True)

    def save_model(self):
        if self.model is None:
            self.load_model()
        self.model.save(self.local_dir)

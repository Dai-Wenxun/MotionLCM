import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class MldTextEncoder(nn.Module):

    def __init__(self, modelpath: str, last_hidden_state: bool = False) -> None:
        super().__init__()

        if 't5' in modelpath:
            self.text_model = SentenceTransformer(modelpath)
            self.tokenizer = self.text_model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
            self.text_model = AutoModel.from_pretrained(modelpath)

        self.max_length = self.tokenizer.model_max_length
        if "clip" in modelpath:
            self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            if last_hidden_state:
                self.name = "clip_hidden"
            else:
                self.name = "clip"
        elif "bert" in modelpath:
            self.name = "bert"
            self.text_encoded_dim = self.text_model.config.hidden_size
        elif 't5' in modelpath:
            self.name = 't5'
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, texts: list[str]) -> torch.Tensor:
        # get prompt text embeddings
        if self.name in ["clip", "clip_hidden"]:
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # split into max length Clip can handle
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
        elif self.name == "bert":
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        if self.name == "clip":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device))
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)
        elif self.name == "clip_hidden":
            # (batch_Size, seq_length , text_encoded_dim)
            text_embeddings = self.text_model.text_model(
                text_input_ids.to(self.text_model.device)).last_hidden_state
        elif self.name == "bert":
            # (batch_Size, seq_length , text_encoded_dim)
            text_embeddings = self.text_model(
                **text_inputs.to(self.text_model.device)).last_hidden_state
        elif self.name == 't5':
            text_embeddings = self.text_model.encode(texts, show_progress_bar=False, convert_to_tensor=True, batch_size=len(texts))
            text_embeddings = text_embeddings.unsqueeze(1)
        else:
            raise NotImplementedError(f"Model {self.name} not implemented")

        return text_embeddings

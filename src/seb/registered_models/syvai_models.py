"""
SyvAI embedding models registered in the benchmark.
"""

from datetime import date
from functools import partial
from typing import Any, Literal, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from seb.interfaces.model import LazyLoadEncoder, ModelMeta, SebModel
from seb.interfaces.task import Task
from seb.registries import models

from .normalize_to_ndarray import normalize_to_ndarray
from .sentence_transformer_models import silence_warnings_from_sentence_transformers


def task_to_instruction(task: Task) -> str:
    """Generate task-specific instructions for Qwen embedding models."""
    if task.task_type in ["STS"]:
        return "Retrieve semantically similar text"
    if task.task_type in ["Summarization"]:
        return "Given a news summary, retrieve other semantically similar summaries"
    if task.task_type in ["BitextMining"]:
        task_name_to_instruct: dict[str, str] = {
            "Bornholm Parallel": "Retrieve parallel sentences in Danish and Bornholmsk",
            "Norwegian courts": "Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
        }
        default_instruction = "Retrieve parallel sentences."
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Classification"]:
        task_name_to_instruct: dict[str, str] = {
            "Angry Tweets": "Classify Danish tweets by sentiment. (positive, negative, neutral)",
            "DKHate": "Classify Danish tweets based on offensiveness (offensive, not offensive)",
            "Da Political Comments": "Classify Danish political comments for sentiment",
            "DaLAJ": "Classify texts based on linguistic acceptability in Swedish",
            "LCC": "Classify texts based on sentiment",
            "Language Identification": "Classify texts based on language",
            "Massive Intent": "Given a user utterance as query, find the user intents",
            "Massive Scenario": "Given a user utterance as query, find the user scenarios",
            "NoReC": "Classify Norwegian reviews by sentiment",
            "SweReC": "Classify Swedish reviews by sentiment",
            "Norwegian parliament": "Classify parliament speeches in Norwegian based on political affiliation",
            "ScaLA": "Classify passages in Scandinavian Languages based on linguistic acceptability",
        }
        default_instruction = "Classify user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Clustering"]:
        task_name_to_instruct: dict[str, str] = {
            "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
            "VG Clustering": "Identify the categories (e.g. sports) of given articles in Norwegian",
            "SNL Clustering": "Identify categories in a Norwegian lexicon",
            "SwednClustering": "Identify news categories in Swedish passages",
        }
        default_instruction = "Identify categories in user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Reranking"]:
        return "Retrieve semantically similar passages."
    if task.task_type in ["Retrieval"]:
        task_name_to_instruct: dict[str, str] = {
            "Twitterhjerne": "Retrieve answers to questions asked in Danish tweets",
            "SwednRetrieval": "Given a Swedish news headline retrieve summaries or news articles",
            "TV2Nord Retrieval": "Given a summary of a Danish news article retrieve the corresponding news article",
            "DanFEVER": "Given a claim in Danish, retrieve documents that support the claim",
            "SNL Retrieval": "Given a lexicon headline in Norwegian, retrieve its article",
            "NorQuad": "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
            "SweFAQ": "Retrieve answers given questions in Swedish",
            "ArguAna": "Given a claim, find documents that refute the claim",
            "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
        }
        default_instruction = "Retrieve text based on user query."
        return task_name_to_instruct.get(task.name, default_instruction)
    return ""


class SyvaiEmbedNanoEncoder(SentenceTransformer):
    """
    A sentence transformer wrapper for SyvAI embed-nano-0925 that supports custom prompts.
    """

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task: Optional[Task] = None,  # noqa: ARG002
        encode_type: Literal["query", "passage"] = "passage",
        **kwargs: Any,
    ) -> np.ndarray:
        # Use the query prompt for queries, otherwise encode normally
        if encode_type == "query":
            # Use the query prompt as shown in the example
            prompt = "Instruct: Find text that answers the question.\n\nQuery: "
            emb = super().encode(sentences, prompt=prompt, batch_size=batch_size, **kwargs)
        else:
            prompt = "Instruct: Find similar text.\n\nQuery: "
            emb = super().encode(sentences, batch_size=batch_size, **kwargs)
        
        return normalize_to_ndarray(emb)

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
        sep = " "
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, encode_type="passage", **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, encode_type="query", **kwargs)


class QwenEmbeddingEncoder(SentenceTransformer):
    """
    A sentence transformer wrapper for Qwen3-Embedding models with flash attention support.
    """

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        task: Optional[Task] = None,
        encode_type: Literal["query", "passage"] = "passage",
        **kwargs: Any,
    ) -> np.ndarray:
        # Use task-specific instructions for better performance
        if task is not None:
            instruction = task_to_instruction(task)
            if instruction:
                # Prepend the instruction to each sentence
                instructed_sentences = [f"{instruction}\n\n{sentence}" for sentence in sentences]
                emb = super().encode(instructed_sentences, batch_size=batch_size, **kwargs)
            else:
                # Fall back to built-in prompt system for queries
                if encode_type == "query":
                    emb = super().encode(sentences, prompt_name="query", batch_size=batch_size, **kwargs)
                else:
                    emb = super().encode(sentences, batch_size=batch_size, **kwargs)
        else:
            # Use the built-in prompt system for Qwen models when no task context
            if encode_type == "query":
                emb = super().encode(sentences, prompt_name="query", batch_size=batch_size, **kwargs)
            else:
                emb = super().encode(sentences, batch_size=batch_size, **kwargs)
        
        return normalize_to_ndarray(emb)

    def encode_corpus(self, corpus: list[dict[str, str]], **kwargs: Any) -> np.ndarray:
        sep = " "
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, encode_type="passage", **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, encode_type="query", **kwargs)


def wrap_syvai_embed_nano(model_name: str, **kwargs: Any) -> SyvaiEmbedNanoEncoder:
    """Wrap the SyvAI embed-nano model with custom encoding logic."""
    silence_warnings_from_sentence_transformers()
    return SyvaiEmbedNanoEncoder(model_name, **kwargs)


def wrap_qwen_embedding(model_name: str, **kwargs: Any) -> QwenEmbeddingEncoder:
    """Wrap the Qwen embedding model with flash attention and custom encoding logic."""
    silence_warnings_from_sentence_transformers()
    
    # Enable flash attention and set padding side for better performance
    model_kwargs = kwargs.get("model_kwargs", {})
    model_kwargs.update({
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
        "torch_dtype": torch.float16  # Flash Attention requires fp16 or bf16
    })
    
    tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
    tokenizer_kwargs.update({
        "padding_side": "left"
    })
    
    kwargs.update({
        "model_kwargs": model_kwargs,
        "tokenizer_kwargs": tokenizer_kwargs
    })
    
    return QwenEmbeddingEncoder(model_name, **kwargs)


@models.register("embed-nano-0925")
def create_syvai_embed_nano_0925() -> SebModel:
    """Create the SyvAI embed-nano-0925 model."""
    hf_name = "syvai/embed-nano-0925"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["da", "nb", "sv", "en"],  # Based on the example showing Danish text
        open_source=True,
        embedding_size=1024,  # Verified from actual model output
        architecture="SentenceTransformer",
        release_date=date(2025, 9, 3),  # Based on the model name
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_syvai_embed_nano, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("Qwen3-Embedding-0.6B")
def create_qwen3_embedding_0_6b() -> SebModel:
    """Create the Qwen3-Embedding-0.6B model."""
    hf_name = "Qwen/Qwen3-Embedding-0.6B"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],  # English-focused model
        open_source=True,
        embedding_size=1024,  # Same architecture as SyvAI model
        architecture="SentenceTransformer",
        release_date=date(2024, 12, 1),  # Approximate release date
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_qwen_embedding, model_name=hf_name)),  # type: ignore
        meta=meta,
    )

@models.register("Qwen3-Embedding-4B")
def create_qwen3_embedding_4b() -> SebModel:
    """Create the Qwen3-Embedding-4B model."""
    hf_name = "Qwen/Qwen3-Embedding-4B"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],  # English-focused model
        open_source=True,
        embedding_size=1024,  # Same architecture as SyvAI model
        architecture="SentenceTransformer",
        release_date=date(2024, 12, 1),  # Approximate release date
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_qwen_embedding, model_name=hf_name)),  # type: ignore
        meta=meta,
    )


@models.register("Qwen3-Embedding-8B")
def create_qwen3_embedding_8b() -> SebModel:
    """Create the Qwen3-Embedding-8B model."""
    hf_name = "Qwen/Qwen3-Embedding-8B"
    meta = ModelMeta(
        name=hf_name.split("/")[-1],
        huggingface_name=hf_name,
        reference=f"https://huggingface.co/{hf_name}",
        languages=["en"],  # English-focused model
        open_source=True,
        embedding_size=1024,  # Same architecture as SyvAI model
        architecture="SentenceTransformer",
        release_date=date(2024, 12, 1),  # Approximate release date
    )
    return SebModel(
        encoder=LazyLoadEncoder(partial(wrap_qwen_embedding, model_name=hf_name)),  # type: ignore
        meta=meta,
    )
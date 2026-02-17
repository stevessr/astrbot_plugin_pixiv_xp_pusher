"""
AI Embedding 模块

使用 AstrBot Embedding Provider 计算文本向量，支持语义匹配。
借鉴 X 算法 Two-Tower Model 的理念。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.core.provider.provider import EmbeddingProvider


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""

    enabled: bool = False
    provider: str = "astrbot"
    dimensions: int = 256
    cache_ttl_days: int = 30  # 缓存天数
    semantic_weight: float = 0.3  # 语义匹配在最终分数中的权重


class Embedder:
    """
    Embedding 计算器

    支持：
    1. 用户画像 Embedding (基于 Top XP Tags)
    2. 作品 Embedding (基于作品 Tags)
    3. 相似度计算 (余弦相似度)
    """

    @staticmethod
    def _provider_model(provider: EmbeddingProvider) -> str:
        try:
            meta_model = provider.meta().model
            if meta_model:
                return str(meta_model)
        except Exception:
            pass
        try:
            return str(provider.get_model() or "")
        except Exception:
            return ""

    @property
    def model(self) -> str:
        if self._provider is None:
            return ""
        return self._provider_model(self._provider)

    def __init__(self, config: dict, provider: EmbeddingProvider | None = None):
        """
        初始化 Embedder

        Args:
            config: ai.embedding 配置块
        """
        self.enabled = config.get("enabled", False)
        self.provider = config.get("provider", "astrbot")
        self.dimensions = config.get("dimensions", 256)
        self.semantic_weight = config.get("semantic_weight", 0.3)
        self.cache_ttl_days = config.get("cache_ttl_days", 30)

        self._provider = provider

        if not self.enabled:
            return

        if self._provider is not None:
            try:
                meta = self._provider.meta()
                provider_id = meta.id
            except Exception:
                provider_id = "unknown"

            model_name = self.model or provider_id
            try:
                self.dimensions = int(self._provider.get_dim())
            except Exception:
                pass
            self.provider = "astrbot"
            logger.info(
                "Embedder initialized with AstrBot provider (provider_id=%s, model=%s, dim=%s)",
                provider_id,
                model_name,
                self.dimensions,
            )
            return

        logger.warning(
            "Embedding enabled but no AstrBot embedding provider available, disabling."
        )
        self.enabled = False

    async def embed_text(self, text: str) -> list[float] | None:
        """
        计算单个文本的 Embedding

        Args:
            text: 输入文本

        Returns:
            向量列表，失败返回 None
        """
        if not self.enabled or not text:
            return None

        try:
            if self._provider is None:
                return None
            vector = await self._provider.get_embedding(text)
            if not vector:
                return None
            return [float(v) for v in vector]

        except Exception as e:
            logger.error(f"Embedding 计算失败：{e}")
            return None

    async def embed_tags(
        self, tags: list[str], weights: list[float] | None = None
    ) -> list[float] | None:
        """
        将标签列表转换为 Embedding

        Args:
            tags: 标签列表
            weights: 可选的权重列表 (用于 XP Profile)

        Returns:
            向量列表
        """
        if not tags:
            return None

        # 构建文本表示
        if weights:
            # 带权重的标签：重复高权重标签以增加其影响
            # 简化处理：取 Top 10 并拼接
            tagged = sorted(zip(tags, weights), key=lambda x: x[1], reverse=True)[:10]
            text = ", ".join(t[0] for t in tagged)
        else:
            # 普通标签：直接拼接 (取前 10 个)
            text = ", ".join(tags[:10])

        return await self.embed_text(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """
        批量计算 Embedding

        Args:
            texts: 文本列表

        Returns:
            Embedding 列表 (失败的位置为 None)
        """
        if not self.enabled or not texts:
            return [None] * len(texts)

        try:
            if self._provider is None:
                return [None] * len(texts)
            vectors = await self._provider.get_embeddings(texts)
            normalized: list[list[float] | None] = [
                [float(v) for v in vec] if vec else None for vec in vectors
            ]
            if len(normalized) < len(texts):
                normalized.extend([None] * (len(texts) - len(normalized)))
            return normalized[: len(texts)]

        except Exception as e:
            try:
                lengths = [len(t) for t in texts]
                sample = texts[0][:200] if texts else ""
                logger.error(
                    "批量 Embedding 失败：%s | provider=%s model=%s batch=%d "
                    "len(min/avg/max)=%s sample=%s",
                    e,
                    self.provider,
                    self.model,
                    len(texts),
                    (
                        f"{min(lengths) if lengths else 0}/"
                        f"{(sum(lengths) // len(lengths)) if lengths else 0}/"
                        f"{max(lengths) if lengths else 0}"
                    ),
                    sample,
                )
            except Exception:
                logger.error(f"批量 Embedding 失败：{e}")
            return [None] * len(texts)

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1, vec2: 两个向量

        Returns:
            相似度 (-1 到 1)
        """
        if not vec1 or not vec2:
            return 0.0

        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a_val, b_val in zip(vec1, vec2):
            dot_product += a_val * b_val
            norm_a += a_val * a_val
            norm_b += b_val * b_val
        norm_a = math.sqrt(norm_a)
        norm_b = math.sqrt(norm_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def normalize_similarity(self, similarity: float) -> float:
        """
        将余弦相似度归一化到 0-1 范围

        Args:
            similarity: 余弦相似度 (-1 到 1)

        Returns:
            归一化分数 (0 到 1)
        """
        # 余弦相似度范围是 -1 到 1，我们映射到 0 到 1
        return (similarity + 1) / 2

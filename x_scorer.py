"""
X-Algorithm 风格评分引擎

借鉴 Twitter/X 推荐算法的多阶段排序架构：
1. Candidate Generation (已由 fetcher 完成)
2. Light Ranker - 快速粗排，计算基础特征分数
3. Heavy Ranker - 精排，融合多维信号
4. Final Mixer - 多样性控制、探索注入

参考：
- X Algorithm: https://github.com/twitter/the-algorithm
- OON Scorer (Out-of-Network)
- AuthorDiversityScorer
- Phoenix Scorer (LLM 精排)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from astrbot.api import logger

if TYPE_CHECKING:
    from pixiv_client import Illust


@dataclass
class FeatureWeights:
    """
    特征权重配置 (可实时调节)

    参考 X 算法的 Heavy Ranker 特征权重
    """

    # 核心匹配特征
    tag_match: float = 0.35  # XP 标签匹配
    semantic: float = 0.15  # 语义 Embedding 相似度
    popularity: float = 0.20  # 热度 (收藏数归一化)

    # 参与度信号 (Engagement Signals)
    like_ratio: float = 0.10  # 点赞率 (likes / views)
    recency: float = 0.10  # 时效性 (越新越高)
    author_affinity: float = 0.10  # 画师亲和度 (关注/历史互动)

    # 来源加成 (Source Boost) - 乘法因子
    source_boost: dict = field(
        default_factory=lambda: {
            "xp_search": 1.0,
            "subscription": 1.15,
            "ranking": 0.95,
            "related": 1.10,
            "engagement_artists": 1.20,
        }
    )

    # 多样性控制
    author_diversity_decay: float = 0.7  # 同画师衰减因子
    author_diversity_floor: float = 0.1  # 衰减下限

    # 探索与随机
    exploration_ratio: float = 0.1  # 探索比例
    shuffle_noise: float = 0.05  # 随机噪声幅度


@dataclass
class ScoredIllust:
    """带分数的作品包装"""

    illust: Illust
    # 各维度分数
    tag_score: float = 0.0
    semantic_score: float = 0.0
    popularity_score: float = 0.0
    like_ratio_score: float = 0.0
    recency_score: float = 0.0
    author_affinity_score: float = 0.0
    source_multiplier: float = 1.0

    # 阶段分数
    light_rank_score: float = 0.0
    heavy_rank_score: float = 0.0
    final_score: float = 0.0

    # 元信息
    is_exploration: bool = False


class XScorer:
    """
    X-Algorithm 风格的多阶段评分引擎

    Pipeline:
        Candidates -> Light Ranker -> Heavy Ranker -> Final Mixer -> Output
    """

    def __init__(
        self,
        weights: FeatureWeights | dict | None = None,
        embedder=None,
        ai_scorer=None,
        xp_profile: dict[str, float] | None = None,
        negative_profile: dict[str, float] | None = None,
        subscribed_artists: set[int] | None = None,
        daily_limit: int = 20,
        max_per_artist: int = 3,
    ):
        # 权重配置
        if isinstance(weights, dict):
            self.weights = FeatureWeights(**weights)
        elif weights is None:
            self.weights = FeatureWeights()
        else:
            self.weights = weights

        # 可选组件
        self.embedder = embedder
        self.ai_scorer = ai_scorer

        # 用户画像
        self.xp_profile = xp_profile or {}
        self.negative_profile = negative_profile or {}
        self.subscribed_artists = subscribed_artists or set()

        # 限制
        self.daily_limit = daily_limit
        self.max_per_artist = max_per_artist

        # 缓存
        self._max_bookmark = 1
        self._user_embedding = None

    async def score_pipeline(
        self,
        candidates: list[Illust],
        user_id: int = 0,
    ) -> list[Illust]:
        """
        执行完整评分管道

        Args:
            candidates: 候选作品列表
            user_id: 用户 ID (用于 Embedding 缓存)

        Returns:
            排序后的作品列表 (带 match_score 属性)
        """
        if not candidates:
            return []

        logger.info(f"[XScorer] 开始评分管道，候选数: {len(candidates)}")

        # 1. 初始化统计量
        self._max_bookmark = max(c.bookmark_count for c in candidates) or 1

        # 2. 包装为 ScoredIllust
        scored = [ScoredIllust(illust=c) for c in candidates]

        # 3. Light Ranker (快速粗排)
        scored = await self._light_rank(scored)
        logger.debug(f"[XScorer] Light Ranker 完成，剩余: {len(scored)}")

        # 4. Heavy Ranker (精排)
        scored = await self._heavy_rank(scored, user_id)
        logger.debug(f"[XScorer] Heavy Ranker 完成")

        # 5. Final Mixer (多样性 + 探索)
        result = self._final_mix(scored)
        logger.info(f"[XScorer] 管道完成，输出: {len(result)}")

        # 6. 附加分数到原始对象
        for item in result:
            item.illust.match_score = item.final_score

        return [item.illust for item in result]

    async def _light_rank(self, scored: list[ScoredIllust]) -> list[ScoredIllust]:
        """
        Light Ranker: 快速粗排

        计算低成本特征:
        - Tag 匹配度
        - 热度归一化
        - 时效性
        - 来源加成
        """
        w = self.weights

        for item in scored:
            illust = item.illust

            # 1. Tag 匹配分数
            item.tag_score = self._calc_tag_score(illust)

            # 2. 热度分数 (对数归一化，避免大 V 垄断)
            if illust.bookmark_count > 0:
                log_bookmark = math.log1p(illust.bookmark_count)
                log_max = math.log1p(self._max_bookmark)
                item.popularity_score = log_bookmark / log_max if log_max > 0 else 0
            else:
                item.popularity_score = 0.0

            # 3. 时效性分数 (指数衰减，半衰期 7 天)
            item.recency_score = self._calc_recency_score(illust)

            # 4. 来源加成
            source = getattr(illust, "source", "xp_search")
            item.source_multiplier = w.source_boost.get(source, 1.0)

            # 5. 画师亲和度
            if illust.user_id in self.subscribed_artists:
                item.author_affinity_score = 1.0
            else:
                item.author_affinity_score = 0.0

            # Light Rank 分数 (快速版，不含 Embedding)
            item.light_rank_score = (
                w.tag_match * item.tag_score
                + w.popularity * item.popularity_score
                + w.recency * item.recency_score
                + w.author_affinity * item.author_affinity_score
            ) * item.source_multiplier

        # 按 light_rank_score 排序，取 Top-N 进入 Heavy Ranker
        scored.sort(key=lambda x: x.light_rank_score, reverse=True)

        # Heavy Ranker 最多处理 200 个候选
        heavy_limit = min(200, len(scored))
        return scored[:heavy_limit]

    async def _heavy_rank(
        self, scored: list[ScoredIllust], user_id: int
    ) -> list[ScoredIllust]:
        """
        Heavy Ranker: 精排

        计算高成本特征:
        - 语义 Embedding 相似度
        - Like Ratio (点赞率)
        - AI Scorer (LLM 精排)
        """
        w = self.weights

        # 1. 计算语义分数 (如果启用 Embedder)
        if self.embedder and self.embedder.enabled and self.xp_profile:
            await self._compute_semantic_scores(scored, user_id)

        # 2. 计算 Like Ratio (view/bookmark 比率)
        for item in scored:
            illust = item.illust
            # Pixiv 没有 view count，用 bookmark 作为 engagement 代理
            # 这里用 bookmark / total_bookmarks 的相对值
            if hasattr(illust, "total_view") and illust.total_view > 0:
                item.like_ratio_score = min(
                    illust.bookmark_count / illust.total_view, 1.0
                )
            else:
                # Fallback: 使用热度分数
                item.like_ratio_score = item.popularity_score * 0.5

        # 3. Heavy Rank 分数
        for item in scored:
            item.heavy_rank_score = (
                w.tag_match * item.tag_score
                + w.semantic * item.semantic_score
                + w.popularity * item.popularity_score
                + w.like_ratio * item.like_ratio_score
                + w.recency * item.recency_score
                + w.author_affinity * item.author_affinity_score
            ) * item.source_multiplier

        # 4. AI Scorer 二次精排 (可选)
        if self.ai_scorer and self.ai_scorer.enabled:
            await self._apply_ai_scorer(scored)

        # 按 heavy_rank_score 排序
        scored.sort(key=lambda x: x.heavy_rank_score, reverse=True)
        return scored

    def _final_mix(self, scored: list[ScoredIllust]) -> list[ScoredIllust]:
        """
        Final Mixer: 多样性控制与探索注入

        1. Author Diversity Decay (同画师分数衰减)
        2. Exploration (随机混入潜力作品)
        3. 硬性去重 (max_per_artist)
        4. 随机噪声打散
        """
        w = self.weights

        # 1. Author Diversity Decay
        author_position: dict[int, int] = {}
        for item in scored:
            uid = item.illust.user_id
            pos = author_position.get(uid, 0)

            # 衰减公式: (1 - floor) * decay^pos + floor
            decay_mult = (1.0 - w.author_diversity_floor) * (
                w.author_diversity_decay**pos
            ) + w.author_diversity_floor

            item.final_score = item.heavy_rank_score * decay_mult
            author_position[uid] = pos + 1

        # 2. 添加随机噪声
        if w.shuffle_noise > 0:
            for item in scored:
                noise = random.uniform(-w.shuffle_noise, w.shuffle_noise)
                item.final_score += noise

        # 3. 重新排序
        scored.sort(key=lambda x: x.final_score, reverse=True)

        # 4. 硬性去重 + Exploration
        main_count = int(self.daily_limit * (1 - w.exploration_ratio))
        explore_count = self.daily_limit - main_count

        # 主流选择 (top candidates, 遵守 max_per_artist)
        author_count: dict[int, int] = {}
        main_result: list[ScoredIllust] = []
        remaining: list[ScoredIllust] = []

        for item in scored:
            uid = item.illust.user_id
            cnt = author_count.get(uid, 0)
            if cnt < self.max_per_artist:
                if len(main_result) < main_count:
                    main_result.append(item)
                    author_count[uid] = cnt + 1
                else:
                    remaining.append(item)
            else:
                remaining.append(item)

        # 5. Exploration: 从后半部分随机抽取
        explore_result: list[ScoredIllust] = []
        if explore_count > 0 and remaining:
            # 候选池：排名 main_count 之后的作品
            explore_pool = remaining[: main_count * 2]
            if len(explore_pool) >= explore_count:
                explore_result = random.sample(explore_pool, explore_count)
            else:
                explore_result = explore_pool[:]

            for item in explore_result:
                item.is_exploration = True

            logger.info(f"[XScorer] 探索混入: {len(explore_result)} 个")

        # 6. 合并并打散
        final_result = main_result + explore_result
        if explore_result:
            random.shuffle(final_result)

        return final_result[: self.daily_limit]

    def _calc_tag_score(self, illust: Illust) -> float:
        """
        计算 Tag 匹配分数

        改进算法：
        1. 归一化权重累加
        2. Top 20% 高权重标签加成
        3. 匹配数量对数奖励
        4. 负向画像惩罚
        """
        if not illust.tags or not self.xp_profile:
            return 0.0

        from utils import normalize_tag

        sorted_weights = sorted(self.xp_profile.values(), reverse=True)
        max_weight = sorted_weights[0] if sorted_weights else 1.0
        top_threshold = (
            sorted_weights[len(sorted_weights) // 5]
            if len(sorted_weights) >= 5
            else max_weight * 0.8
        )

        total_score = 0.0
        matched_count = 0
        high_weight_matches = 0
        negative_penalty = 0.0

        for tag in illust.tags:
            normalized_tag = normalize_tag(tag)

            # 正向匹配
            weight = self.xp_profile.get(normalized_tag)
            if weight is None:
                weight = self.xp_profile.get(tag.lower())

            if weight is not None:
                total_score += weight
                matched_count += 1
                if weight >= top_threshold:
                    high_weight_matches += 1

            # 负向匹配
            if self.negative_profile:
                neg_weight = self.negative_profile.get(normalized_tag, 0)
                if neg_weight > 0:
                    negative_penalty += neg_weight * 0.5

        if matched_count == 0:
            if negative_penalty > 0:
                return max(-negative_penalty / (max_weight + 1), -0.5)
            return 0.0

        # 基础分
        base_score = total_score / (matched_count * max_weight) if max_weight > 0 else 0

        # 匹配数量奖励
        quantity_bonus = min(math.log(1 + matched_count) / math.log(6), 0.3)

        # 高权重奖励
        quality_bonus = min(high_weight_matches * 0.05, 0.2)

        # 负向惩罚
        penalty = negative_penalty / (max_weight + 1) if max_weight > 0 else 0

        final = base_score + quantity_bonus + quality_bonus - penalty
        return max(min(final, 1.0), 0.0)

    def _calc_recency_score(self, illust: Illust) -> float:
        """
        计算时效性分数 (指数衰减)

        半衰期: 7 天
        公式: e^(-days / decay_days)
        """
        if not illust.create_date:
            return 0.5

        now = datetime.now(timezone.utc)
        create_date = illust.create_date
        if create_date.tzinfo is None:
            create_date = create_date.replace(tzinfo=timezone.utc)

        days_old = (now - create_date).days
        decay_days = 7.0  # 半衰期

        score = math.exp(-days_old / decay_days)
        return max(min(score, 1.0), 0.0)

    async def _compute_semantic_scores(
        self, scored: list[ScoredIllust], user_id: int
    ) -> None:
        """计算语义 Embedding 相似度分数"""
        import database as db

        try:
            import hashlib

            # 获取或计算用户 Embedding
            profile_str = str(sorted(list(self.xp_profile.items())[:20]))
            profile_hash = hashlib.md5(profile_str.encode()).hexdigest()[:16]

            cached = await db.get_user_embedding(user_id)
            if cached and cached[1] == profile_hash:
                user_emb = cached[0]
            else:
                top_tags = [
                    t
                    for t, _ in sorted(
                        self.xp_profile.items(), key=lambda x: x[1], reverse=True
                    )[:15]
                ]
                user_emb = await self.embedder.embed_tags(top_tags)
                if user_emb:
                    await db.save_user_embedding(
                        user_id, user_emb, self.embedder.model, profile_hash
                    )

            if not user_emb:
                return

            # 批量获取作品 Embedding
            illust_ids = [item.illust.id for item in scored]
            cached_embs = await db.get_illust_embeddings_batch(illust_ids)

            # 计算缓存命中的分数
            uncached = []
            for item in scored:
                emb = cached_embs.get(item.illust.id)
                if emb:
                    sim = self.embedder.cosine_similarity(user_emb, emb)
                    item.semantic_score = self.embedder.normalize_similarity(sim)
                else:
                    uncached.append(item)

            # 批量计算未缓存的
            if uncached:
                texts = [", ".join(item.illust.tags[:10]) for item in uncached]
                embeddings = await self.embedder.embed_batch(texts)

                to_save = []
                for i, item in enumerate(uncached):
                    emb = embeddings[i] if i < len(embeddings) else None
                    if emb:
                        to_save.append((item.illust.id, emb, self.embedder.model))
                        sim = self.embedder.cosine_similarity(user_emb, emb)
                        item.semantic_score = self.embedder.normalize_similarity(sim)

                if to_save:
                    await db.save_illust_embeddings_batch(to_save)

        except Exception as e:
            logger.warning(f"[XScorer] 语义匹配失败: {e}")

    async def _apply_ai_scorer(self, scored: list[ScoredIllust]) -> None:
        """应用 AI Scorer (LLM 精排)"""
        import database as db

        try:
            candidates = [item.illust for item in scored[: self.ai_scorer.max_candidates]]
            if len(candidates) < 5:
                return

            recent_likes = await db.get_recent_liked_tags(limit=5)
            recent_dislikes = await db.get_recent_disliked_tags(limit=5)

            ai_scores = await self.ai_scorer.score_candidates(
                candidates, self.xp_profile, recent_likes, recent_dislikes
            )

            if ai_scores:
                # 混合 AI 分数
                for item in scored:
                    ai_score = ai_scores.get(item.illust.id)
                    if ai_score is not None:
                        # 加权混合
                        item.heavy_rank_score = (
                            1 - self.ai_scorer.score_weight
                        ) * item.heavy_rank_score + self.ai_scorer.score_weight * ai_score

                logger.info(f"[XScorer] AI 精排已应用: {len(ai_scores)} 个")

        except Exception as e:
            logger.warning(f"[XScorer] AI 精排失败: {e}")


def create_scorer_from_config(
    config: dict,
    embedder=None,
    ai_scorer=None,
    xp_profile: dict | None = None,
    negative_profile: dict | None = None,
    subscribed_artists: set[int] | None = None,
) -> XScorer:
    """
    从配置创建 XScorer 实例

    Args:
        config: 插件配置字典
        embedder: Embedder 实例
        ai_scorer: AIScorer 实例
        xp_profile: 用户 XP 画像
        negative_profile: 负向画像
        subscribed_artists: 关注画师集合

    Returns:
        XScorer 实例
    """
    filter_cfg = config.get("filter", {})
    algorithm_cfg = filter_cfg.get("algorithm", {})

    # 构建权重配置
    weights_cfg = algorithm_cfg.get("weights", {})
    weights = FeatureWeights(
        tag_match=weights_cfg.get("tag_match", 0.35),
        semantic=weights_cfg.get("semantic", 0.15),
        popularity=weights_cfg.get("popularity", 0.20),
        like_ratio=weights_cfg.get("like_ratio", 0.10),
        recency=weights_cfg.get("recency", 0.10),
        author_affinity=weights_cfg.get("author_affinity", 0.10),
        source_boost=algorithm_cfg.get(
            "source_boost",
            {
                "xp_search": 1.0,
                "subscription": 1.15,
                "ranking": 0.95,
                "related": 1.10,
                "engagement_artists": 1.20,
            },
        ),
        author_diversity_decay=algorithm_cfg.get("author_diversity_decay", 0.7),
        author_diversity_floor=algorithm_cfg.get("author_diversity_floor", 0.1),
        exploration_ratio=algorithm_cfg.get("exploration_ratio", 0.1),
        shuffle_noise=algorithm_cfg.get("shuffle_noise", 0.05),
    )

    return XScorer(
        weights=weights,
        embedder=embedder,
        ai_scorer=ai_scorer,
        xp_profile=xp_profile,
        negative_profile=negative_profile,
        subscribed_artists=subscribed_artists,
        daily_limit=filter_cfg.get("daily_limit", 20),
        max_per_artist=filter_cfg.get("max_per_artist", 3),
    )

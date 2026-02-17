"""
AI 行为预测评分器 (LLM 精排)

借鉴 X 算法 Phoenix Scorer 的理念，使用 LLM 预测用户对作品的喜好程度。
仅在候选数量较少时启用（默认 < 50），作为粗排后的精排步骤。

Prompt 设计参考：
- 提供用户的 Top XP Tags
- 提供近期反馈历史（喜欢/不喜欢的标签）
- 让 LLM 输出喜爱概率分数
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from utils import download_image_with_referer, get_pixiv_cat_url, save_persistent_img

from astrbot.api import logger
from astrbot.api.provider import Provider


@dataclass
class AIScoreConfig:
    """AI 评分配置"""

    enabled: bool = False
    provider: str = "astrbot"
    model: str = ""
    max_candidates: int = 50  # 超过此数量时跳过 AI 评分
    score_weight: float = 0.3  # AI 分数在最终排序中的权重
    vision_enabled: bool = False
    image_max_bytes: int = 2000000
    proxy_url: str = ""


class AIScorer:
    """
    AI 行为预测评分器

    使用 LLM 对候选作品进行二次精排，预测用户喜好。
    """

    PROMPT_TEMPLATE = """你是推荐系统评分器。根据用户偏好，给每个作品打"喜爱概率"分（0.0-1.0）。

用户偏好：
- 最爱 Tag: {top_tags}
- 最近喜欢的作品标签：{recent_likes}
- 最近不喜欢的作品标签：{recent_dislikes}

候选作品：
{candidates}

返回 JSON 数组，格式：[{{"id": 123, "score": 0.85}}]
只返回 JSON，不要解释。根据标签与用户偏好的匹配程度评分。"""

    def __init__(self, config: dict, provider: Provider | None = None):
        """
        初始化 AI 评分器

        Args:
            config: ai.scorer 配置块
        """
        self.enabled = config.get("enabled", False)
        self.provider = config.get("provider", "astrbot")
        self.model = config.get("model") or ""
        self.max_candidates = config.get("max_candidates", 50)
        self.score_weight = config.get("score_weight", 0.3)
        self.vision_enabled = bool(config.get("vision_enabled", False))
        self.image_max_bytes = int(config.get("image_max_bytes", 2000000))
        self.proxy_url = config.get("proxy_url", "")

        self._provider = provider
        if not self.enabled:
            return

        if self._provider is not None:
            if not self.model:
                try:
                    self.model = (
                        self._provider.get_model() or self._provider.meta().model or ""
                    )
                except Exception:
                    self.model = ""
            provider_id = "unknown"
            try:
                provider_id = self._provider.meta().id
            except Exception:
                pass
            logger.info(
                "AI Scorer initialized with AstrBot provider (provider_id=%s, model=%s)",
                provider_id,
                self.model or "default",
            )
            return

        logger.warning(
            "AI Scorer enabled but no AstrBot chat provider available, disabling."
        )
        self.enabled = False

    async def score_candidates(
        self,
        candidates: list,
        xp_profile: dict[str, float],
        recent_likes: list[str] = None,
        recent_dislikes: list[str] = None,
    ) -> dict[int, float]:
        """
        对候选作品进行 AI 评分

        Args:
            candidates: 候选作品列表 (Illust)
            xp_profile: 用户 XP 画像 {tag: weight}
            recent_likes: 近期喜欢的作品标签
            recent_dislikes: 近期不喜欢的作品标签

        Returns:
            {illust_id: ai_score} 映射
        """
        if not self.enabled or not candidates:
            return {}

        # 超过阈值时跳过
        if len(candidates) > self.max_candidates:
            logger.debug(
                f"候选数量 {len(candidates)} 超过阈值 {self.max_candidates}，跳过 AI 评分"
            )
            return {}

        try:
            # 构建 Prompt
            top_tags = [
                t
                for t, _ in sorted(
                    xp_profile.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            # 构建候选列表描述
            candidate_lines = []
            for illust in candidates:
                tags_str = ", ".join(illust.tags[:8])  # 只取前 8 个标签
                candidate_lines.append(f"- ID: {illust.id}, Tags: [{tags_str}]")

            prompt = self.PROMPT_TEMPLATE.format(
                top_tags=", ".join(top_tags),
                recent_likes=", ".join(recent_likes[:5]) if recent_likes else "无",
                recent_dislikes=", ".join(recent_dislikes[:5])
                if recent_dislikes
                else "无",
                candidates="\n".join(candidate_lines),
            )

            image_payloads: list[tuple[int, str]] = []
            if self.vision_enabled:
                image_payloads = await self._build_image_payloads(candidates)
                if image_payloads:
                    order = ", ".join(str(iid) for iid, _ in image_payloads)
                    prompt += f"\n\n候选图片顺序如下（与候选列表对应）：{order}"

            if self._provider is None:
                raise RuntimeError("AstrBot chat provider is required")
            image_urls = [
                self._to_provider_image_url(data_url) for _, data_url in image_payloads
            ]
            response = await self._provider.text_chat(
                prompt=prompt,
                image_urls=image_urls or None,
                system_prompt="你是推荐系统评分器，只返回 JSON 数组，不要解释。",
                model=self.model or None,
                temperature=0.3,
                max_tokens=1000,
                persist=False,
            )
            content = (response.completion_text or "").strip()

            # 解析 JSON
            # 尝试提取 JSON 数组
            if "```" in content:
                # 去除 markdown 代码块
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            scores = json.loads(content)

            # 转换为字典
            result = {}
            for item in scores:
                illust_id = item.get("id")
                score = item.get("score", 0.5)
                if illust_id:
                    result[illust_id] = max(0.0, min(1.0, float(score)))

            logger.info(f"AI 评分完成：{len(result)}/{len(candidates)} 个作品")
            return result

        except Exception as e:
            logger.error(f"AI 评分失败：{e}")
            return {}

    @staticmethod
    def _to_provider_image_url(data_url: str) -> str:
        prefix = "data:image/jpeg;base64,"
        if data_url.startswith(prefix):
            return f"base64://{data_url[len(prefix) :]}"
        return data_url

    async def _build_image_payloads(self, candidates: list) -> list[tuple[int, str]]:
        if not candidates:
            return []
        import aiohttp

        results: list[tuple[int, str]] = []
        async with aiohttp.ClientSession() as session:
            for illust in candidates:
                url = None
                if getattr(illust, "image_urls", None):
                    url = illust.image_urls[0]
                if not url:
                    url = get_pixiv_cat_url(illust.id, 0)
                try:
                    data = await download_image_with_referer(
                        session, url, proxy=self.proxy_url or None
                    )
                    if not data:
                        continue
                    if len(data) > self.image_max_bytes:
                        continue
                    save_persistent_img(data, url=url)
                    b64 = base64.b64encode(data).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{b64}"
                    results.append((illust.id, data_url))
                except Exception:
                    continue
        return results

    def blend_scores(
        self, base_scores: dict[int, float], ai_scores: dict[int, float]
    ) -> dict[int, float]:
        """
        混合基础分数和 AI 分数

        公式：final = (1 - weight) * base + weight * ai

        Args:
            base_scores: 基础分数 {id: score}
            ai_scores: AI 分数 {id: score}

        Returns:
            混合后的分数
        """
        result = {}
        for illust_id, base_score in base_scores.items():
            ai_score = ai_scores.get(illust_id)
            if ai_score is not None:
                final = (
                    1 - self.score_weight
                ) * base_score + self.score_weight * ai_score
            else:
                final = base_score
            result[illust_id] = final
        return result

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiohttp
from database import cache_illust, init_db, mark_pushed
from fetcher import ContentFetcher
from filter import ContentFilter
from pixiv_client import PixivClient
from profiler import XPProfiler
from utils import download_image_with_referer, get_pixiv_cat_url

from astrbot.api import logger
from astrbot.core.utils.io import save_temp_img

try:
    from astrbot.api import AstrBotConfig
    from astrbot.api.event import AstrMessageEvent, MessageChain, filter
    from astrbot.api.star import Context, Star
except Exception:  # pragma: no cover - allow standalone CLI usage
    AstrBotConfig = None
    AstrMessageEvent = None
    MessageChain = None
    Context = None
    Star = None
    filter = None

if TYPE_CHECKING:
    from pixiv_client import Illust


async def retry_async(
    coro_func,
    *args,
    max_retries: int = 3,
    delay: float = 5.0,
    backoff: float = 2.0,
    **kwargs,
):
    """
    通用异步重试函数

    Args:
        coro_func: 要执行的异步函数
        max_retries: 最大重试次数
        delay: 初始延迟秒数
        backoff: 延迟倍增系数

    Returns:
        函数返回值，或在所有重试失败后返回 None
    """
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"操作失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，{current_delay:.1f}s 后重试..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"操作最终失败 (已重试 {max_retries} 次): {e}")

    return None


# 全局运行锁，防止任务并发
_task_lock = asyncio.Lock()


async def setup_notifiers(
    config: dict,
    client: PixivClient,
    profiler: XPProfiler,
    sync_client: PixivClient = None,
):
    raise RuntimeError(
        "Non-AstrBot notifiers have been removed; use notifiers_factory."
    )


async def setup_services(config: dict, notifiers_factory=None):
    """初始化全局服务 (DB, Client, Profiler, Notifiers)"""
    await init_db()

    # 公共网络配置
    network_cfg = config.get("network", {})
    pixiv_cfg = config.get("pixiv", {})
    proxy_url = network_cfg.get("proxy_url")

    client_kwargs = {
        "requests_per_minute": network_cfg.get("requests_per_minute", 60),
        "random_delay": tuple(network_cfg.get("random_delay", [1.0, 3.0])),
        "max_concurrency": network_cfg.get("max_concurrency", 5),
        "proxy_url": proxy_url,
    }

    # 主客户端 (用于搜索、排行榜等高风险操作)
    main_client = PixivClient(
        refresh_token=pixiv_cfg.get("refresh_token"), **client_kwargs
    )
    await main_client.login()

    # 同步客户端 (用于获取收藏、关注动态等低风险操作)
    sync_token = pixiv_cfg.get("sync_token")
    if sync_token:
        sync_client = PixivClient(refresh_token=sync_token, **client_kwargs)
        await sync_client.login()
        logger.info("✅ 已启用同步专用 Token (sync_token)")
    else:
        sync_client = main_client  # 回退到主客户端
        logger.info("未配置 sync_token，收藏同步将使用主 Token")

    # Init Profiler (使用 sync_client，只读操作)
    profiler_cfg = config.get("profiler", {})
    profiler = XPProfiler(
        client=sync_client,  # 使用同步客户端获取收藏
        stop_words=profiler_cfg.get("stop_words"),
        discovery_rate=profiler_cfg.get("discovery_rate", 0.1),
        time_decay_days=profiler_cfg.get("time_decay_days", 180),
        ai_config=profiler_cfg.get("ai"),
        saturation_threshold=profiler_cfg.get("saturation_threshold", 0.5),
        ai_provider=config.get("_astrbot_chat_provider"),
    )

    # Init Notifiers (使用 main_client 用于下载图片等，sync_client 用于 on_action 回调)
    if notifiers_factory:
        notifiers = await notifiers_factory(config, main_client, profiler, sync_client)
    else:
        notifiers = await setup_notifiers(config, main_client, profiler, sync_client)

    # 返回双客户端
    return main_client, sync_client, profiler, notifiers


async def main_task(
    config: dict,
    client: PixivClient,
    profiler: XPProfiler,
    notifiers: list,
    sync_client: PixivClient = None,
):
    """
    执行一次完整的推送任务 (依赖外部服务)

    Args:
        client: 主客户端 (用于搜索、排行榜、下载)
        sync_client: 同步客户端 (用于获取关注动态，可选)
    """
    # 如果未传入 sync_client，使用 main_client
    if sync_client is None:
        sync_client = client

    if _task_lock.locked():
        logger.info("⏳ 推送任务正在运行中，本次触发已跳过或排队")

    async with _task_lock:
        logger.info("=== 开始推送任务 ===")

    try:
        # 1. 构建/更新 XP 画像
        profiler_cfg = config.get("profiler", {})

        await profiler.build_profile(
            user_id=config["pixiv"]["user_id"],
            scan_limit=profiler_cfg.get("scan_limit", 500),
            include_private=profiler_cfg.get("include_private", True),
        )

        top_tags = await profiler.get_top_tags(profiler_cfg.get("top_n", 20))
        logger.info(f"Top XP Tags: {[t[0] for t in top_tags[:10]]}")

        if config.get(
            "test"
        ):  # Test mode skip heavy DB load if possible, but we need it for xp_profile
            pass

        # 获取完整的 XP Profile 用于匹配度计算
        import database as db_module

        xp_profile = await db_module.get_xp_profile()

        # 2. 获取内容
        fetcher_cfg = config.get("fetcher", {})

        # 1.5 获取关注列表（使用 sync_client，低风险操作）
        following_ids = set()
        pixiv_uid = config.get("pixiv", {}).get("user_id", 0)
        if pixiv_uid:
            try:
                following_ids = await sync_client.fetch_following(user_id=pixiv_uid)
            except Exception as e:
                logger.warning(f"获取关注列表失败：{e}")

        manual_subs = set(fetcher_cfg.get("subscribed_artists") or [])
        all_subs = list(following_ids | manual_subs)
        logger.info(
            f"有效关注画师数：{len(all_subs)} (API 获取：{len(following_ids)}, 手动：{len(manual_subs)})"
        )

        # ContentFetcher: 搜索/排行榜用 client，订阅检查用 sync_client
        fetcher = ContentFetcher(
            client=client,
            sync_client=sync_client,  # 新增：同步客户端
            bookmark_threshold=fetcher_cfg.get(
                "bookmark_threshold", {"search": 1000, "subscription": 0}
            ),
            date_range_days=fetcher_cfg.get("date_range_days", 7),
            subscribed_artists=list(manual_subs),
            discovery_rate=profiler_cfg.get("discovery_rate", 0.1),
            ranking_config=fetcher_cfg.get("ranking"),
            dynamic_threshold_config=fetcher_cfg.get(
                "dynamic_threshold"
            ),  # 动态阈值配置
            search_limit=fetcher_cfg.get("search_limit", 50),  # 搜索数量限制 (默认 50)
        )

        # 执行 Discovery (Search + Ranking + Subs)
        top_tags = await profiler.get_top_tags(
            profiler_cfg.get("top_n", 20)
        )  # Re-get is cheap

        # 执行 Discovery (Search + Ranking + Subs) -> MAB Scheduled
        top_tags = await profiler.get_top_tags(
            profiler_cfg.get("top_n", 20)
        )  # Re-get is cheap

        all_illusts = await fetcher.fetch_content(
            xp_tags=top_tags, total_limit=fetcher_cfg.get("discovery_limit", 200)
        )
        logger.info(f"共获取 {len(all_illusts)} 个候选作品")

        # 3. 过滤
        filter_cfg = config.get("filter", {})
        match_cfg = fetcher_cfg.get("match_score", {})

        astrbot_chat_provider = config.get("_astrbot_chat_provider")
        astrbot_embedding_provider = config.get("_astrbot_embedding_provider")

        # 初始化可选的 Embedder (AI 语义匹配)
        embedder = None
        ai_cfg = config.get("ai", {})
        embedding_cfg = ai_cfg.get("embedding", {})
        if embedding_cfg.get("enabled", False):
            try:
                from embedder import Embedder

                embedder = Embedder(
                    embedding_cfg, embedding_provider=astrbot_embedding_provider
                )
                if embedder.enabled:
                    logger.info(f"已启用 AI 语义匹配 (model={embedder.model})")
            except Exception as e:
                logger.warning(f"Embedder 初始化失败：{e}")

        # 初始化可选的 AIScorer (LLM 精排)
        ai_scorer = None
        scorer_cfg = ai_cfg.get("scorer", {})
        if scorer_cfg.get("enabled", False):
            try:
                from ai_scorer import AIScorer

                # 支持复用 profiler.ai 的 API 配置
                if scorer_cfg.get("use_profiler_api", True):
                    profiler_ai_cfg = config.get("profiler", {}).get("ai", {})
                    # 合并配置：scorer 优先，缺失的从 profiler.ai 继承
                    merged_cfg = {
                        "enabled": scorer_cfg.get("enabled", False),
                        "provider": scorer_cfg.get("provider")
                        or profiler_ai_cfg.get("provider", "openai"),
                        "api_key": scorer_cfg.get("api_key")
                        or profiler_ai_cfg.get("api_key", ""),
                        "base_url": scorer_cfg.get("base_url")
                        or profiler_ai_cfg.get("base_url", ""),
                        "model": scorer_cfg.get("model")
                        or profiler_ai_cfg.get("model", "gpt-4o-mini"),
                        "max_candidates": scorer_cfg.get("max_candidates", 50),
                        "score_weight": scorer_cfg.get("score_weight", 0.3),
                    }
                    ai_scorer = AIScorer(merged_cfg, provider=astrbot_chat_provider)
                else:
                    ai_scorer = AIScorer(scorer_cfg, provider=astrbot_chat_provider)

                if ai_scorer.enabled:
                    logger.info(f"已启用 AI 精排评分 (model={ai_scorer.model})")
            except Exception as e:
                logger.warning(f"AIScorer 初始化失败：{e}")

        content_filter = ContentFilter(
            blacklist_tags=filter_cfg.get("blacklist_tags"),
            daily_limit=filter_cfg.get("daily_limit", 20),
            exclude_ai=filter_cfg.get("exclude_ai", True),
            min_match_score=match_cfg.get("min_threshold", 0.0),
            match_weight=match_cfg.get("weight_in_sort", 0.5),
            max_per_artist=filter_cfg.get("max_per_artist", 3),
            subscribed_artists=all_subs,
            artist_boost=filter_cfg.get("artist_boost", 0.3),
            min_create_days=filter_cfg.get("min_create_days", 0),
            r18_mode=filter_cfg.get("r18_mode", False),
            # 新增：借鉴 X 算法的增强选项
            author_diversity=filter_cfg.get("author_diversity"),
            source_boost=filter_cfg.get("source_boost"),
            embedder=embedder,  # 可选的语义匹配
            ai_scorer=ai_scorer,  # 可选的 LLM 精排
            # 多样性增强
            shuffle_factor=filter_cfg.get("shuffle_factor", 0.0),
            exploration_ratio=filter_cfg.get("exploration_ratio", 0.0),
        )

        pixiv_uid = config.get("pixiv", {}).get("user_id", 0)
        filtered = await content_filter.filter(
            all_illusts, xp_profile=xp_profile, user_id=pixiv_uid
        )
        logger.info(f"过滤后 {len(filtered)} 个作品")

        # 4. 推送
        if notifiers and filtered:
            try:
                # 缓存作品信息 (包含来源归因)
                for illust in filtered:
                    await cache_illust(
                        illust.id,
                        illust.tags,
                        illust.user_id,
                        illust.user_name,
                        source=illust.source,
                    )

                all_sent_ids = set()
                for notifier in notifiers:
                    try:
                        sent_ids = await notifier.send(filtered)
                        all_sent_ids.update(sent_ids)
                    except Exception as e:
                        logger.error(f"推送器 {type(notifier).__name__} 发送失败：{e}")

                if all_sent_ids:
                    # 记录推送历史
                    filtered_map = {ill.id: ill for ill in filtered}
                    for pid in all_sent_ids:
                        if pid in filtered_map:
                            illust = filtered_map[pid]
                            source = getattr(illust, "source", "unknown")
                            await mark_pushed(pid, source)

                            # 更新 MAB 策略统计 (Total Count)
                            if source in [
                                "xp_search",
                                "subscription",
                                "ranking",
                                "related",
                                "engagement_artists",
                            ]:
                                await db_module.update_strategy_stats(
                                    source, is_success=False
                                )

                    # 将消息 ID 写入数据库缓存（用于连锁推送引用）
                    for notifier in notifiers:
                        if hasattr(notifier, "_message_illust_map"):
                            for (
                                msg_id,
                                illust_id,
                            ) in notifier._message_illust_map.items():
                                if illust_id in all_sent_ids:
                                    await db_module.set_chain_meta(
                                        illust_id, chain_depth=0, chain_msg_id=msg_id
                                    )

                    logger.info(
                        f"推送完成：{len(all_sent_ids)}/{len(filtered)} 个作品成功"
                    )
                else:
                    logger.error("没有任何作品被成功推送")

                # 5. AI 错误报警
                ai_errors = profiler.ai_processor.occurred_errors
                if ai_errors:
                    err_count = len(ai_errors)
                    err_id = ai_errors[0]
                    msg = f"⚠️ 警告：本次任务有 {err_count} 批 Tag AI 优化失败。\n已自动记录并降级处理。"
                    buttons = [("🔄 重试修复", f"retry_ai:{err_id}")]
                    logger.warning(f"AI 优化失败 {err_count} 次，发送警告")

                    for notifier in notifiers:
                        if hasattr(notifier, "send_text"):
                            try:
                                await notifier.send_text(msg, buttons)
                            except Exception as e:
                                logger.debug(f"AI 错误提示发送失败：{e}")
            except Exception as e:
                logger.error(f"推送过程出错：{e}")
        elif not filtered:
            logger.info("无新作品可推送")
        else:
            logger.warning("未配置推送器")

    except Exception as e:
        logger.error(f"任务执行出错：{e}", exc_info=True)

    logger.info("=== 推送任务结束 ===")


async def run_once(config: dict, notifiers_factory=None):
    """立即执行一次"""
    main_client, sync_client, profiler, notifiers = await setup_services(
        config, notifiers_factory=notifiers_factory
    )

    # Run-once 是 Fire-and-Forget 行为

    try:
        await main_task(config, main_client, profiler, notifiers, sync_client)
    finally:
        await main_client.close()
        # 如果 sync_client 是独立实例，也需要关闭
        if sync_client is not main_client:
            await sync_client.close()
        for n in notifiers or []:
            if hasattr(n, "close"):
                try:
                    await n.close()
                except Exception as e:
                    logger.debug(f"关闭推送器失败：{e}")


async def daily_report_task(config: dict, notifiers: list, profiler=None):
    """每日维护任务：生成日报 + 数据清理 + AI 标签刷新

    设计原则：
    - 每个步骤独立 try/except，即使某一步失败，其他步骤仍可继续
    - 网络相关操作（AI、发送）使用 retry_async 自动重试
    """
    logger.info("📊 开始执行每日维护任务...")

    maintenance_summary = []
    lines = ["📊 **每日 XP 日报**\n"]

    # ========== 1. 生成日报 (Top Tags + MAB Stats) ==========
    try:
        from database import get_all_strategy_stats, get_top_xp_tags

        top_tags = await get_top_xp_tags(10)
        stats = await get_all_strategy_stats()

        if top_tags:
            lines.append("🎯 **Top 10 XP 标签**")
            for i, (tag, weight) in enumerate(top_tags[:10], 1):
                lines.append(f"  {i}. `{tag}` ({weight:.1f})")
            lines.append("")

        if stats:
            lines.append("📈 **MAB 策略表现**")
            strategy_names = {
                "search": "XP 搜索",
                "xp_search": "XP 搜索",
                "subscription": "订阅",
                "ranking": "排行榜",
            }
            for strategy, data in stats.items():
                name = strategy_names.get(strategy, strategy)
                rate_pct = data["rate"] * 100
                lines.append(
                    f"  • {name}: {data['success']}/{data['total']} ({rate_pct:.1f}%)"
                )
    except Exception as e:
        logger.error(f"生成日报统计失败：{e}")
        maintenance_summary.append(f"⚠️ 日报统计失败：{e}")

    # ========== 2. 同步屏蔽标签到 XP 画像 ==========
    try:
        from database import sync_blocked_tags_to_xp

        blocked_removed = await sync_blocked_tags_to_xp()
        if blocked_removed > 0:
            maintenance_summary.append(
                f"🚫 从画像中移除 {blocked_removed} 个已屏蔽标签"
            )
            logger.info(f"已从 XP 画像中移除 {blocked_removed} 个屏蔽标签")
    except Exception as e:
        logger.error(f"同步屏蔽标签失败：{e}")
        maintenance_summary.append(f"⚠️ 同步屏蔽标签失败：{e}")

    # ========== 3. AI 标签增量处理 (带重试) ==========
    if profiler and hasattr(profiler, "ai_processor") and profiler.ai_processor.enabled:
        try:
            from database import get_uncached_tags

            uncached_tags = await get_uncached_tags(limit=200)
            if uncached_tags:
                logger.info(f"发现 {len(uncached_tags)} 个未处理标签，启动 AI 清洗...")

                async def _ai_process():
                    return await profiler.ai_processor.process_tags(uncached_tags)

                result = await retry_async(_ai_process, max_retries=3, delay=10.0)
                if result:
                    valid_tags, mapping = result
                    maintenance_summary.append(
                        f"🤖 AI 清洗 {len(uncached_tags)} 个标签 → {len(valid_tags)} 个有效"
                    )
                    logger.info(
                        f"AI 清洗完成：{len(valid_tags)}/{len(uncached_tags)} 有效"
                    )
                else:
                    maintenance_summary.append("⚠️ AI 清洗失败 (已重试)")
        except Exception as e:
            logger.error(f"AI 清洗失败：{e}")
            maintenance_summary.append(f"⚠️ AI 清洗失败：{e}")

    # ========== 4. 清理旧推送历史 ==========
    try:
        from database import cleanup_old_sent_history

        old_removed = await cleanup_old_sent_history(days=30)
        if old_removed > 0:
            maintenance_summary.append(f"🗑️ 清理 {old_removed} 条过期推送记录")
            logger.info(f"已清理 {old_removed} 条 30 天前的推送历史")
    except Exception as e:
        logger.error(f"清理推送历史失败：{e}")
        maintenance_summary.append(f"⚠️ 清理推送历史失败：{e}")

    # ========== 5. 清理旧作品缓存 ==========
    try:
        from database import cleanup_old_illust_cache

        cache_removed = await cleanup_old_illust_cache(days=60)
        if cache_removed > 0:
            maintenance_summary.append(f"🗑️ 清理 {cache_removed} 条过期作品缓存")
            logger.info(f"已清理 {cache_removed} 条 60 天前的作品缓存")
    except Exception as e:
        logger.error(f"清理作品缓存失败：{e}")
        maintenance_summary.append(f"⚠️ 清理作品缓存失败：{e}")

    # ========== 6. 添加维护摘要到日报 ==========
    if maintenance_summary:
        lines.append("")
        lines.append("🛠️ **维护记录**")
        for item in maintenance_summary:
            lines.append(f"  {item}")

    report_msg = "\n".join(lines)

    # ========== 7. 发送日报 (带重试) ==========
    async def _send_report():
        for n in notifiers:
            if hasattr(n, "send_text"):
                await n.send_text(report_msg)
                return True
        return False

    result = await retry_async(_send_report, max_retries=5, delay=30.0, backoff=2.0)
    if not result:
        logger.error("发送日报最终失败")

    logger.info("✅ 每日维护任务完成")


async def run_scheduler(
    config: dict, run_immediately: bool = False, notifiers_factory=None
):
    """启动调度器 (Daemon Mode)"""
    main_client, sync_client, profiler, notifiers = await setup_services(
        config, notifiers_factory=notifiers_factory
    )

    if run_immediately:
        logger.info("🚀 正在立即执行首次任务...")
        asyncio.create_task(
            main_task(config, main_client, profiler, notifiers, sync_client)
        )

    scheduler = AsyncIOScheduler()
    scheduler_cfg = config.get("scheduler", {})
    coalesce = scheduler_cfg.get("coalesce", True)

    # 获取调度配置 (优先读取数据库)
    from database import get_state

    db_cron = await get_state("schedule_cron")
    config_cron = config.get("scheduler", {}).get("cron", "0 20 * * *")

    schedule_str = db_cron if db_cron else config_cron

    # 将 scheduler 注入到 config 中以便 callback 访问
    config["scheduler"] = scheduler

    # 支持多个时间点
    # 逻辑优化：
    # 1. 先尝试将整个字符串作为一个 Cron，如果成功则认为是一个任务 (解决 "0 12,21 * * *" 被误拆的问题)
    # 2. 如果失败，再尝试用逗号分割 (兼容旧的多任务写法 "0 12 * * *, 0 21 * * *")

    cron_list = []

    # 尝试解析整体
    try:
        CronTrigger.from_crontab(schedule_str.strip())
        cron_list = [schedule_str.strip()]
        logger.info(f"识别为单一定时任务：{schedule_str}")
    except ValueError:
        # 整体解析失败，尝试分割
        potential_crons = [c.strip() for c in schedule_str.split(",") if c.strip()]
        valid_crons = []
        for c in potential_crons:
            try:
                CronTrigger.from_crontab(c)
                valid_crons.append(c)
            except ValueError:
                logger.warning(f"忽略无效的 Cron 表达式片段：{c}")

        if valid_crons:
            cron_list = valid_crons
            logger.info(f"识别为 {len(cron_list)} 个独立定时任务")
        else:
            # 如果分割也全错，那可能就是整体写错了，保留整体让后面报错
            cron_list = [schedule_str]

    for i, cron_expr in enumerate(cron_list):
        try:
            scheduler.add_job(
                main_task,
                CronTrigger.from_crontab(cron_expr),
                args=[config, main_client, profiler, notifiers, sync_client],
                id=f"push_job_{i}",
                coalesce=coalesce,
                misfire_grace_time=3600,
            )
            logger.info(f"已添加定时任务 #{i + 1}: {cron_expr}")
        except Exception as e:
            logger.error(f"添加定时任务失败 ({cron_expr}): {e}")

    # 每日维护任务 (日报 + 清理)
    daily_cron = scheduler_cfg.get("daily_report_cron", "0 0 * * *")  # 默认每天 00:00
    try:
        scheduler.add_job(
            daily_report_task,
            CronTrigger.from_crontab(daily_cron),
            args=[config, notifiers, profiler],  # 传入 profiler 以支持 AI 清洗
            id="daily_report_job",
            coalesce=True,
            misfire_grace_time=3600,
        )
        logger.info(f"已添加每日维护任务：{daily_cron}")
    except Exception as e:
        logger.error(f"添加每日维护任务失败：{e}")

    scheduler.start()
    logger.info(f"调度器已启动，共 {len(cron_list)} 个推送任务 + 1 个每日维护任务")

    try:
        stop_event = asyncio.Event()
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        scheduler.shutdown()
        raise
    finally:
        await main_client.close()
        # 如果 sync_client 是独立实例，也需要关闭
        if sync_client is not main_client:
            await sync_client.close()
        for n in notifiers or []:
            if hasattr(n, "close"):
                try:
                    await n.close()
                except Exception as e:
                    logger.debug(f"关闭推送器失败：{e}")


def _apply_test_overrides(config: dict) -> None:
    config.setdefault("profiler", {})["scan_limit"] = 10
    config["profiler"]["discovery_rate"] = 0
    config.setdefault("fetcher", {})["bookmark_threshold"] = {
        "search": 0,
        "subscription": 0,
    }
    config.setdefault("fetcher", {})["discovery_limit"] = 1
    config["fetcher"]["ranking"] = {"modes": ["day"], "limit": 1}
    config["test"] = True


def _get_list(cfg: dict, key: str, default: list):
    value = cfg.get(key, default)
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    return [value]


def _build_config_from_astrbot(plugin_cfg: "AstrBotConfig") -> dict:
    """Build Pixiv-XP-Pusher config from AstrBot plugin config."""
    pixiv_cfg = plugin_cfg.get("pixiv", {}) or {}
    profiler_cfg = plugin_cfg.get("profiler", {}) or {}
    profiler_ai_cfg = profiler_cfg.get("ai", {}) or {}
    ai_cfg = plugin_cfg.get("ai", {}) or {}
    scheduler_cfg = plugin_cfg.get("scheduler", {}) or {}
    filter_cfg = plugin_cfg.get("filter", {}) or {}
    fetcher_cfg = plugin_cfg.get("fetcher", {}) or {}
    network_cfg = plugin_cfg.get("network", {}) or {}

    config = {
        "pixiv": {
            "user_id": pixiv_cfg.get("user_id", 0),
            "refresh_token": pixiv_cfg.get("refresh_token", ""),
            "sync_token": pixiv_cfg.get("sync_token", ""),
        },
        "strategies": _get_list(
            plugin_cfg,
            "strategies",
            ["xp_search", "related", "ranking", "subscription"],
        ),
        "profiler": {
            "ai": {
                "enabled": profiler_ai_cfg.get("enabled", False),
                "provider": profiler_ai_cfg.get("provider", "openai"),
                "api_key": profiler_ai_cfg.get("api_key", ""),
                "base_url": profiler_ai_cfg.get("base_url", ""),
                "model": profiler_ai_cfg.get("model", "gpt-4o-mini"),
                "concurrency": profiler_ai_cfg.get("concurrency", 10),
                "batch_size": profiler_ai_cfg.get("batch_size", 200),
                "filter_meaningless": profiler_ai_cfg.get("filter_meaningless", True),
                "merge_synonyms": profiler_ai_cfg.get("merge_synonyms", True),
            },
            "scan_limit": profiler_cfg.get("scan_limit", 1000),
            "discovery_rate": profiler_cfg.get("discovery_rate", 0.1),
            "time_decay_days": profiler_cfg.get("time_decay_days", 180),
            "saturation_threshold": profiler_cfg.get("saturation_threshold", 0.5),
            "top_n": profiler_cfg.get("top_n", 20),
            "include_private": profiler_cfg.get("include_private", True),
            "stop_words": _get_list(
                profiler_cfg,
                "stop_words",
                ["original", "manga", "pixiv", "illustration"],
            ),
        },
        "ai": {
            "embedding": ai_cfg.get("embedding", {}),
            "scorer": ai_cfg.get("scorer", {}),
        },
        "scheduler": {
            "cron": scheduler_cfg.get("cron", "0 12 * * *"),
            "coalesce": scheduler_cfg.get("coalesce", True),
            "daily_report_cron": scheduler_cfg.get("daily_report_cron", "0 0 * * *"),
        },
        "filter": {
            "match_score": {
                "min_threshold": filter_cfg.get("min_match_score", 0.0),
                "weight_in_sort": filter_cfg.get("match_weight", 0.6),
            },
            "daily_limit": filter_cfg.get("daily_limit", 20),
            "exclude_ai": filter_cfg.get("exclude_ai", True),
            "max_per_artist": filter_cfg.get("max_per_artist", 3),
            "artist_boost": filter_cfg.get("artist_boost", 0.3),
            "min_create_days": filter_cfg.get("min_create_days", 0),
            "r18_mode": filter_cfg.get("r18_mode", "mixed"),
            "shuffle_factor": filter_cfg.get("shuffle_factor", 0.0),
            "exploration_ratio": filter_cfg.get("exploration_ratio", 0.0),
            "blacklist_tags": _get_list(filter_cfg, "blacklist_tags", []),
            "author_diversity": filter_cfg.get("author_diversity", {}),
            "source_boost": filter_cfg.get("source_boost", {}),
        },
        "fetcher": {
            "bookmark_threshold": {
                "search": fetcher_cfg.get("bookmark_threshold", {}).get("search", 1000),
                "subscription": fetcher_cfg.get("bookmark_threshold", {}).get(
                    "subscription", 0
                ),
            },
            "subscribed_artists": _get_list(fetcher_cfg, "subscribed_artists", []),
            "date_range_days": fetcher_cfg.get("date_range_days", 7),
            "dynamic_threshold": fetcher_cfg.get("dynamic_threshold", {}),
            "search_limit": fetcher_cfg.get("search_limit", 50),
            "ranking": fetcher_cfg.get(
                "ranking",
                {"enabled": True, "modes": ["day", "week", "month"], "limit": 100},
            ),
            "match_score": fetcher_cfg.get("match_score", {}),
            "mab_limits": fetcher_cfg.get(
                "mab_limits", {"min_quota": 0.2, "max_quota": 0.6}
            ),
        },
        "network": {
            "requests_per_minute": network_cfg.get("requests_per_minute", 60),
            "random_delay": network_cfg.get("random_delay", [1.0, 3.0]),
            "max_concurrency": network_cfg.get("max_concurrency", 5),
            "proxy_url": network_cfg.get("proxy_url", ""),
        },
        "astrbot": {
            "use_astrbot_providers": plugin_cfg.get("use_astrbot_providers", False),
            "chat_provider_id": plugin_cfg.get("chat_provider_id", ""),
            "embedding_provider_id": plugin_cfg.get("embedding_provider_id", ""),
        },
        "notifier": {
            "max_pages": plugin_cfg.get("max_pages", 10),
            "multi_page_mode": plugin_cfg.get("multi_page_mode", "cover_link"),
        },
    }

    return config


def _build_push_sessions(plugin_cfg: "AstrBotConfig") -> list[str]:
    return _get_list(plugin_cfg, "push_sessions", [])


class AstrBotNotifier:
    """Send Pixiv messages through AstrBot's builtin proactive channels."""

    def __init__(
        self,
        context: "Context",
        sessions: list[str],
        max_pages: int = 10,
        multi_page_mode: str = "cover_link",
        use_pixiv_cat: bool = True,
        proxy_url: str | None = None,
    ) -> None:
        self.context = context
        self.sessions = sessions
        self.max_pages = max_pages
        self.multi_page_mode = multi_page_mode
        self.use_pixiv_cat = use_pixiv_cat
        self.proxy_url = proxy_url
        self._session: aiohttp.ClientSession | None = None

    def _pick_image_urls(self, illust: "Illust") -> list[str]:
        if not illust.page_count:
            return []
        limit = max(1, min(self.max_pages, illust.page_count))
        if self.use_pixiv_cat:
            return [get_pixiv_cat_url(illust.id, i) for i in range(limit)]
        if illust.image_urls:
            return illust.image_urls[:limit]
        return []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _download_to_file(self, url: str) -> str | None:
        try:
            session = await self._get_session()
            data = await download_image_with_referer(
                session, url, proxy=self.proxy_url
            )
            return save_temp_img(data)
        except Exception as e:
            logger.warning(f"下载图片失败：{e}")
            return None

    def format_message(self, illust: "Illust") -> str:
        tags = ", ".join(illust.tags[:20]) if illust.tags else "N/A"
        r18 = "R18" if illust.is_r18 else "SAFE"
        return (
            f"🎨 {illust.title} (#{illust.id})\n"
            f"👤 {illust.user_name} ({illust.user_id})\n"
            f"🔖 {tags}\n"
            f"⭐ {illust.bookmark_count} | 👀 {illust.view_count} | {r18}\n"
            f"🔗 https://www.pixiv.net/artworks/{illust.id}"
        )

    def handle_feedback(self, illust_id: int, action: str) -> bool:
        return False

    async def _send_chain(self, chain: "MessageChain") -> None:
        for session in self.sessions:
            await self.context.send_message(session, chain)

    async def send(self, illusts: list["Illust"]) -> list[int]:
        if not illusts or not self.sessions:
            return []

        success_ids = []
        for illust in illusts:
            chain = MessageChain()
            chain.message(self.format_message(illust))

            image_urls = self._pick_image_urls(illust)
            if image_urls:
                urls = (
                    image_urls
                    if self.multi_page_mode == "multi_image"
                    else [image_urls[0]]
                )
                for url in urls:
                    path = await self._download_to_file(url)
                    if path:
                        chain.file_image(path)
                    else:
                        chain.url_image(url)

            await self._send_chain(chain)
            success_ids.append(illust.id)
        return success_ids

    async def send_text(
        self, text: str, buttons: list[tuple[str, str]] | None = None
    ) -> bool:
        if not self.sessions:
            return False
        chain = MessageChain()
        chain.message(text)
        await self._send_chain(chain)
        return True

    async def push_illusts(
        self,
        illusts: list["Illust"],
        message_prefix: str = "",
        reply_to_message_id: int | None = None,
    ) -> dict[int, int | None]:
        sent_map: dict[int, int | None] = {}
        if not illusts:
            return sent_map
        for illust in illusts:
            chain = MessageChain()
            if message_prefix:
                chain.message(f"{message_prefix}\n")
            chain.message(self.format_message(illust))

            image_urls = self._pick_image_urls(illust)
            if image_urls:
                path = await self._download_to_file(image_urls[0])
                if path:
                    chain.file_image(path)
                else:
                    chain.url_image(image_urls[0])
            await self._send_chain(chain)
            sent_map[illust.id] = None
        return sent_map

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


if Star is not None:

    class PixivXPPusherPlugin(Star):
        """Pixiv XP Pusher plugin wrapper for AstrBot."""

        def __init__(self, context: Context, config: AstrBotConfig) -> None:
            super().__init__(context)
            self.context = context
            self.plugin_config = config
            self.plugin_dir = Path(__file__).parent
            self._scheduler_task: asyncio.Task | None = None
            self._run_once_lock = asyncio.Lock()
            self._last_error: str | None = None

            self._auto_start = bool(self.plugin_config.get("auto_start", True))
            self._run_immediately = bool(
                self.plugin_config.get("run_immediately", False)
            )
            self._test_mode = bool(self.plugin_config.get("test_mode", False))
            self._use_pixiv_cat = bool(self.plugin_config.get("use_pixiv_cat", True))

        async def initialize(self):
            if self._auto_start:
                started, message = await self._start_scheduler()
                if not started:
                    logger.error(f"AstrBot: 自动启动失败：{message}")

        async def terminate(self):
            await self._stop_scheduler()

        def _load_runtime_config(self) -> dict | None:
            config = _build_config_from_astrbot(self.plugin_config)
            astrbot_cfg = config.get("astrbot", {})
            if astrbot_cfg.get("use_astrbot_providers"):
                chat_provider_id = astrbot_cfg.get("chat_provider_id") or ""
                embedding_provider_id = astrbot_cfg.get("embedding_provider_id") or ""

                chat_provider = None
                if chat_provider_id:
                    chat_provider = self.context.get_provider_by_id(chat_provider_id)
                    if not chat_provider:
                        # 允许用“提供商源 ID/显示名”前缀匹配
                        candidates = [
                            p
                            for p in self.context.get_all_providers()
                            if p.meta().id.startswith(f"{chat_provider_id}/")
                        ]
                        if len(candidates) == 1:
                            chat_provider = candidates[0]
                            logger.info(
                                f"AstrBot: 已将 LLM Provider ID 解析为 {chat_provider.meta().id}"
                            )
                        elif candidates:
                            logger.warning(
                                f"AstrBot: LLM Provider ID '{chat_provider_id}' 匹配到多个候选："
                                + ", ".join(p.meta().id for p in candidates)
                            )
                        else:
                            logger.warning(
                                f"AstrBot: 未找到 LLM Provider ID: {chat_provider_id}"
                            )
                    elif not hasattr(chat_provider, "text_chat"):
                        logger.warning(
                            f"AstrBot: Provider ID {chat_provider_id} 不是可用的 LLM Provider"
                        )
                        chat_provider = None
                else:
                    try:
                        chat_provider = self.context.get_using_provider()
                    except Exception as exc:
                        logger.warning(f"AstrBot: 获取默认 LLM Provider 失败：{exc}")

                embedding_provider = None
                if embedding_provider_id:
                    embedding_provider = self.context.get_provider_by_id(
                        embedding_provider_id
                    )
                    if not embedding_provider:
                        candidates = [
                            p
                            for p in self.context.get_all_embedding_providers()
                            if p.meta().id.startswith(f"{embedding_provider_id}/")
                        ]
                        if len(candidates) == 1:
                            embedding_provider = candidates[0]
                            logger.info(
                                "AstrBot: 已将 Embedding Provider ID 解析为 "
                                f"{embedding_provider.meta().id}"
                            )
                        elif candidates:
                            logger.warning(
                                f"AstrBot: Embedding Provider ID '{embedding_provider_id}' 匹配到多个候选："
                                + ", ".join(p.meta().id for p in candidates)
                            )
                        else:
                            logger.warning(
                                f"AstrBot: 未找到 Embedding Provider ID: {embedding_provider_id}"
                            )
                    elif not hasattr(embedding_provider, "get_embedding"):
                        logger.warning(
                            f"AstrBot: Provider ID {embedding_provider_id} 不是可用的 Embedding Provider"
                        )
                        embedding_provider = None
                else:
                    providers = self.context.get_all_embedding_providers()
                    if providers:
                        embedding_provider = providers[0]
                    else:
                        logger.warning("AstrBot: 未配置 Embedding Provider")

                config["_astrbot_chat_provider"] = chat_provider
                config["_astrbot_embedding_provider"] = embedding_provider
            if self._test_mode:
                _apply_test_overrides(config)
            return config

        async def _start_scheduler(
            self, run_immediately: bool | None = None
        ) -> tuple[bool, str]:
            if self._scheduler_task and not self._scheduler_task.done():
                return False, "Scheduler already running."

            config = self._load_runtime_config()
            if not config:
                return False, "Config not found or empty."

            immediate = (
                self._run_immediately if run_immediately is None else run_immediately
            )
            sessions = _build_push_sessions(self.plugin_config)
            if not sessions:
                return False, "No push sessions configured."

            async def _notifier_factory(_, __, ___, ____):
                return [
                    AstrBotNotifier(
                        context=self.context,
                        sessions=sessions,
                        max_pages=config.get("notifier", {}).get("max_pages", 10),
                        multi_page_mode=config.get("notifier", {}).get(
                            "multi_page_mode", "cover_link"
                        ),
                        use_pixiv_cat=self._use_pixiv_cat,
                        proxy_url=config.get("network", {}).get("proxy_url"),
                    )
                ]

            task = asyncio.create_task(
                run_scheduler(
                    config,
                    run_immediately=immediate,
                    notifiers_factory=_notifier_factory,
                )
            )
            task.add_done_callback(self._on_scheduler_done)
            self._scheduler_task = task
            return True, "Scheduler started."

        async def _stop_scheduler(self) -> tuple[bool, str]:
            if not self._scheduler_task or self._scheduler_task.done():
                return False, "Scheduler not running."

            self._scheduler_task.cancel()
            try:
                await asyncio.wait_for(self._scheduler_task, timeout=15)
            except asyncio.CancelledError:
                self._scheduler_task = None
            except asyncio.TimeoutError:
                return False, "Scheduler stop timeout."
            except Exception:
                self._scheduler_task = None
                raise
            else:
                self._scheduler_task = None
            return True, "Scheduler stopped."

        def _on_scheduler_done(self, task: asyncio.Task) -> None:
            try:
                task.result()
            except asyncio.CancelledError:
                logger.info("AstrBot: scheduler task cancelled.")
            except Exception as exc:
                self._last_error = str(exc)
                logger.error(f"AstrBot: scheduler crashed: {exc}", exc_info=True)

        async def _run_once_background(self) -> tuple[bool, str]:
            config = self._load_runtime_config()
            if not config:
                return False, "Config not found or empty."

            sessions = _build_push_sessions(self.plugin_config)
            if not sessions:
                return False, "No push sessions configured."

            async def _notifier_factory(_, __, ___, ____):
                return [
                    AstrBotNotifier(
                        context=self.context,
                        sessions=sessions,
                        max_pages=config.get("notifier", {}).get("max_pages", 10),
                        multi_page_mode=config.get("notifier", {}).get(
                            "multi_page_mode", "cover_link"
                        ),
                        use_pixiv_cat=self._use_pixiv_cat,
                    )
                ]

            async def _run():
                async with self._run_once_lock:
                    await run_once(config, notifiers_factory=_notifier_factory)

            asyncio.create_task(_run())
            return True, "Run-once task started."

        @filter.command_group("pixivxp", alias={"pixiv", "xp"})
        def pixivxp(self):
            """Pixiv-XP-Pusher control group."""
            pass

        @pixivxp.command("status")
        async def status(self, event: AstrMessageEvent):
            running = bool(self._scheduler_task and not self._scheduler_task.done())
            last_error = f" Last error: {self._last_error}" if self._last_error else ""
            yield event.plain_result(
                f"Pixiv-XP-Pusher status: {'running' if running else 'stopped'}.{last_error}"
            )

        @filter.permission_type(filter.PermissionType.ADMIN)
        @pixivxp.command("start")
        async def start(self, event: AstrMessageEvent):
            started, message = await self._start_scheduler()
            yield event.plain_result(message)

        @filter.permission_type(filter.PermissionType.ADMIN)
        @pixivxp.command("stop")
        async def stop(self, event: AstrMessageEvent):
            stopped, message = await self._stop_scheduler()
            yield event.plain_result(message)

        @filter.permission_type(filter.PermissionType.ADMIN)
        @pixivxp.command("once")
        async def once(self, event: AstrMessageEvent):
            ok, message = await self._run_once_background()
            yield event.plain_result(message if ok else f"Run-once failed: {message}")

        @filter.permission_type(filter.PermissionType.ADMIN)
        @pixivxp.command("reload")
        async def reload(self, event: AstrMessageEvent):
            await self._stop_scheduler()
            started, message = await self._start_scheduler()
            yield event.plain_result(message)

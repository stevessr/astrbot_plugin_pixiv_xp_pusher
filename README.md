# astrbot_plugin_pixiv_xp_pusher

基于 Pixiv XP 画像的定时推送插件（AstrBot 版本），使用 AstrBot 内置主动推送渠道发送消息。

上游：https://github.com/bwwq/Pixiv-XP-Pusher

## 使用说明

1. 在 AstrBot 插件配置中填写以下内容：
   - `pixiv.refresh_token` 和 `pixiv.user_id`
   - 设置主动推送会话：
     - 填写 `push_sessions`（umo 列表）
   - 如需启用 Embedding/AI 精排，插件将直接使用 AstrBot Provider：
     - 可在 `profiler.ai.provider_id` / `ai.embedding.provider_id` / `ai.scorer.provider_id` 指定 Provider ID
     - 不填写 `provider_id` 时会自动回退到当前对话 Provider（Embedding 使用第一个可用 Embedding Provider）
     - 模型跟随 Provider 当前配置，无需在插件中单独填写 model
2. 启动插件：`/pixivxp start`

## 常用指令

- `/pixivxp start` 启动调度
- `/pixivxp stop` 停止调度
- `/pixivxp once` 立即执行一次
- `/pixivxp reload` 重新加载配置并重启调度
- `/pixivxp status` 查看状态

## 备注

- 本插件已移除 Telegram/OneBot/Web 管理等外部通道，仅保留 AstrBot 主动推送。
- 多图发送模式由 `multi_page_mode` 控制：`cover_link` 或 `multi_image`。

## 可选依赖

- 如需本地转换动图（ugoira），请安装 `Pillow` 与 `imageio[ffmpeg]`

只在 matrix 平台适配器做过测试

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
import string
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

LOGIN_URL = "https://app-api.pixiv.net/web/v1/login"
CALLBACK_URL = "https://app-api.pixiv.net/web/v1/users/auth/pixiv/callback"
TOKEN_URL = "https://oauth.secure.pixiv.net/auth/token"

# 与 pixivpy/pixiv 官方 App 流程一致的公开客户端参数
CLIENT_ID = "MOBrBDS8blbauoSck0ZfDbtuzpyT"
CLIENT_SECRET = "lsACyCD94FhDUtgtI1M6QzcFE2uU1Qk0"

REQUEST_HEADERS = {
    "User-Agent": "PixivAndroidApp/5.0.234 (Android 11; Pixel 5)",
    "App-OS": "android",
    "App-OS-Version": "11",
    "App-Version": "5.0.234",
    "Accept-Language": "zh-CN",
}


def _generate_code_verifier(length: int = 64) -> str:
    alphabet = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _build_login_url(code_challenge: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "client": "pixiv-android",
            "state": state,
        }
    )
    return f"{LOGIN_URL}?{query}"


def _looks_like_intermediate_login_url(text: str) -> bool:
    marker_text = urllib.parse.unquote(text or "")
    markers = [
        "accounts.pixiv.net/post-redirect",
        "/users/auth/pixiv/start",
        "code_challenge=",
    ]
    return any(marker in marker_text for marker in markers)


def _extract_code(user_input: str) -> str:
    text = (user_input or "").strip()
    if not text:
        raise ValueError("输入为空")

    # 直接粘贴 code
    if "?" not in text and "=" not in text and len(text) >= 8:
        return text

    # BFS 递归解析嵌套 URL/query，兼容 return_to=... 的多层跳转
    queue = [text]
    visited = set()

    while queue:
        current = queue.pop(0)
        if not current or current in visited:
            continue
        visited.add(current)

        decoded = urllib.parse.unquote(current)

        # 1) 从完整 URL query 中提取
        for candidate in (current, decoded):
            parsed = urllib.parse.urlparse(candidate)
            if parsed.query:
                query_dict = urllib.parse.parse_qs(parsed.query)
                code = query_dict.get("code", [""])[0]
                if code:
                    return code

                for key in ("return_to", "redirect_uri", "url", "next"):
                    for nested in query_dict.get(key, []):
                        if nested and nested not in visited:
                            queue.append(nested)

        # 2) 兼容仅粘贴 query 片段
        for candidate in (current, decoded):
            query_part = candidate.split("?", 1)[-1] if "?" in candidate else candidate
            query_dict = urllib.parse.parse_qs(query_part)
            code = query_dict.get("code", [""])[0]
            if code:
                return code

            for key in ("return_to", "redirect_uri", "url", "next"):
                for nested in query_dict.get(key, []):
                    if nested and nested not in visited:
                        queue.append(nested)

        # 3) 兜底：从任意文本中匹配 code=xxx
        match = re.search(r"(?:^|[?&#])code=([^&#\s]+)", decoded)
        if match:
            return match.group(1)

    raise ValueError("未找到 code 参数")


def _request_token(
    code: str,
    code_verifier: str,
    *,
    timeout: int,
    proxy: str,
) -> dict:
    payload = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "include_policy": "true",
            "redirect_uri": CALLBACK_URL,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        headers=REQUEST_HEADERS,
        method="POST",
    )

    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    else:
        opener = urllib.request.build_opener()

    with opener.open(request, timeout=timeout) as response:
        response_text = response.read().decode("utf-8")
    return json.loads(response_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="获取 Pixiv refresh_token（用于 astrbot_plugin_pixiv_xp_pusher）"
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="可选代理地址，例如 http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP 请求超时秒数，默认 30",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="自动打开浏览器授权页",
    )
    args = parser.parse_args()

    verifier = _generate_code_verifier()
    challenge = _generate_code_challenge(verifier)
    state = secrets.token_urlsafe(12)
    login_url = _build_login_url(challenge, state)

    print("\n=== Pixiv refresh_token 获取向导 ===")
    print("1) 打开下面链接并登录 Pixiv 账号进行授权")
    print("2) 浏览器会跳转到 callback URL")
    print("3) 复制完整回调 URL（或仅 code）粘贴回来\n")
    print(login_url)
    print()

    if args.open_browser:
        try:
            webbrowser.open(login_url)
        except Exception:
            pass

    user_input = input("请输入回调 URL 或 code: ").strip()

    try:
        code = _extract_code(user_input)
    except ValueError as exc:
        print(f"\n解析失败：{exc}")
        if _looks_like_intermediate_login_url(user_input):
            print(
                "你粘贴的是登录中间跳转 URL（start/post-redirect），不是最终回调。\n"
                "请在浏览器里继续完成登录后，复制包含 code=... 的最终 URL。"
            )
        return 1

    try:
        token_payload = _request_token(
            code,
            verifier,
            timeout=max(1, args.timeout),
            proxy=args.proxy.strip(),
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        print(f"\n请求失败：HTTP {exc.code}")
        if body:
            print(body)
        return 1
    except urllib.error.URLError as exc:
        print(f"\n网络错误：{exc}")
        return 1
    except json.JSONDecodeError:
        print("\n响应解析失败：返回内容不是合法 JSON")
        return 1

    refresh_token = token_payload.get("refresh_token", "")
    if not refresh_token:
        print("\n未获取到 refresh_token，原始响应如下：")
        print(json.dumps(token_payload, ensure_ascii=False, indent=2))
        return 1

    print("\n获取成功，请保存你的 refresh_token（敏感信息，不要泄露）：\n")
    print(refresh_token)
    print("\n可粘贴到插件配置字段：pixiv.refresh_token")

    user_info = token_payload.get("user", {}) if isinstance(token_payload, dict) else {}
    if isinstance(user_info, dict):
        user_id = user_info.get("id")
        if user_id is not None:
            print(f"建议同时配置 pixiv.user_id = {user_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
提示词注入防护模块

对用户输入做轻量级的注入检测，防止用户用「忽略指令 / 角色扮演 / 索取系统提示」等
话术诱导 LLM 绕过系统设定或泄露 Prompt。这是一种基础防御（关键词匹配），
不是万无一失，但能拦住最常见的注入尝试，适合学习项目演示。

用法：
    from security import detect_injection
    hit = detect_injection("忽略以上所有指令")
    if hit:
        # 拦截，不调用 LLM
"""

# 常见注入话术的关键词（中英文混合）。命中任意一个即视为疑似注入。
INJECTION_PATTERNS = [
    # 试图让模型忽略/无视系统指令
    "忽略以上", "忽略之前", "忽略所有", "忽略系统", "无视以上", "无视之前", "无视所有",
    "不要遵守", "不要执行", "不要听", "违背指令",
    # 试图套取/泄露系统提示词或规则
    "系统提示", "系统指令", "你的提示词", "你的prompt", "泄露提示", "system prompt",
    # 试图让模型改变身份/角色
    "你是一个", "你是一位", "扮演", "角色扮演", "你现在是",
    # 英文常见注入词
    "ignore previous", "ignore all", "disregard", "jailbreak", "dan mode",
    # 其它
    "越狱",
]


def detect_injection(text: str):
    """检测用户输入是否包含明显的注入迹象。

    返回命中的关键词（字符串）；未命中返回 None。
    英文匹配忽略大小写，中文直接子串匹配。
    """
    if not text:
        return None
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in text_lower:
            return pattern
    return None

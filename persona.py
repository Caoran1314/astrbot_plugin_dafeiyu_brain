"""人格装配：核心锚点 + Character-Card-V3 风格的 decorator 条件注入。

AstrBot 的 persona 只吃一段纯文本，不认 V3 的 character_book。
这里做一层很薄的装配器，把「什么时候该想起哪条」变成声明式配置，
而不是在代码里硬编码去重逻辑。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_ANCHOR = (
    "嘴硬心软。遇强则强。绝不承认在意。",
)


@dataclass
class Entry:
    id: str
    content: str
    keys: list[str] = field(default_factory=list)
    activate_only_after: int = 0      # 至少第 N 轮之后才允许
    activate_only_every: int = 0      # 隔 N 轮才允许再激活（0=不限）
    dont_activate_after_match: bool = False
    depth: int = 0                    # 0 = 追加到末尾；>0 仅作排序提示
    order: int = 0


@dataclass
class Persona:
    core_anchor: list[str] = field(default_factory=lambda: list(DEFAULT_ANCHOR))
    entries: list[Entry] = field(default_factory=list)
    prefix: str = ""
    max_chars: int = 1200


def load_persona(path: str) -> Persona:
    if not path or not os.path.isfile(path):
        return Persona()
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        return Persona()
    anchor = data.get("core_anchor")
    entries = []
    for i, raw in enumerate(data.get("entries") or []):
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        entries.append(Entry(
            id=str(raw.get("id") or f"e{i}"),
            content=content,
            keys=[str(k) for k in (raw.get("keys") or []) if str(k).strip()],
            activate_only_after=int(raw.get("activate_only_after") or 0),
            activate_only_every=int(raw.get("activate_only_every") or 0),
            dont_activate_after_match=bool(raw.get("dont_activate_after_match", False)),
            depth=int(raw.get("depth") or 0),
            order=i,
        ))
    return Persona(
        core_anchor=[str(x) for x in anchor] if isinstance(anchor, list) and anchor
        else list(DEFAULT_ANCHOR),
        entries=entries,
        prefix=str(data.get("prefix") or ""),
        max_chars=int(data.get("max_chars") or 1200),
    )


def anchor_text(persona: Persona) -> str:
    """每轮都要重钉的硬约束。放在 system_prompt 末尾，别放开头。"""
    if not persona.core_anchor:
        return ""
    body = "".join(x.strip().rstrip("。") + "。" for x in persona.core_anchor if x.strip())
    return f"\n\n〔你不论如何都要保持〕\n{body}"


def select_entries(persona: Persona, turn: int, message_text: str,
                   state: dict) -> tuple[list[Entry], dict]:
    """按轮次与关键词挑选要注入的条目。state 会被原地更新并返回。"""
    text = message_text or ""
    used: list[Entry] = []
    for e in persona.entries:
        sid = e.id
        rec = state.get(sid) or {}
        last_turn = int(rec.get("last_turn", -10 ** 9))
        matched = bool(rec.get("matched", False))

        if e.dont_activate_after_match and matched:
            continue
        if turn < e.activate_only_after:
            continue
        if e.activate_only_every and (turn - last_turn) < e.activate_only_every:
            continue
        if e.keys:
            hit = any(k and k.lower() in text.lower() for k in e.keys)
            if not hit:
                continue
        used.append(e)
        state[sid] = {"last_turn": turn, "matched": True}
    used.sort(key=lambda e: (e.depth, e.order))
    return used, state


def render(entries: list[Entry], max_chars: int = 1200) -> str:
    if not entries:
        return ""
    out: list[str] = []
    total = 0
    for e in entries:
        block = f"- {e.content}"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)

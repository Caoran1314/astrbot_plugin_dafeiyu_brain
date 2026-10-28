"""大肥鱼记忆大脑（dafeiyu_brain）。

负责三件事：
1. 长期记忆：SQLite 四表（episodes / facts / profiles / memory_ops），双通道检索。
2. 人格装配：核心锚点每轮重钉 + V3 风格的条件注入。
3. 给模型两个工具：remember（她主动记）和 recall（她主动想）。

设计原则：任何失败都不阻塞对话。检索超时/嵌入挂了就退化成纯词法。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.config.astrbot_config import AstrBotConfig

from .memory import MemoryStore, content_hash, normalize_time
from .persona import anchor_text, load_persona, render, select_entries

PLUGIN_NAME = "dafeiyu_brain"

# 自动兜底捕捉：模型没主动 remember 时，靠规则先存下来，宁可多记不要漏记。
# 只匹配明确的「值得记」信号，避免把闲聊塞满库里。
_CAPTURE_EPISODE = re.compile(
    r"(记住|记一下|记着|别忘了|提醒我|约好|答应我|我(?:下周|明天|后天|这周|周末))")
_CAPTURE_FACT = re.compile(
    r"我(?P<attr>喜欢|爱吃|讨厌|不吃|忌口|过敏|生日是|生日|住在|老家在|在.*上班)"
    r"(?P<val>[^，。！？\n]{1,20})")


class DafeiyuBrain(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.cfg: dict = dict(config or {})
        self.data_dir = str(StarTools.get_data_dir(PLUGIN_NAME))
        os.makedirs(self.data_dir, exist_ok=True)

        self.store = MemoryStore(os.path.join(self.data_dir, "memory.db"))
        self.persona_path = self._resolve_persona_path()
        self.persona = load_persona(self.persona_path)

        try:
            removed = self.store.purge_questions()
            if removed:
                logger.info(f"[{PLUGIN_NAME}] 清理了 {removed} 条误存的问句")
        except Exception:
            pass

        self.turn: dict[str, int] = {}
        self.pstate: dict[str, dict] = {}
        self._vec_cache: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"[{PLUGIN_NAME}] 已加载：记忆 {self.store.stats()}，人格条目 {len(self.persona.entries)}")

        self._maint_task: asyncio.Task | None = None
        if bool(self.cfg.get("maintain_enable", True)):
            try:
                self._maint_task = asyncio.get_running_loop().create_task(self._maintain_loop())
            except RuntimeError:
                self._maint_task = None   # 无事件循环时跳过，靠 /brain maintain 手动跑

        self._backfill_task: asyncio.Task | None = None   # 懒启动，见 _ensure_backfill()

    def _ensure_backfill(self) -> None:
        """在第一次请求时启动回填——那时一定有运行中的事件循环。"""
        if self._backfill_task is not None:
            return
        if not bool(self.cfg.get("embedding_enable", False)):
            return
        if not bool(self.cfg.get("embedding_backfill", True)):
            return
        try:
            self._backfill_task = asyncio.get_running_loop().create_task(self._backfill_loop())
            logger.info(f"[{PLUGIN_NAME}] 向量回填任务已启动")
        except RuntimeError:
            self._backfill_task = None

    async def _backfill_loop(self) -> None:
        """给历史记忆补向量。没向量的记忆只能走关键词，补上之后语义检索才完整。"""
        try:
            await asyncio.sleep(20)
            total = 0
            while True:
                rows = await asyncio.to_thread(self.store.rows_missing_embedding, 16)
                if not rows:
                    if total:
                        logger.info(f"[{PLUGIN_NAME}] 向量回填完成，共 {total} 条")
                    return
                done = 0
                for table, rid, text in rows:
                    if not text.strip():
                        continue
                    vec = await self.embed_one(text)
                    if not vec:
                        logger.warning(f"[{PLUGIN_NAME}] 回填中断：嵌入服务不可用")
                        return
                    await asyncio.to_thread(self.store.set_embedding, table, rid, vec)
                    done += 1
                await asyncio.to_thread(self.store.db.commit)
                total += done
                logger.info(f"[{PLUGIN_NAME}] 向量回填 +{done}（累计 {total}）")
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 回填异常（忽略）: {e!r}")

    async def _maintain_loop(self) -> None:
        try:
            await asyncio.sleep(30)         # 起来先别抢资源
            while True:
                try:
                    res = await asyncio.to_thread(self.store.maintain)
                    logger.info(f"[{PLUGIN_NAME}] 维护完成: {res}")
                except Exception as e:
                    logger.warning(f"[{PLUGIN_NAME}] 维护失败（忽略）: {e!r}")
                await asyncio.sleep(max(3600.0, float(self.cfg.get("maintain_interval_hours", 6)) * 3600))
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------ 配置

    def _c(self, key: str, default):
        v = self.cfg.get(key, default)
        return default if v is None else v

    def _resolve_persona_path(self) -> str:
        raw = str(self._c("persona_file", "") or "").strip()
        for cand in (raw,
                     os.path.join(self.data_dir, "persona.json"),
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona.json")):
            if cand and os.path.isfile(cand):
                return cand
        return ""

    # ------------------------------------------------------------ 嵌入

    def _embed_sync(self, texts: list[str]) -> list[list[float]] | None:
        base = str(self._c("embedding_api_base", "") or "").strip()
        model = str(self._c("embedding_model", "") or "").strip()
        if not base or not model:
            return None
        import urllib.request
        url = base.rstrip("/") + "/embeddings"
        body = json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        key = str(self._c("embedding_api_key", "") or "").strip()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=float(self._c("embedding_timeout", 8))) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            return [d["embedding"] for d in data.get("data", [])]
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 嵌入失败，退化为词法检索: {e!r}")
            return None

    async def embed_one(self, text: str) -> list[float] | None:
        if not text:
            return None
        if not self._c("embedding_enable", False):
            return None          # 没配嵌入服务就别去连，避免每轮连接被拒
        h = content_hash(text)
        if h in self._vec_cache:
            return self._vec_cache[h]
        vecs = await asyncio.to_thread(self._embed_sync, [text])
        if not vecs:
            return None
        v = vecs[0]
        if len(self._vec_cache) > 2000:
            self._vec_cache.clear()
        self._vec_cache[h] = v
        return v

    # ------------------------------------------------------------ 自动兜底捕捉

    def _auto_capture_sync(self, text: str, umo: str) -> str:
        """在后台线程里跑：规则命中就存一条。返回描述，便于打日志。"""
        owner = str(self._c("owner_qq", "") or "")
        m = _CAPTURE_FACT.search(text)
        if m:
            attr = (m.group("attr") or "").strip()
            val = (m.group("val") or "").strip()
            if attr and val:
                attr_map = {"爱吃": "喜欢", "不吃": "忌口", "生日是": "生日",
                            "生日": "生日", "老家在": "老家", "住在": "住址"}
                attr = attr_map.get(attr, attr)
                self.store.upsert_fact(owner or "主人", attr, val, 0.7)
                return f"自动记事实: {attr}={val}"

        if _CAPTURE_EPISODE.search(text) and len(text) >= 6:
            if self.store.has_recent(text, 12):
                return ""
            gist = re.sub(r"^(?:记住|记一下|记着|别忘了|提醒我|提醒一下)[，,：:\s]*", "", text)
            gist = gist.strip() or text
            self.store.add_episode(gist[:120], importance=0.55, source="auto",
                                   people=owner)
            return f"自动记事件: {gist[:40]}"
        return ""

    async def _auto_capture(self, event) -> None:
        try:
            if not self._c("auto_capture", True):
                return
            text = str(getattr(event, "message_str", "") or "").strip()
            if len(text) < 4 or text.startswith(("/", "!", "#")):
                return
            # 问句不是记忆：「我下周去哪？」「还记得吗」都不该被存下来
            if re.search(r"[？?]", text) or re.search(r"(?:吗|呢|没|不|吧|么)[。！!~…]*$", text):
                return
            if "还记得" in text or "记得吗" in text:
                return
            umo = str(getattr(event, "unified_msg_origin", "") or "")
            msg = await asyncio.to_thread(self._auto_capture_sync, text, umo)
            if msg:
                logger.info(f"[{PLUGIN_NAME}] {msg}")
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 自动捕捉失败（忽略）: {e!r}")

    # ------------------------------------------------------------ 注入

    @filter.on_llm_request(priority=95000)
    async def brain_inject(self, event: AstrMessageEvent, req) -> None:
        try:
            if not self._c("enable", True):
                return
            umo = str(getattr(event, "unified_msg_origin", "") or "default")
            self.turn[umo] = self.turn.get(umo, 0) + 1
            turn = self.turn[umo]

            self._ensure_backfill()

            # 0) 规则兜底：模型可能忘了调用 remember，这里先接住
            asyncio.create_task(self._auto_capture(event))

            blocks: list[str] = []

            # 1) V3 条件条目
            state = self.pstate.setdefault(umo, {})
            used, _ = select_entries(self.persona, turn, str(getattr(event, "message_str", "") or ""), state)
            if used:
                body = render(used, self.persona.max_chars)
                if body:
                    blocks.append("〔本来就知道的事〕\n" + body)

            # 2) 长期记忆（线索式，不给档案）
            if self._c("memory_inject", True):
                query = str(getattr(event, "message_str", "") or "").strip()
                if len(query) >= 2:
                    vec = await self.embed_one(query) if self._c("embedding_enable", True) else None
                    hits = await asyncio.to_thread(
                        self.store.search, query,
                        int(self._c("top_k", 3)),
                        float(self._c("threshold", 0.14)),
                        vec,
                        str(self._c("owner_qq", "") or ""),
                        True,
                        query,
                    )
                    if hits:
                        lines = [self._fmt_hit(h) for h in hits]
                        blocks.append(
                            "〔想起来的（只是线索，别当档案背，合适才提）〕\n" + "\n".join(lines))
                        logger.info(
                            f"[{PLUGIN_NAME}] 召回 {len(hits)} 条: "
                            + " | ".join(f"{h.score:.2f}:{h.text[:18]}" for h in hits))

            if blocks:
                self._append_system(req, "\n\n" + "\n\n".join(blocks))

            # 3) 核心锚点永远最后
            anchor = anchor_text(self.persona)
            self._append_system(req, anchor)
            logger.info(
                f"[{PLUGIN_NAME}] 第 {turn} 轮注入：条目 {len(used)}，"
                f"记忆 {sum(1 for b in blocks if b.startswith('〔想起来的')) and 1 or 0} 块，"
                f"锚点 {len(anchor)} 字，system_prompt {len(getattr(req, 'system_prompt', '') or '')} 字")
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 注入异常已忽略: {e!r}")

    @staticmethod
    def _fmt_hit(h) -> str:
        """线索式展示：正文里已经带日期就不重复加。"""
        import re as _re
        text = h.text or ""
        if h.kind == "fact":
            return f"- {text}"
        if _re.match(r"^\d{4}-\d{2}-\d{2}", text):
            return f"- {text}"
        when = (h.meta.get("occurred_at") or "")[:10]
        return f"- {when} {text}".strip()

    @staticmethod
    def _append_system(req, text: str) -> None:
        if not text:
            return
        try:
            cur = getattr(req, "system_prompt", None)
            if isinstance(cur, str):
                req.system_prompt = cur + text
                return
            parts = getattr(req, "extra_user_content_parts", None)
            if isinstance(parts, list):
                parts.append({"type": "text", "text": text})
        except Exception:
            pass

    # ------------------------------------------------------------ 工具

    @filter.llm_tool(name="remember")
    async def remember(self, event: AstrMessageEvent, content: str,
                       kind: str = "episode", importance: float = 0.5,
                       people: str = "") -> str:
        '''把一件值得记住的事写进长期记忆。
        - 只记「以后可能还会提起」的事：约定、偏好、重要日子、情绪大事。
        - 平常闲聊不要记，会污染检索。
        - 写之前先把相对时间换掉：说「今天」就写成具体日期。

        Args:
            content(string): 一句话，不超过 60 字，写清楚发生了什么
            kind(string): episode（发生的事）或 fact（稳定事实）
            importance(float): 0~1，越重要越高
            people(string): 涉及谁，填 QQ 号或称呼，多个用逗号
        '''
        try:
            text = normalize_time((content or "").strip())
            if not text:
                return "内容空，没记。"
            if len(text) > 200:
                text = text[:200]
            if kind == "fact":
                # fact 需要三段式，容错：subject=people 或 owner，predicate=内容
                subject = (people or str(self._c("owner_qq", "owner"))).split(",")[0].strip()
                vec = await self.embed_one(text)
                self.store.upsert_fact(subject, "相关", text, float(importance), vec)
            else:
                vec = await self.embed_one(text)
                self.store.add_episode(text, people=people or "", importance=float(importance), embedding=vec)
            logger.info(f"[{PLUGIN_NAME}] 记住: {text[:40]}")
            return "记住了。"
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] remember 失败: {e!r}")
            return "没记住，等会儿再说吧。"

    @filter.llm_tool(name="remember_fact")
    async def remember_fact(self, event: AstrMessageEvent, subject: str,
                            attribute: str, value: str,
                            importance: float = 0.8) -> str:
        '''记一个「稳定事实」——不会随时间过期的信息。
        和 remember 的区别：remember 记「发生的事」，这里记「是什么」。
        如果新值和旧的矛盾（比如以前不吃辣，现在爱吃辣），直接再记一次，
        系统会自动把旧事实标为失效，不要自己删。

        Args:
            subject(string): 这是谁的事实，主人就填「主人」
            attribute(string): 属性，尽量用这些词：喜欢 / 讨厌 / 忌口 / 过敏 / 生日 / 称呼 / 工作 / 住址 / 重要日期 / 计划
            value(string): 具体内容，不超过 30 字
            importance(float): 0~1
        '''
        try:
            attribute = (attribute or "").strip()[:12]
            value = normalize_time((value or "").strip())[:60]
            if not (subject and attribute and value):
                return "信息不全，没记。"
            self.store.upsert_fact(subject.strip()[:20], attribute, value, float(importance))
            logger.info(f"[{PLUGIN_NAME}] 记住事实: {subject}/{attribute}={value}")
            return "记住了。"
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] remember_fact 失败: {e!r}")
            return "没记住。"

    @filter.llm_tool(name="recall")
    async def recall(self, event: AstrMessageEvent, query: str, k: int = 5) -> str:
        '''主动想一件事。

        Args:
            query(string): 想找什么，关键词就行
            k(int): 最多几条，默认 5
        '''
        try:
            vec = await self.embed_one(query)
            hits = await asyncio.to_thread(
                self.store.search, query, max(1, min(int(k), 10)), 0.10, vec,
                str(self._c("owner_qq", "") or ""))
            if not hits:
                return "想不起来了。"
            return "\n".join(self._fmt_hit(h).lstrip("- ") for h in hits)
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] recall 失败: {e!r}")
            return "想不起来了。"

    # ------------------------------------------------------------ 命令

    @filter.command("brain")
    async def brain_cmd(self, event: AstrMessageEvent):
        """记忆大脑运维：/brain / /brain search 关键词 / /brain export / /brain backup"""
        try:
            if not event.is_admin():
                return
            args = (str(getattr(event, "message_str", "") or "")).split()
            sub = args[1] if len(args) > 1 else "stats"
            if sub == "stats":
                st = self.store.stats()
                yield event.plain_result(
                    f"记忆 {st['episodes']} 条事件 / {st['facts']} 条事实 / "
                    f"{st['profiles']} 个人物；人格条目 {len(self.persona.entries)}")
            elif sub == "search" and len(args) > 2:
                q = " ".join(args[2:])
                hits = await asyncio.to_thread(self.store.search, q, 8, 0.05, None, "")
                if not hits:
                    yield event.plain_result("没有命中")
                    return
                lines = [f"{h.meta.get('occurred_at','')[:10]} {h.text} ({h.score:.2f})".strip()
                         for h in hits]
                yield event.plain_result("\n".join(lines))
            elif sub == "export":
                path = os.path.join(self.data_dir, f"export_{int(time.time())}.json")
                n = await asyncio.to_thread(self.store.export_json, path)
                yield event.plain_result(f"导出 {n} 条到 {os.path.basename(path)}")
            elif sub == "backup":
                path = os.path.join(self.data_dir, f"backup_{time.strftime('%Y%m%d')}.db")
                await asyncio.to_thread(self.store.backup, path)
                yield event.plain_result(f"备份完成 {os.path.basename(path)}")
            elif sub == "maintain":
                res = await asyncio.to_thread(self.store.maintain, None, 0.99, 20000, 0.0)
                yield event.plain_result(f"维护完成 {res}")
            else:
                yield event.plain_result("用法: /brain [stats|search 词|export|backup|maintain]")
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 命令失败: {e!r}")

    async def terminate(self) -> None:
        for attr in ("_maint_task", "_backfill_task"):
            t = getattr(self, attr, None)
            if t:
                t.cancel()
        self.store.close()
        logger.info(f"[{PLUGIN_NAME}] 已卸载")

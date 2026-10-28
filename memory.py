"""大肥鱼记忆核心：SQLite 四表 + 双通道检索 + 时间归一化 + 衰减。

刻意不依赖 FAISS（避开中文路径的坑），也不依赖 FTS5 的中文分词——
中文词法通道用字符 bigram 重合度，几千条记忆下性能完全够。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

try:  # 有 numpy 就快 10 倍，没有也能跑
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

# ------------------------------------------------------------------ 时间归一化

_REL_TIME = [
    (("大前天",), -3),
    (("前天",), -2),
    (("昨天", "昨晚", "昨儿"), -1),
    (("今天", "今晚", "今儿"), 0),
    (("明天", "明早", "明晚"), 1),
    (("后天",), 2),
    (("大后天",), 3),
]


def normalize_time(text: str, now: float | None = None) -> str:
    """把「今天/昨天/昨晚」换成绝对日期，防止三个月后记忆错位。"""
    if not text:
        return text
    now = now if now is not None else time.time()
    tm = time.localtime(now)
    out = text
    for words, delta in _REL_TIME:
        for w in words:
            if w in out:
                ts = now + delta * 86400
                day = time.strftime("%Y-%m-%d", time.localtime(ts))
                # 「昨晚」保留「晚」的语义
                suffix = "晚" if w.endswith(("晚", "儿")) and w != "今儿" else ""
                out = out.replace(w, f"{day}{suffix}")
    return out


# ------------------------------------------------------------------ 词法通道

_CJK = re.compile(r"[\u4e00-\u9fff]")


def bigrams(text: str) -> set[str]:
    """中文字符 bigram 集合；英文/数字按词。"""
    if not text:
        return set()
    text = text.lower()
    grams: set[str] = set()
    chars = _CJK.findall(text)
    for i in range(len(chars) - 1):
        grams.add(chars[i] + chars[i + 1])
    for w in re.findall(r"[a-z0-9_]{2,}", text):
        grams.add(w)
    return grams


def unigrams(text: str) -> set[str]:
    """中文单字集合。短查询里「周三」和「下周三」共享不了 bigram，只能靠单字兜。"""
    if not text:
        return set()
    return set(_CJK.findall(text.lower()))


# 同义/口语归一：不装嵌入模型时，这是提升召回最便宜的手段
_SYNONYMS = {
    "礼拜": "周", "星期": "周",
    "明儿": "明天", "今儿": "今天", "昨儿": "昨天", "后儿": "后天",
    "啥": "什么", "咋样": "怎么样", "咋": "怎么",
    "赚": "挣",
}


def normalize_lex(text: str) -> str:
    """归一化口语/同义词，让「礼拜三」和「周三」能匹配上。"""
    if not text:
        return text
    out = text.lower()
    for k, v in _SYNONYMS.items():
        if k in out:
            out = out.replace(k, v)
    return out


def lexical_score(query: str, doc: str) -> float:
    """按「查询覆盖率」打分：查询里有多少内容在文档里出现过。

    不用 Jaccard 分母（长文档会被稀释到接近 0），这是检索场景该用的归一化。
    bigram 抓短语，unigram 兜短查询，6:4 混合。
    """
    query = normalize_lex(query)
    doc = normalize_lex(doc)
    qb, db = bigrams(query), bigrams(doc)
    qu, du = unigrams(query), unigrams(doc)
    cov_b = len(qb & db) / len(qb) if qb else 0.0
    cov_u = len(qu & du) / len(qu) if qu else 0.0
    return 0.6 * cov_b + 0.4 * cov_u


# ------------------------------------------------------------------ 打分与衰减

TAU_DAYS = {"core": 36500.0, "important": 30.0, "normal": 7.0}


def decay_factor(delta_days: float, tau_days: float) -> float:
    if tau_days <= 0:
        return 1.0
    return math.exp(-max(0.0, delta_days) / tau_days)


# bge-m3 对「不相干」的中文短句余弦也常在 0.35~0.45，
# 直接拿 cos 当相关度会让一堆噪声挤进来。先扣掉基线再拉伸到 0~1。
COS_BASELINE = 0.35
COS_SPAN = 0.65


def norm_cos(cos: float | None) -> float | None:
    if cos is None:
        return None
    return max(0.0, (cos - COS_BASELINE) / COS_SPAN)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2.0


CONTRAST_SPAN = 0.30   # 相对中位数高出 0.30 视为「显著更相关」


def contrastive_relevance(cos: float | None, median_cos: float | None,
                          lex: float) -> float:
    """对比式相关性：看这条比同批候选的中位数高出多少。

    bge-m3 对无关中文短句的余弦在 0.40~0.56，绝对阈值分不开；
    真正相关的那条相对同批会高出 0.1~0.25，减掉中位数后信号才干净。
    cos 缺失时退回纯词法。
    """
    if cos is None or median_cos is None:
        return lex
    delta = max(0.0, (cos - median_cos) / CONTRAST_SPAN)
    return min(1.0, 0.7 * delta + 0.3 * lex)


def base_relevance(cos: float | None, lex: float) -> float:
    """内容相关性，始终落在 0~1，门限就用它判定。"""
    nc = norm_cos(cos)
    if nc is None:
        return lex
    return 0.65 * nc + 0.35 * lex


def score(cos: float | None, lex: float, delta_days: float, tau_days: float,
          hits: int = 0, is_owner: bool = False, jitter: float = 1.0) -> float:
    """最终分。cos 缺失时只用词法通道。"""
    base = base_relevance(cos, lex)
    w = decay_factor(delta_days, tau_days)
    w *= 1.8 if is_owner else 1.0
    w *= 1.0 + 0.1 * min(int(hits or 0), 5)
    return base * w * jitter


# ------------------------------------------------------------------ 向量编解码

def pack_vec(vec: Iterable[float]) -> bytes:
    v = list(vec)
    return struct.pack(f"<{len(v)}f", *v)


def unpack_vec(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    if _np is not None:
        va, vb = _np.asarray(a, dtype="float32"), _np.asarray(b, dtype="float32")
        na, nb = float(_np.linalg.norm(va)), float(_np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ------------------------------------------------------------------ 存储

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  occurred_at TEXT NOT NULL,
  people TEXT DEFAULT '',
  gist TEXT NOT NULL,
  detail TEXT DEFAULT '',
  emotion TEXT DEFAULT '',
  importance REAL DEFAULT 0.5,
  kind TEXT DEFAULT 'normal',
  hits INTEGER DEFAULT 0,
  last_hit_at TEXT,
  source TEXT DEFAULT '',
  embedding BLOB,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  confidence REAL DEFAULT 0.8,
  first_seen TEXT DEFAULT (datetime('now','localtime')),
  last_confirmed TEXT DEFAULT (datetime('now','localtime')),
  superseded_by INTEGER,
  embedding BLOB
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_fact ON facts(subject, predicate, object);
CREATE TABLE IF NOT EXISTS profiles (
  subject TEXT PRIMARY KEY,
  card TEXT DEFAULT '{}',
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS memory_ops (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now','localtime')),
  op TEXT NOT NULL,
  payload TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS state (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""


@dataclass
class Hit:
    kind: str
    id: int
    text: str
    score: float
    meta: dict = field(default_factory=dict)


class MemoryStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------------- 写

    def add_episode(self, gist: str, detail: str = "", people: str = "",
                    emotion: str = "", importance: float = 0.5,
                    kind: str = "normal", source: str = "",
                    embedding: list[float] | None = None,
                    now: float | None = None) -> int:
        gist = normalize_time((gist or "").strip(), now)
        if not gist:
            raise ValueError("gist 不能为空")
        occurred = time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(now if now is not None else time.time()))
        cur = self.db.execute(
            "INSERT INTO episodes(occurred_at, people, gist, detail, emotion,"
            " importance, kind, source, embedding) VALUES (?,?,?,?,?,?,?,?,?)",
            (occurred, people, gist, detail, emotion, float(importance), kind, source,
             pack_vec(embedding) if embedding else None),
        )
        self._op("add_episode", {"id": cur.lastrowid, "gist": gist})
        self.db.commit()
        return int(cur.lastrowid or 0)

    def upsert_fact(self, subject: str, predicate: str, obj: str,
                    confidence: float = 0.8,
                    embedding: list[float] | None = None) -> int:
        subject, predicate, obj = subject.strip(), predicate.strip(), obj.strip()
        if not (subject and predicate and obj):
            raise ValueError("三元组不完整")
        row = self.db.execute(
            "SELECT id FROM facts WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, obj)).fetchone()
        if row:
            self.db.execute(
                "UPDATE facts SET last_confirmed=datetime('now','localtime'),"
                " confidence=MAX(confidence, ?) WHERE id=?", (confidence, row["id"]))
            self.db.commit()
            return int(row["id"])
        # 同 subject+predicate 的旧事实标记为被替代（不删）
        old = self.db.execute(
            "SELECT id FROM facts WHERE subject=? AND predicate=? AND superseded_by IS NULL",
            (subject, predicate)).fetchall()
        cur = self.db.execute(
            "INSERT INTO facts(subject,predicate,object,confidence,embedding)"
            " VALUES (?,?,?,?,?)",
            (subject, predicate, obj, confidence,
             pack_vec(embedding) if embedding else None))
        new_id = int(cur.lastrowid or 0)
        for r in old:
            self.db.execute("UPDATE facts SET superseded_by=? WHERE id=?", (new_id, r["id"]))
        self._op("upsert_fact", {"id": new_id, "s": subject, "p": predicate, "o": obj})
        self.db.commit()
        return new_id

    def set_profile(self, subject: str, card: dict) -> None:
        self.db.execute(
            "INSERT INTO profiles(subject, card, updated_at)"
            " VALUES (?,?,datetime('now','localtime'))"
            " ON CONFLICT(subject) DO UPDATE SET card=excluded.card,"
            " updated_at=excluded.updated_at",
            (subject, json.dumps(card, ensure_ascii=False)))
        self.db.commit()

    def get_profile(self, subject: str) -> dict:
        row = self.db.execute("SELECT card FROM profiles WHERE subject=?", (subject,)).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["card"])
        except Exception:
            return {}

    def _op(self, op: str, payload: dict) -> None:
        self.db.execute("INSERT INTO memory_ops(op, payload) VALUES (?,?)",
                        (op, json.dumps(payload, ensure_ascii=False)))

    # ---------------- 读

    def _rows(self, table: str) -> list[sqlite3.Row]:
        return self.db.execute(f"SELECT * FROM {table}").fetchall()

    def search(self, query: str, k: int = 5, threshold: float = 0.25,
               query_vec: list[float] | None = None,
               owner_subject: str = "",
               touch: bool = True,
               exclude_text: str = "") -> list[Hit]:
        """两阶段检索。

        先收集候选并算出原始余弦，再用「候选集中位数」做对比式归一化：
        bge-m3 对互不相关的中文短句余弦常在 0.40~0.56，绝对阈值分不开；
        真正相关的那条相对同批候选会明显更高，减掉中位数后差距才显现。
        """
        now = time.time()
        cand: list[dict] = []
        _ex = normalize_lex(exclude_text or "")

        for r in self._rows("episodes"):
            if _ex and normalize_lex(r["gist"] or "") == _ex:
                continue      # 就是用户当前这句话，别再回灌给自己
            text = " ".join(x for x in (r["gist"], r["detail"], r["people"]) if x)
            lex = lexical_score(query, text)
            cos = None
            if query_vec is not None and r["embedding"]:
                cos = cosine(query_vec, unpack_vec(r["embedding"]))
            if cos is None and lex <= 0:
                continue
            try:
                dt = time.mktime(time.strptime(r["occurred_at"], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                dt = now
            cand.append({"kind": "episode", "row": r, "text": r["gist"], "lex": lex,
                         "cos": cos, "delta_days": max(0.0, (now - dt) / 86400.0),
                         "tau": TAU_DAYS.get(r["kind"], 7.0),
                         "hits": r["hits"] or 0,
                         "owner": bool(owner_subject) and owner_subject in (r["people"] or ""),
                         "meta": {"occurred_at": r["occurred_at"], "kind": r["kind"]}})

        for r in self._rows("facts"):
            if r["superseded_by"]:
                continue
            text = f"{r['subject']} {r['predicate']} {r['object']}"
            lex = lexical_score(query, text)
            cos = None
            if query_vec is not None and r["embedding"]:
                cos = cosine(query_vec, unpack_vec(r["embedding"]))
            if cos is None and lex <= 0:
                continue
            try:
                dt = time.mktime(time.strptime(r["first_seen"], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                dt = now
            cand.append({"kind": "fact", "row": r,
                         "text": f"{r['subject']}{r['predicate']}{r['object']}",
                         "lex": lex, "cos": cos,
                         "delta_days": max(0.0, (now - dt) / 86400.0),
                         "tau": TAU_DAYS["important"], "hits": 0,
                         "owner": bool(owner_subject) and r["subject"] == owner_subject,
                         "meta": {"subject": r["subject"], "predicate": r["predicate"]}})

        if not cand:
            return []

        cosines = [c["cos"] for c in cand if c["cos"] is not None]
        median_cos = _median(cosines) if cosines else None

        hits: list[Hit] = []
        for c in cand:
            rel = contrastive_relevance(c["cos"], median_cos, c["lex"])
            if rel < threshold:
                continue
            s = score(c["cos"], c["lex"], c["delta_days"], c["tau"], c["hits"], c["owner"])
            # 排序用相对突出度，避免被 1.8 倍主人加成拉偏
            s = rel * max(0.2, s) if s else rel
            if c["delta_days"] < 1.0:
                s *= 1.2
            hits.append(Hit(c["kind"], c["row"]["id"], c["text"], s, c["meta"]))

        hits.sort(key=lambda h: h.score, reverse=True)
        top = [h for h in hits[: max(1, k)] if h.score >= threshold]
        if touch and top:
            self._touch(top)
        return top

    def _touch(self, hits: list[Hit]) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for h in hits:
            if h.kind == "episode":
                self.db.execute(
                    "UPDATE episodes SET hits=hits+1, last_hit_at=? WHERE id=?", (now, h.id))
        self.db.commit()

    def purge_questions(self) -> int:
        """删掉被误当记忆存下来的「问句」——问句本身不是记忆，还会反复被召回。"""
        cur = self.db.execute(
            "DELETE FROM episodes WHERE gist LIKE '%吗%' OR gist LIKE '%呢%'"
            " OR gist LIKE '%？' OR gist LIKE '%?'"
            " OR gist LIKE '%还记得%' OR gist LIKE '%记着没%' OR gist LIKE '%记得吗%'")
        self.db.commit()
        return cur.rowcount

    def rows_missing_embedding(self, limit: int = 32) -> list[tuple[str, int, str]]:
        """找出还没算向量的记忆，用于回填。返回 [(表名, id, 文本)]。"""
        out: list[tuple[str, int, str]] = []
        for r in self.db.execute(
                "SELECT id, gist, detail, people FROM episodes WHERE embedding IS NULL"
                " ORDER BY id LIMIT ?", (limit,)):
            text = " ".join(x for x in (r["gist"], r["detail"], r["people"]) if x)
            out.append(("episodes", r["id"], text))
        for r in self.db.execute(
                "SELECT id, subject, predicate, object FROM facts WHERE embedding IS NULL"
                " ORDER BY id LIMIT ?", (limit,)):
            out.append(("facts", r["id"],
                        f"{r['subject']} {r['predicate']} {r['object']}"))
        return out[:limit]

    def set_embedding(self, table: str, row_id: int, vec: list[float]) -> None:
        if table not in ("episodes", "facts"):
            raise ValueError(table)
        self.db.execute(f"UPDATE {table} SET embedding=? WHERE id=?",
                        (pack_vec(vec), row_id))

    def has_recent(self, gist: str, hours: float = 24.0) -> bool:
        """24 小时内是否已存过同义内容，避免自动捕捉重复写。"""
        if not gist:
            return False
        rows = self.db.execute(
            "SELECT gist FROM episodes"
            " WHERE occurred_at >= datetime('now','localtime', ?)",
            (f"-{int(max(1, hours))} hour",)).fetchall()
        target = normalize_lex(gist)
        for r in rows:
            cand = normalize_lex(r["gist"] or "")
            if not cand:
                continue
            if cand == target or lexical_score(target, cand) >= 0.85:
                return True
        return False

    def stats(self) -> dict:
        ep = self.db.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
        ft = self.db.execute(
            "SELECT COUNT(*) c FROM facts WHERE superseded_by IS NULL").fetchone()["c"]
        pf = self.db.execute("SELECT COUNT(*) c FROM profiles").fetchone()["c"]
        return {"episodes": ep, "facts": ft, "profiles": pf}

    # ---------------- 维护

    def get_state(self, k: str, default: str = "") -> str:
        row = self.db.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
        return row["v"] if row else default

    def set_state(self, k: str, v: str) -> None:
        self.db.execute(
            "INSERT INTO state(k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, v))
        self.db.commit()

    def maintain(self, now: float | None = None, decay_per_day: float = 0.99,
                 max_episodes: int = 20000, min_interval_hours: float = 6.0) -> dict:
        """按天衰减重要性、清理长期没被想起的旧记忆、压缩操作日志。

        对应「会淡忘，但真正重要的忘不掉」——衰减是乘法，核心记忆因为
        importance 高、命中次数多，实际衰减速度远慢于闲聊记忆。
        """
        now = now if now is not None else time.time()
        last = float(self.get_state("last_maintain", "0") or 0)
        hours = (now - last) / 3600.0
        if last and hours < min_interval_hours:
            return {"skipped": True, "hours_since": round(hours, 1)}

        days = max(0.0, (now - last) / 86400.0) if last else 0.0
        out: dict = {"decayed_days": round(days, 2)}

        if days > 0:
            factor = decay_per_day ** days
            self.db.execute(
                "UPDATE episodes SET importance = MAX(0.05, importance * ?)"
                " WHERE importance > 0.05", (factor,))
            out["decay_factor"] = round(factor, 4)

        cur = self.db.execute(
            "DELETE FROM episodes WHERE hits = 0 AND importance < 0.25"
            " AND occurred_at < datetime('now','localtime','-90 day')")
        out["pruned_stale"] = cur.rowcount

        total = self.db.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
        if total > max_episodes:
            self.db.execute(
                "DELETE FROM episodes WHERE id IN ("
                " SELECT id FROM episodes ORDER BY (importance * (1 + hits)) ASC"
                " LIMIT ?)", (total - max_episodes,))
            out["pruned_overflow"] = total - max_episodes

        ops = self.db.execute("SELECT COUNT(*) c FROM memory_ops").fetchone()["c"]
        if ops > 5000:
            self.db.execute(
                "DELETE FROM memory_ops WHERE id NOT IN ("
                " SELECT id FROM memory_ops ORDER BY id DESC LIMIT 5000)")
            out["pruned_ops"] = ops - 5000

        self.db.commit()
        self.set_state("last_maintain", str(now))
        out["stats"] = self.stats()
        return out

    def export_json(self, path: str) -> int:
        data = {
            "episodes": [dict(r) for r in self._rows("episodes")],
            "facts": [dict(r) for r in self._rows("facts")],
            "profiles": [dict(r) for r in self._rows("profiles")],
        }
        for rows in data.values():
            for r in rows:
                r.pop("embedding", None)  # 向量可重建，不导出
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return sum(len(v) for v in data.values())

    def backup(self, path: str) -> None:
        """WAL 安全备份：不能直接 cp 文件。"""
        tmp = sqlite3.connect(path)
        with tmp:
            self.db.backup(tmp)
        tmp.close()

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


def content_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]

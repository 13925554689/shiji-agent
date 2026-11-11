# -*- coding: utf-8 -*-
"""
史记智能体 — 基于 baojie/shiji-kb 开源知识图谱 (57万字《史记》→ 14,065实体/3,198事件/130时间线)
学习来源: 2026-09 头条文章《有人把两千年前司马迁的〈史记〉变成了开源代码》
架构: 柳叶刀方法 — 实体/纪年/事件用确定性查表, 语义推理用 LLM, 文本按需取+本地缓存

用法:
  python shiji_agent.py "项羽和刘邦是什么关系?"
  python shiji_agent.py --demo          # 跑内置验收用例
依赖: 仅 Python 标准库; DeepSeek API key 放环境变量 DEEPSEEK_API_KEY 或 ROOT/deepseek_key.tmp
"""
import json, math, os, re, sys, threading, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "shiji-kb-data")
CACHE = os.path.join(ROOT, "chapter_cache")
GH_MIRRORS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/baojie/shiji-kb/main/",
    "https://ghfast.top/https://raw.githubusercontent.com/baojie/shiji-kb/main/",
    "https://raw.githubusercontent.com/baojie/shiji-kb/main/",
]

# ---------- 数据层 (本地优先, 缺失时从镜像拉取并缓存) ----------

def _fetch(path: str) -> bytes:
    last = None
    for u in GH_MIRRORS:
        try:
            req = urllib.request.Request(u + urllib.request.quote(path),
                                         headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            last = e
    raise RuntimeError(f"数据拉取失败: {path} ({last})")

def _atomic_write(p: str, raw: bytes):
    tmp = f"{p}.{threading.get_ident()}.tmp"   # ponytail: 并发写者各用私名, Windows os.replace 不再撞句柄
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, p)          # 原子落盘: 并发线程不会读到半写文件

def load_json(rel: str):
    p = os.path.join(DATA, rel.replace("/", "__"))
    if not os.path.exists(p):
        raw = _fetch(rel)
        json.loads(raw)         # 镜像200+HTML的坏响应不落盘
        os.makedirs(DATA, exist_ok=True)
        _atomic_write(p, raw)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        os.remove(p); raise     # 仅坏JSON删缓存; PermissionError等瞬时错误不误删有效缓存

def chapter_path(num: str) -> str:
    """num: '007' → 'chapter_md/007_项羽本纪.tagged.md'; LLM可能传整数, 先转str"""
    num = str(num).zfill(3)
    summary = load_json("kg/events/events_summary.json")
    for ch in summary["chapters"]:
        if ch["num"] == num.zfill(3):
            return f"chapter_md/{ch['chapter']}.tagged.md"
    raise KeyError(f"章节不存在: {num} (001-130)")

def get_chapter(num: str) -> str:
    path = chapter_path(str(num))
    p = os.path.join(CACHE, os.path.basename(path))
    if not os.path.exists(p):
        raw = _fetch(path)
        if raw[:1] == b"<":
            raise RuntimeError(f"镜像返回HTML而非章节: {path}")
        os.makedirs(CACHE, exist_ok=True)
        _atomic_write(p, raw)
    return open(p, encoding="utf-8").read()

# ---------- 标注剥离 (22类标记语法 → 可读文本, 保留段落号供引用) ----------

_RE = [(re.compile(r"〖(?:[@#$%&•=◆^;~+!_{[\]:?])?([^〗]*)〗"), 1),  # 实体标记: 前缀符号可选(catch-all不安全—会把@/=/◆等前缀吞进分组, selfcheck已断言), :/?为058/056/128章实测新增
       (re.compile(r"⟦[◉]?([^⟧]*)⟧"), 1),               # 军事等动词
       (re.compile(r"〘[※]?([^〙]*)〙"), 1),             # 成语/评注
       (re.compile(r"[◈◦‧○]"), 0)]                       # 单字符动词标记

def strip_tags(tagged: str) -> str:
    out = tagged
    for rx, gi in _RE:
        # gi=0 → 纯删除; 否则取分组并解析 "原名|消歧名" 取消歧名
        out = rx.sub(lambda m, gi=gi: "" if gi == 0 else (m.group(gi).split("|")[-1] if m.group(gi) else ""), out)
    return out

def find_sections(num: str, keyword: str, limit: int = 6):
    """在章节内按关键词定位段落, 返回 [(段落号, 文本)]"""
    text = strip_tags(get_chapter(num))
    hits, cur_no, cur_buf = [], "", []
    for line in text.splitlines():
        m = re.match(r"\s*\[(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if m:
            if cur_no and keyword in "".join(cur_buf):
                hits.append((cur_no, "".join(cur_buf).strip()[:600]))
                if len(hits) >= limit:
                    break
            cur_no, cur_buf = m.group(1), [m.group(2)]
        elif cur_no:
            cur_buf.append(line)
    if cur_no and keyword in "".join(cur_buf) and len(hits) < limit:
        hits.append((cur_no, "".join(cur_buf).strip()[:600]))
    return hits

# ---------- 向量检索 (character-bigram TF-IDF + cosine, stdlib-only) ----------
# ponytail: O(130) 每查询线性扫描, 毫秒级; 130章规模无需 ANN, 章节数过千再换 sqlite+外部索引

_BG_RE = re.compile(r"[\u4e00-\u9fff]+")

def _bigrams(text: str) -> dict:
    toks = _BG_RE.findall(text)
    d = {}
    for t in toks:
        for i in range(len(t) - 1):
            d[t[i:i+2]] = d.get(t[i:i+2], 0) + 1
    return d

def _load_passages():
    """130章 → 段落级语料, 返回 [(章节号, 章节名, 段落号, 文本)], 惰性构建一次"""
    return _load_index()[0]

def _load_index():
    """构建 (passages, norms) 并以单指针原子发布—并发线程要么拿到完整索引要么自己重建, 无锁"""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    summary = load_json("kg/events/events_summary.json")
    passages, norms = [], []
    for ch in summary["chapters"]:
        try:
            text = strip_tags(get_chapter(ch["num"]))
        except Exception:
            continue  # ponytail: 单章拉取失败跳过, 不阻塞整库
        cur_no, buf = "", []
        for line in text.splitlines():
            m = re.match(r"\s*\[(\d+(?:\.\d+)?)\]\s*(.*)", line)
            if m:
                if cur_no and len("".join(buf)) > 20:
                    _add_passage(passages, norms, ch["num"], ch["chapter"], cur_no, "".join(buf))
                cur_no, buf = m.group(1), [m.group(2)]
            elif cur_no:
                buf.append(line)
        if cur_no and len("".join(buf)) > 20:
            _add_passage(passages, norms, ch["num"], ch["chapter"], cur_no, "".join(buf))
    _INDEX = (passages, norms)
    return _INDEX

def _add_passage(out, norms_out, num, chname, pno, body):
    bg = _bigrams(body)
    if not bg:
        return
    norm = math.sqrt(sum(v * v for v in bg.values()))
    out.append((num, chname, pno, body[:400]))
    norms_out.append((bg, norm))

_INDEX = None

def search_text(query: str, k: int = 5) -> dict:
    """跨130章语义定位段落: bigram TF-IDF cosine"""
    qb = _bigrams(query)
    if not qb:
        return {"hits": []}
    qnorm = math.sqrt(sum(v * v for v in qb.values()))
    scored = []
    for (num, chname, pno, body), (bg, norm) in zip(*_load_index()):
        dot = sum(v * bg.get(g, 0) for g, v in qb.items())
        if dot:
            scored.append((dot / (qnorm * norm), num, chname, pno, body))
    if not scored:  # ponytail: 单字查询无bigram(如"楚") → 退化为子串匹配, 避免静默0命中
        toks = _BG_RE.findall(query)
        scored = [(0.0, num, chname, pno, body)
                  for (num, chname, pno, body), _ in zip(*_load_index())
                  if toks and all(t in body for t in toks)]
    scored.sort(reverse=True)
    return {"hits": [{"score": round(s, 4), "章节": f"{chname}({num})", "段落": p, "text": body}
                     for s, num, chname, p, body in scored[:k]]}

# ---------- 四个查询工具 (确定性查表, 不走 LLM) ----------

def tool_entity(name: str) -> dict:
    idx = load_json("kg/entity_index.json")
    out = []
    for etype, entities in idx.items():
        hit = entities.get(name)
        if hit is None:  # 别名反查
            for ename, h in entities.items():
                if name in (h.get("aliases") or []):
                    hit, name = h, ename
                    break
        if hit:
            chapters = {}
            for ch, _sec in hit.get("refs", []):
                chapters[ch] = chapters.get(ch, 0) + 1
            top = sorted(chapters.items(), key=lambda x: -x[1])[:8]
            out.append({"实体": name, "类型": etype, "别名": hit.get("aliases", [])[:10],
                        "标注次数": hit.get("count", len(hit.get("refs", []))),
                        "章节分布": [f"{c}({n}次)" for c, n in top]})
    return {"查询": name, "结果": out or "未收录"}

def tool_reign(ruler: str) -> dict:
    rp = load_json("kg/chronology/data/reign_periods.json").get("rulers", {})
    hits = {k: v for k, v in rp.items() if ruler in k}   # 宽松匹配: "武帝" 命中 "汉武帝"
    return {"查询": ruler, "在位年": hits or "未收录"}

def tool_chapter_year(num: str) -> dict:
    num = str(num).zfill(3)   # LLM可能传JSON number
    m = load_json("kg/chronology/data/year_ce_map.json")
    v = m.get(num)
    return {"章节": num, "公元纪年映射": v or "未收录"}

def tool_events(num: str = None) -> dict:
    es = load_json("kg/events/events_summary.json")
    if num:
        num = str(num).zfill(3)   # LLM可能传JSON number
        for ch in es["chapters"]:
            if ch["num"] == num:
                return ch
        return {"error": f"章节 {num} 无事件数据"}
    return {"总事件数": es["total_events"], "类型分布": es["type_distribution"]}

# ---------- LLM 工具调用循环 (DeepSeek, OpenAI 兼容协议) ----------

def _key():
    k = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not k and os.path.exists(os.path.join(ROOT, "deepseek_key.tmp")):
        k = open(os.path.join(ROOT, "deepseek_key.tmp")).read().strip()
    if not k:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY (环境变量或 ROOT/deepseek_key.tmp)")  # web线程中sys.exit会掐死请求
    return k

def llm(messages, tools=None, retries: int = 2):
    body = {"model": "deepseek-chat", "messages": messages, "temperature": 0.3, "max_tokens": 3000}
    if tools:
        body["tools"] = tools
    key = _key()               # 缺key立即报错, 不重试
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            raw = urllib.request.urlopen(req, timeout=180).read().decode()
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:   # 401/400 确定性失败重试无意义
                raise RuntimeError(f"DeepSeek 请求被拒 ({e.code}): {e.read().decode('utf-8', 'replace')[:200]}")
            last = e
        except Exception as e:
            last = e
        if attempt < retries:
            time.sleep(2)      # ponytail: 仅瞬态错误且还有下次尝试时退避
    else:
        raise RuntimeError(f"DeepSeek 调用失败: {last}")
    try:
        return json.loads(raw)["choices"][0]["message"]
    except Exception:
        raise RuntimeError(f"DeepSeek 响应异常: {raw[:200]}")

TOOL_DEFS = [
    {"type": "function", "function": {"name": "tool_entity", "description": "查实体: 人名/地名/官职/邦国/时间等18类, 返回类型/别名/标注次数/章节分布",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "实体名, 如 项羽"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "tool_text", "description": "取某章原文并按关键词定位段落 (《史记》001-130章, 如 007=项羽本纪)",
     "parameters": {"type": "object", "properties": {"num": {"type": "string"}, "keyword": {"type": "string", "description": "定位关键词, 如 鸿门"}}, "required": ["num", "keyword"]}}},
    {"type": "function", "function": {"name": "tool_reign", "description": "查君主在位年及公元换算 (前841年起)",
     "parameters": {"type": "object", "properties": {"ruler": {"type": "string"}}, "required": ["ruler"]}}},
    {"type": "function", "function": {"name": "tool_chapter_year", "description": "查某章覆盖的公元纪年范围",
     "parameters": {"type": "object", "properties": {"num": {"type": "string"}}, "required": ["num"]}}},
    {"type": "function", "function": {"name": "tool_events", "description": "查事件统计: 全书总览或单章事件清单(战争/继位/改革等11类)",
     "parameters": {"type": "object", "properties": {"num": {"type": "string", "description": "留空=全书统计"}}}}},
    {"type": "function", "function": {"name": "search_text", "description": "跨130章全文向量检索段落, 不知道该查哪一章时用 (如: 某事件/某句话出自哪里)",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "检索词, 如 鸿门宴 破釜沉舟"}, "k": {"type": "integer", "description": "返回条数, 默认5"}}}}},
]

_dispatch = {"tool_entity": lambda a: tool_entity(a["name"]),
             "tool_text": lambda a: ({"章节": a["num"], "命中段落": find_sections(a["num"], a["keyword"])} if a.get("keyword")
                                     else {"error": "tool_text 需要非空 keyword 定位词"}),
             "tool_reign": lambda a: tool_reign(a["ruler"]),
             "tool_chapter_year": lambda a: tool_chapter_year(a["num"]),
             "tool_events": lambda a: tool_events(a.get("num") or ""),
             "search_text": lambda a: search_text(a["query"], int(a.get("k") or 5))}

SYSTEM = """你是《史记》知识图谱智能体, 底层是 shiji-kb 开源知识库(14,065实体/3,198事件/130篇)。
规则:
1. 涉及人物/地点/年代/事件数据的问题, 先调用工具查确定性数据, 再作答; 不编造引文。
2. 引用原文时标注出处, 格式: (《章节名》[段落号])。
3. 用户要求对比/查矛盾时, 分别取不同章节文本再交叉比对, 明确指出不一致处。
4. 无工具可答的常识问题直接回答, 不空转调用。"""

_TRACE = threading.local()  # ponytail: ThreadingMixIn 下 per-request 轨迹, 全局列表会跨请求串扰

def ask(question: str, verbose: bool = False) -> str:
    _TRACE.list = []
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for turn in range(6):
        msg = llm(messages, TOOL_DEFS)
        calls = msg.get("tool_calls")
        if not calls:
            return msg["content"]
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": [{"id": c.get("id"), "type": "function",
                                         "function": {"name": c.get("function", {}).get("name", ""),
                                                      "arguments": c.get("function", {}).get("arguments", "{}")}} for c in calls]})
        for c in calls:
            fn = c.get("function", {}).get("name", "")
            try:
                result = _dispatch[fn](json.loads(c["function"]["arguments"] or "{}"))
            except Exception as e:
                result = {"error": f"{fn}: {e}"}
            if verbose:
                print(f"  [工具] {fn} → {json.dumps(result, ensure_ascii=False)[:180]}")
            _TRACE.list.append(fn)
            messages.append({"role": "tool", "tool_call_id": c.get("id"),
                             "content": json.dumps(result, ensure_ascii=False)[:6000]})
    return "（超过最大推理轮数）"

# ---------- Web 界面 (stdlib http.server, 零依赖) ----------

_PAGE = """<!doctype html><html lang=zh><meta charset=utf-8>
<title>史记智能体</title>
<style>
body{font-family:system-ui;max-width:760px;margin:24px auto;padding:0 16px;background:#faf7f2;color:#2b2118}
h1{font-size:22px;border-bottom:2px solid #8b0000;padding-bottom:8px}
#chat{min-height:300px}
.msg{margin:14px 0;padding:12px 14px;border-radius:10px;line-height:1.7;white-space:pre-wrap}
.q{background:#8b0000;color:#fff;margin-left:15%}
.a{background:#fff;border:1px solid #e0d5c5}
.tools{font-size:12px;color:#888;margin-top:6px}
form{display:flex;gap:8px;position:sticky;bottom:0;background:#faf7f2;padding:10px 0}
input{flex:1;padding:10px 14px;border:1px solid #c9b8a3;border-radius:8px;font-size:15px}
button{padding:10px 22px;background:#8b0000;color:#fff;border:0;border-radius:8px;cursor:pointer}
button:disabled{opacity:.5}
.sug button{background:#f0e6d8;color:#5a4632;margin:4px 4px 0 0;font-size:13px;padding:6px 12px}
</style>
<h1>📜 史记智能体 <small style=font-size:12px;color:#888>shiji-kb · 14,065实体 · 3,198事件 · 130篇</small></h1>
<div id=chat></div>
<form onsubmit="send();return false"><input id=q placeholder="问《史记》，如：长平之战秦赵主帅是谁？" autofocus>
<button id=btn>问</button></form>
<div class=sug>
<button onclick="qs('项羽和刘邦谁先入关中？')">项羽和刘邦谁先入关中？</button>
<button onclick="qs('汉武帝在位多少年？')">汉武帝在位多少年？</button>
<button onclick="qs('韩信之死哪个章记载？原文？')">韩信之死原文？</button>
</div>
<script>
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function add(cls, html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;
 document.getElementById('chat').appendChild(d);window.scrollTo(0,1e9);return d}
async function send(){
 const inp=document.getElementById('q'),q=inp.value.trim();if(!q)return false;
 inp.value='';add('q',esc(q));
 const btn=document.getElementById('btn');btn.disabled=true;
 const box=add('a','<span style=color:#aaa>查谱中…</span>');
 try{
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q}),signal:AbortSignal.timeout(120000)});
  const j=await r.json();
  box.innerHTML=esc(j.answer||('错误: '+j.error))+(j.tools?'<div class=tools>🔎 '+esc(j.tools)+'</div>':'');
 }catch(e){box.textContent='请求失败: '+e}
 btn.disabled=false;inp.focus();return false}
function qs(q){document.getElementById('q').value=q;send()}
</script>"""

def _web():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    class T(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    class H(BaseHTTPRequestHandler):
        timeout = 30  # ponytail: 无超时=慢客户端可永久占线程
        def log_message(self, *a): pass
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        def do_POST(self):
            if self.headers.get("Sec-Fetch-Site") == "cross-site":   # 跨站盲打(CSRF)防护: 恶意网页可text/plain简单请求烧LLM额度
                return self._send(403, b'{"error": "cross-site forbidden"}', "application/json; charset=utf-8")
            if (self.headers.get("Host") or "") not in H.allowed:    # DNS rebinding: evil.com解析到127.0.0.1时Sec-Fetch-Site是same-origin, 只能靠Host识别
                return self._send(403, b'{"error": "host forbidden"}', "application/json; charset=utf-8")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 65536)  # 上限防内存放大, 超长body截断即解析失败
                q = json.loads(self.rfile.read(n).decode("utf-8", "replace")).get("q", "").strip()
                answer = ask(q) if q else "空问题"
                trace = getattr(_TRACE, "list", [])
                payload = json.dumps({"answer": answer, "tools": " → ".join(trace) or None},
                                     ensure_ascii=False).encode("utf-8")
            except Exception as e:
                payload = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
    port = int(os.environ.get("SHIJI_PORT", "5210"))
    H.allowed = (f"127.0.0.1:{port}", f"localhost:{port}")
    import socket as _s
    if _s.socket().connect_ex(("127.0.0.1", port)) == 0:  # Windows SO_REUSEADDR 允许重复绑定→僵尸进程叠加, 启动前先探活
        sys.exit(f"端口 {port} 已被占用(可能有旧进程): netstat -ano | findstr :{port}")
    print(f"史记智能体 → http://127.0.0.1:{port}")
    T(("127.0.0.1", port), H).serve_forever()

# ---------- 自检 ----------

def selfcheck():
    t = strip_tags("〖@项籍〗者，〖=下相〗人也，字〖@羽|项羽〗。〖◆楚〗〖;将〗〖@项燕〗〖%二世元年七月〗⟦◉杀⟧。")
    assert t == "项籍者，下相人也，字项羽。楚将项燕二世元年七月杀。", t
    e = tool_entity("项羽")
    assert e["结果"] and e["结果"][0]["类型"] == "person", "实体索引查询失败"
    hits = find_sections("007", "鸿门")
    assert hits and "鸿门" in hits[0][1], "段落定位失败"
    r = tool_reign("武帝")
    assert r["在位年"], "在位年查询失败"
    ev = tool_events()
    assert ev["总事件数"] > 3000, "事件统计异常"
    s = search_text("破釜沉舟")
    assert s["hits"] and "项羽" in json.dumps(s["hits"], ensure_ascii=False), "向量检索失败"
    print(f"[selfcheck] 6/6 通过 (语料 {len(_load_passages())} 段)")

# ---------- CLI ----------

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selfcheck":
        selfcheck(); sys.exit(0)
    if args and args[0] == "--demo":
        selfcheck()
        for q in ["长平之战发生在哪一年? 秦赵双方主帅是谁?",
                  "对比《秦始皇本纪》和《项羽本纪》中关于秦始皇出游会稽的记载是否一致"]:
            print(f"\n=== 问: {q}")
            print(f"答: {ask(q, verbose=True)}")
        sys.exit(0)
    if args and args[0] == "--stats":
        ev = tool_events()
        print(json.dumps(ev, ensure_ascii=False, indent=1))
        print("\n实体索引类型:", ", ".join(list(load_json('kg/entity_index.json').keys())))
        sys.exit(0)
    if args and args[0] == "--web":
        _web(); sys.exit(0)
    q = " ".join(args) or input("问《史记》: ")
    print(ask(q, verbose=True))

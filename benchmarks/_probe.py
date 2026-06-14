import time, json, sys
from sochdb import QueryLanes, SochDBClient, create_agent_memory
sys.path.insert(0, "benchmarks")
from run_rag_e2e import embed_texts, chat, chunk_context

raw = json.load(open("/root/locomo/data/locomo10.json"))
it = raw[0]
conv = it["conversation"]
sk = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
c = SochDBClient("127.0.0.1:50061")

def T(msg, t0):
    print(f"{msg}: {time.perf_counter()-t0:.2f}s", flush=True)

t = time.perf_counter()
m = create_agent_memory(c, namespace="probe1", token_limit=8192)
T("create_memory", t)

t = time.perf_counter()
for s in sk:
    turns = conv.get(s) or []
    txt = "\n".join(f"{x.get('speaker')}: {x.get('text')}" for x in turns)
    if txt:
        m.write_episode(txt, metadata={"s": s})
T(f"ingest {len(sk)} sessions", t)

qa = it["qa"][0]
q = qa["question"]
print("Q:", q, flush=True)

t = time.perf_counter()
res = m.search(q, lanes=QueryLanes.LEXICAL, token_limit=8192)
T("search", t)
print("ctx_chars", len(res.context or ""), "tokens", res.total_tokens, flush=True)

t = time.perf_counter()
ch = chunk_context(res.context or "")
v = embed_texts([q] + ch, "http://localhost:11434/v1", "nomic-embed-text")
T(f"ollama embed {len(ch)+1}", t)

t = time.perf_counter()
pred, u = chat("http://100.127.255.44:8000/v1", "nvidia/Qwen3.6-35B-A3B-NVFP4",
               "Answer concisely from context.",
               f"Context:\n{(res.context or '')[:24000]}\n\nQ: {q}\nA:", 128)
T("qwen_answer", t)
print("PRED:", repr(pred)[:160], flush=True)
print("ALL_OK", flush=True)

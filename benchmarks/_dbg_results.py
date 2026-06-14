from sochdb import SochDBClient, open_collection
from fastembed import TextEmbedding

emb = TextEmbedding("BAAI/bge-small-en-v1.5")
docs = ["the cat sat on the mat", "dogs run fast in the green park daily", "birds migrate south in winter"]
vecs = [v.tolist() for v in emb.embed(docs)]
qv = list(emb.embed(["where do dogs run"]))[0].tolist()

c = SochDBClient("127.0.0.1:50061")
# INDEX trio: create_index + insert_vectors + search
c.create_index("dbgidx", dimension=len(vecs[0]))
ins = c.insert_vectors("dbgidx", ids=[10, 11, 12], vectors=vecs)
print("grpc index inserted:", ins)
res = c.search("dbgidx", qv, k=3)
print("grpc search type:", type(res), "len", len(res))
for d in res:
    attrs = [a for a in dir(d) if not a.startswith("_")]
    print("  SearchResult attrs:", attrs)
    print("  SearchResult vars:", {a: getattr(d, a) for a in attrs if not callable(getattr(d, a))})
    break

print("=== embedded vector_search + keyword_search ===")
ec = open_collection("dbg2", path=":memory:", dimension=len(vecs[0]))
ec.insert_batch(documents=[(10, vecs[0], {"dia": "D1:0"}, docs[0]),
                           (11, vecs[1], {"dia": "D1:1"}, docs[1]),
                           (12, vecs[2], {"dia": "D1:2"}, docs[2])])
print("embedded count:", ec.count())
for label, fn in [("keyword", lambda: ec.keyword_search("park dogs run", k=3)),
                  ("vector", lambda: ec.vector_search(qv, k=3)),
                  ("hybrid", lambda: ec.hybrid_search(qv, "park dogs run", k=3))]:
    try:
        r = fn()
        items = list(r) if hasattr(r, "__iter__") else getattr(r, "results", [])
        print(f"{label}: n={len(items)}")
        if items:
            it = items[0]
            attrs = [a for a in dir(it) if not a.startswith("_")]
            print("  item attrs:", attrs)
            print("  item vars:", {a: getattr(it, a) for a in attrs if not callable(getattr(it, a))})
    except Exception as e:
        print(f"{label} ERR: {e!r}")

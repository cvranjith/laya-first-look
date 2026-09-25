"""Batch-categorize 100 synthetic YouTube videos with Laya and Jev, and score
both against the authored ground truth (`true_category` in the dataset).

Laya: one `predict_batch` call over all 100 states (true batching, one or a
few forward passes on local hardware).

Jev: the API only batches *questions* within one state, not multiple states
per call (see README) — so classifying 100 videos is inherently 100 HTTP
requests. We report three numbers for it, since a naive sequential loop
badly understates what the service can actually do:

  1. sequential (warm connection) -- one request at a time, reusing a single
     TCP/TLS connection instead of opening a fresh one per call.
  2. concurrent -- the same 100 requests fired with a thread pool, which is
     the realistic way you'd actually call an HTTP API for a batch job.
  3. server-side only -- the `x-envoy-upstream-service-time` response header,
     i.e. what TypeSafe's own gateway says its inference took, with the
     network round trip stripped out entirely.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from laya import Router
from jev_client import predict as jev_predict

CONCURRENCY = 10

CATEGORIES = ["News", "AI News", "Tech News", "Entertainment", "Comedy", "Music", "Tutorial", "Other"]
CATEGORY_HINTS = {
    "News": "general world/national news, politics, weather, events — not specifically about AI or tech products",
    "AI News": "news about AI models, AI companies, or AI research specifically",
    "Tech News": "news about tech products, gadgets, software, or companies — not AI-specific",
    "Entertainment": "movies, TV shows, celebrities, awards shows",
    "Comedy": "sketches, pranks, comedic bits meant purely to be funny",
    "Music": "music videos, live performances, official audio",
    "Tutorial": "how-to or instructional content teaching a skill or task",
    "Other": "anything that doesn't clearly fit the other categories",
}

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which category best fits this YouTube video, based on its title and description?",
        "criteria": CATEGORY_HINTS,
    }
}

with open("data/youtube_dataset.json") as f:
    dataset = json.load(f)

states = [
    {"title": v["title"], "description": v["description"], "channel_name": v["channel_name"]}
    for v in dataset
]
truth = [v["true_category"] for v in dataset]

# --- Laya: one batched call over all 100 states ---
router = Router()
router.predict(states[0], QUESTIONS)  # warm up, exclude cold-start from the timed batch

laya_requests = [{"state": s, "questions": QUESTIONS} for s in states]
t0 = time.perf_counter()
laya_results = router.predict_batch(laya_requests)
laya_elapsed = time.perf_counter() - t0
laya_preds = [r["answers"]["category"]["choice"] for r in laya_results]

# --- Jev: warm up the connection once (pay TCP/TLS setup outside the timed runs) ---
jev_predict(states[0], QUESTIONS)


def call_jev(state):
    try:
        result, elapsed, server_ms = jev_predict(state, QUESTIONS)
        return result["answers"]["category"]["choice"], elapsed, server_ms
    except Exception:
        return None, None, None


# 1. Sequential, warm connection
t0 = time.perf_counter()
seq_results = [call_jev(s) for s in states]
jev_sequential_elapsed = time.perf_counter() - t0
jev_preds = [r[0] for r in seq_results]
jev_errors = sum(1 for r in seq_results if r[0] is None)
server_times = [r[2] for r in seq_results if r[2] is not None]

# 2. Concurrent, same connection pool
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    concurrent_results = list(pool.map(call_jev, states))
jev_concurrent_elapsed = time.perf_counter() - t0
concurrent_preds = [r[0] for r in concurrent_results]


def accuracy(preds):
    correct = sum(1 for p, t in zip(preds, truth) if p and p.lower() == t.lower())
    return correct / len(truth)


def per_category_accuracy(preds):
    from collections import defaultdict
    correct, total = defaultdict(int), defaultdict(int)
    for p, t in zip(preds, truth):
        total[t] += 1
        if p and p.lower() == t.lower():
            correct[t] += 1
    return {c: (correct[c], total[c]) for c in CATEGORIES}


avg_server_ms = sum(server_times) / len(server_times) if server_times else None
mismatches_vs_sequential = sum(1 for a, b in zip(jev_preds, concurrent_preds) if a != b)

print("=" * 78)
print(f"Laya:                {laya_elapsed:6.2f}s total ({laya_elapsed/len(states)*1000:6.1f} ms/video)  "
      f"one predict_batch() call, accuracy={accuracy(laya_preds):.1%}")
print(f"Jev (sequential):    {jev_sequential_elapsed:6.2f}s total ({jev_sequential_elapsed/len(states)*1000:6.1f} ms/video)  "
      f"warm connection, {jev_errors} errors, accuracy={accuracy(jev_preds):.1%}")
print(f"Jev (concurrent x{CONCURRENCY}): {jev_concurrent_elapsed:6.2f}s total ({jev_concurrent_elapsed/len(states)*1000:6.1f} ms/video)  "
      f"thread pool, {mismatches_vs_sequential} preds differ from sequential run")
if avg_server_ms is not None:
    print(f"Jev server-side only: avg {avg_server_ms:.1f} ms/video "
          f"(min {min(server_times):.1f}, max {max(server_times):.1f}) via x-envoy-upstream-service-time header")
print("=" * 78)

print(f"\n{'Category':<15}{'Laya':>12}{'Jev':>12}")
for cat in CATEGORIES:
    lc, lt = per_category_accuracy(laya_preds)[cat]
    jc, jt = per_category_accuracy(jev_preds)[cat]
    print(f"{cat:<15}{f'{lc}/{lt}':>12}{f'{jc}/{jt}':>12}")

disagreements = [
    (dataset[i]["title"], truth[i], laya_preds[i], jev_preds[i])
    for i in range(len(states)) if laya_preds[i] != jev_preds[i]
]
print(f"\n{len(disagreements)} disagreements between Laya and Jev:")
for title, true_cat, lp, jp in disagreements[:15]:
    print(f"  [{true_cat}] {title!r}: Laya={lp!r} Jev={jp!r}")

with open("data/batch_results.json", "w") as f:
    json.dump({
        "laya": {"elapsed_s": laya_elapsed, "accuracy": accuracy(laya_preds), "preds": laya_preds},
        "jev": {
            "sequential_elapsed_s": jev_sequential_elapsed,
            "concurrent_elapsed_s": jev_concurrent_elapsed,
            "concurrency": CONCURRENCY,
            "avg_server_ms": avg_server_ms,
            "accuracy": accuracy(jev_preds),
            "preds": jev_preds,
            "errors": jev_errors,
        },
        "truth": truth,
    }, f, indent=2)

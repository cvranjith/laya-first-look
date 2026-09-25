# Laya — Initial Testing (Base Model, No Tuning)

A quick, informal exploration of [Laya](https://github.com/NandhaKishorM/laya), the open-source
"System 1" decision engine from Nandakishor Mukkunnoth / Convai Innovations, run locally on a
Mac mini (Apple M4). All checkpoints were used exactly as shipped — **no temperature
recalibration, fine-tuning, or config changes**. This is not a rigorous benchmark, just a
show-of-possibility writeup: what it is, how it behaves, and some rough latency numbers on
consumer Apple Silicon.

Laya is a bidirectional encoder that answers typed questions (`choice`, `score`, `noul`
i.e. yes/no) about a piece of text in a single forward pass — it does not generate text, so
there's nothing for it to hallucinate on those answers. It was released under Apache 2.0 shortly
after TypeSafe AI's closed commercial model, Jev.

## Machine / environment

| | |
|---|---|
| Machine | Mac mini (Mac16,10) |
| Chip | Apple M4 (10 cores: 4 performance + 6 efficiency) |
| Memory | 16 GB unified |
| OS | macOS 26.6.2 (build 25G83) |
| Python | 3.10.18 |
| Inference device | `mps` (Apple GPU via Metal, auto-selected by Laya) |

Package versions (see [`requirements.txt`](requirements.txt)):

| Package | Version |
|---|---|
| `laya` | 0.3.20 |
| `torch` | 2.14.0 |
| `transformers` | 5.17.0 |
| `huggingface_hub` | 1.33.0 |

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Each script below downloads its checkpoint from Hugging Face on first run (cached afterward)
and can be run directly, e.g. `.venv/bin/python test_laya.py`.

## Checkpoints used

Laya ships three checkpoints; the router auto-selects between `english` and `multilingual`
based on detected script, and `typed-decisions` only when explicitly requested.

| Checkpoint | Params | Backbone | Context | Notes |
|---|---|---|---|---|
| `english` | 421M | ModernBERT-large | 512 tokens | Default for English/Latin script |
| `multilingual` | 322M | mmBERT-base | 1,024–8,192 tokens | 100+ languages, auto-routed by script detection |
| `typed-decisions` | 421M | ModernBERT-large | 1,024 tokens | Fine-tuned on 4 fixed workflows; opt-in only |

## Test cases and results

### 1. Basic decision — [`test_laya.py`](test_laya.py)

Input: *"We were billed twice for March. Please refund or we will cancel."*

| Question | Type | Answer | Confidence |
|---|---|---|---|
| `department` | choice | **billing** | 98.4% (technical 1.0%, other 0.5%) |
| `churn_risk` | yes/no | **0.966** (high) | — |

### 2. Support ticket triage — [`test_ticket.py`](test_ticket.py)

Input: *"My laptop screen is flickering since the last update, and I'm on a business trip
until Friday — need this fixed urgently."*

| Question | Type | Answer | Confidence |
|---|---|---|---|
| `department` | choice | **technical** | 85.9% (lower overall confidence: 54%, since it's arguably hardware) |
| `urgency` | score (0–4) | **3.36** — leaning "extremely urgent" | 60.7% on top level |
| `is_hardware_issue` | yes/no | **0.770** (yes) | — |

### 3. Multilingual routing — [`test_multilingual.py`](test_multilingual.py)

Input (Hindi): *"मेरा ऑर्डर दो हफ्ते पहले आना था और अभी तक नहीं आया। अगर जल्दी नहीं आया तो
मैं पैसे वापस चाहता हूँ।"* ("My order was supposed to arrive two weeks ago and still hasn't.
If it doesn't arrive soon, I want a refund.")

| Question | Type | Answer | Confidence |
|---|---|---|---|
| `department` | choice | **billing** | 63.5% (shipping 32.4%, other 4.1%) |
| `wants_refund` | yes/no | **0.980** (yes) | — |

The router detected the Devanagari script and switched checkpoints automatically:
`reason: "non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it"`.
No code change was needed — same script, same call, different checkpoint loaded on demand.

### 4. Typed-decisions checkpoint — [`test_typed_decisions.py`](test_typed_decisions.py)

Input: *"This is the third time your app has logged me out mid-purchase this month. I've
wasted an hour on this. Fix it or I'm switching to a competitor and telling everyone why."*

Forced via `model="typed-decisions"` (this checkpoint is never auto-selected).

| Question | Type | Answer | Confidence |
|---|---|---|---|
| `category` | choice | **technical** | 48.8% (billing 33.6%, other 17.6% — genuinely ambiguous) |
| `action` | choice | **escalate** | 70.2% |
| `urgency` | score (0–2) | **1.50** — split between "needs attention soon" and "blocking issue" | 51.2% on top level |
| `churn_risk` | yes/no | **0.822** (yes) | — |
| `needs_human` | yes/no | **0.540** (borderline yes) | — |

### 5. Cold start vs. warm latency — [`test_timing.py`](test_timing.py)

`Router()` construction is instant (lazy) — all cost lands on the first `predict()` call,
which pays for disk load, moving weights to the MPS device, and one-time Metal kernel
compilation for that input shape.

| Call | Latency |
|---|---|
| `Router()` construction | 0.0 ms |
| 1st `predict()` (cold) | **5,766 ms** |
| 2nd `predict()` | 47.8 ms |
| 3rd–8th `predict()` | 44.7–52.8 ms (flat, no further warm-up) |

Takeaway: pay the ~5.8s cold-start once per process/checkpoint, keep the process alive for
repeated calls (this is exactly what the optional `laya[serve]` HTTP daemon is for).

### 6. Multiple questions per call — [`test_question_scaling.py`](test_question_scaling.py)

Multiple questions in one `predict()` call are not looped sequentially — each question becomes
one row of a shared batch tensor, and the whole batch goes through **a single forward pass**.
Per-question cost drops as the batch grows, confirming batched parallelism rather than a loop:

| Questions per call | Total latency (avg of 5) | Per-question |
|---|---|---|
| 1 | 33.2 ms | 33.2 ms |
| 2 | 92.5 ms | 46.3 ms |
| 4 | 110.2 ms | 27.6 ms |
| 8 | 235.1 ms | 29.4 ms |
| 16 | 292.9 ms | 18.3 ms |
| 32 | 508.4 ms | 15.9 ms |

Part of the drop past 4→8 questions is a documented optimization in Laya itself: it only
enables fp16 autocast on MPS once the batch reaches a minimum row count (fp16 loses to fp32
on very small batches on Apple GPUs), so the gain isn't purely from amortized overhead.

## Caveats

- On every run, the library printed: *"this checkpoint ships invalid temperatures or values
  outside [0.5, 5]... treat confidence from the affected entries as uncalibrated."* Choice
  confidence values above should be read with that in mind — Laya is telling you itself that
  calibration on the shipped base checkpoints isn't fully trustworthy yet.
- Single machine, single run per test case, no statistical averaging beyond what's noted above.
  Meant to show what's possible, not to be a formal benchmark.
- `typed-decisions` was tested with a custom schema, not one of its four intended workflows
  exactly — it was force-selected via `model=` rather than auto-routed.

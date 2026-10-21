#!/usr/bin/env python3
"""Perf sweep for an OpenAI-compatible TensorFold server. Stdlib only.

python bench.py --preset quick|decode|prefill|cache|long|all [--url ...] [--reps 3]
python bench.py --selftest
"""
import argparse, csv, json, os, random, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8888"
MODEL = "GLM-5.3-Flash-EXL3"
CSV = "results.csv"

CONTENT = {  # MTP acceptance varies by content, so each gets a natural (non-ignore_eos) prompt
    "prose": "Write a long, detailed short story about a lighthouse keeper. At least 1500 words.",
    "code": "Write a complete, well-commented Python implementation of a red-black tree with insert, delete, search and an in-order iterator, plus unit tests.",
    "json": "Output a JSON array of 60 fictional employees, each with id, name, email, department, salary, start_date, skills (list) and manager_id. Output only JSON.",
    "math": "Solve step by step, showing all work: find all real solutions of x^4 - 10x^3 + 35x^2 - 50x + 24 = 0, then prove your answer is complete, then compute the sum of squares of the roots two different ways.",
    "translate": "Translate the following into French, German and Spanish, paragraph by paragraph:\n\n" + " ".join(
        ["The committee met on Tuesday to discuss the budget for next year, and after a long debate, agreed to increase funding for public libraries while reducing spending on road maintenance."] * 12),
}

WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part "
         "place case week company system program question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house service friend father power hour game line").split()


def filler(n_tokens, seed, ratio):
    rnd = random.Random(seed)
    return f"[doc {seed}] " + " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens / ratio)))


def tok_ratio(url):
    text = " ".join(random.Random(0).choice(WORDS) for _ in range(2000))
    r = post(url + "/tokenize", {"model": MODEL, "prompt": text})
    return json.load(r)["count"] / 2000


def post(url, body, timeout=7200):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def parse_stream(lines, t0, clock=time.perf_counter):
    """Consume SSE lines; return metrics dict. Split out so --selftest can feed canned chunks."""
    ttft, final = None, {}
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        delta = (d.get("choices") or [{}])[0].get("delta", {})
        if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
            ttft = clock() - t0
        if "usage" in d:
            final = d
    u, tf = final.get("usage", {}), final.get("tensorfold", {})
    m = {
        "ttft_s": ttft, "e2e_s": clock() - t0,
        "prompt_tok": u.get("prompt_tokens"), "out_tok": u.get("completion_tokens"),
        "cached_tok": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prefill_s": tf.get("prefill_s"), "decode_s": tf.get("decode_s"),
        "drafted": tf.get("drafted"), "accepted": tf.get("accepted"),
    }
    m["prefill_tps"] = m["prompt_tok"] / m["prefill_s"] if m["prefill_s"] else None
    m["decode_tps"] = m["out_tok"] / m["decode_s"] if m["decode_s"] else None
    m["accept_rate"] = m["accepted"] / m["drafted"] if m["drafted"] else None
    return m


def one(url, prompt, max_tokens, ignore_eos, think):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.perf_counter()
    try:
        with post(url + "/v1/chat/completions", body) as r:
            return parse_stream(r, t0)
    except Exception as e:  # keep the sweep going; the error lands in the CSV
        return {"error": repr(e)[:200], "e2e_s": time.perf_counter() - t0}


def wait_idle(url):
    for _ in range(600):
        if not json.load(urllib.request.urlopen(url + "/health", timeout=10)).get("busy"):
            return
        time.sleep(1)


def cells(preset):
    """Yield scenario dicts: scenario, content, conc, prompt_len, max_tokens, ignore_eos, think, cache."""
    base = dict(content="prose", conc=1, prompt_len=0, max_tokens=512, ignore_eos=True, think=False, cache="cold")
    if preset in ("quick", "all"):
        for c in (1, 4):
            for p in (1024, 8192):
                yield {**base, "scenario": "quick", "conc": c, "prompt_len": p}
    if preset in ("decode", "all"):
        for ct in CONTENT:
            for c in (1, 2, 4, 8):
                yield {**base, "scenario": "decode", "content": ct, "conc": c, "max_tokens": 1024, "ignore_eos": False}
        for c in (1, 4):
            yield {**base, "scenario": "decode", "conc": c, "max_tokens": 1024, "ignore_eos": False, "think": True}
    if preset in ("prefill", "all"):
        for p in (128, 1024, 8192, 32768, 131072):
            yield {**base, "scenario": "prefill", "prompt_len": p, "max_tokens": 64}
        for c in (2, 4):
            yield {**base, "scenario": "prefill", "conc": c, "prompt_len": 32768, "max_tokens": 64}
    if preset in ("cache", "all"):
        for p in (8192, 32768, 131072):
            yield {**base, "scenario": "cache", "prompt_len": p, "max_tokens": 64, "cache": "warm"}
    if preset == "long":
        for p in (262144, 524288, 1000000):
            yield {**base, "scenario": "long", "prompt_len": p, "max_tokens": 64}


def run_cell(url, cell, rep, ratio):
    """Returns list of row dicts. cache=warm sends a cold then warm request with the same prefix."""
    def prompt(i):
        task = CONTENT[cell["content"]]
        if not cell["prompt_len"]:
            return task
        seed = f"{time.time_ns()}-{i}"  # unique per request so nothing hits the prefix cache by accident
        return filler(cell["prompt_len"], seed, ratio) + "\n\nIgnore the text above. " + task

    if cell["cache"] == "warm":
        p = prompt(0)
        rows = []
        for phase in ("cold", "warm"):
            t0 = time.perf_counter()
            m = one(url, p, cell["max_tokens"], cell["ignore_eos"], cell["think"])
            rows.append({**cell, "cache": phase, "rep": rep, "wall_s": time.perf_counter() - t0, **m})
        return rows
    t0 = time.perf_counter()
    with ThreadPoolExecutor(cell["conc"]) as ex:
        ms = list(ex.map(lambda i: one(url, prompt(i), cell["max_tokens"], cell["ignore_eos"], cell["think"]), range(cell["conc"])))
    wall = time.perf_counter() - t0
    agg = sum(m.get("out_tok") or 0 for m in ms) / wall
    return [{**cell, "rep": rep, "wall_s": wall, "agg_tps": agg, **m} for m in ms]


FIELDS = ["scenario", "content", "conc", "prompt_len", "max_tokens", "ignore_eos", "think", "cache", "rep",
          "ttft_s", "e2e_s", "wall_s", "agg_tps", "prompt_tok", "out_tok", "cached_tok", "prefill_s", "decode_s",
          "prefill_tps", "decode_tps", "drafted", "accepted", "accept_rate", "error"]
KEY = ["scenario", "content", "conc", "prompt_len", "max_tokens", "think", "cache"]


def summarize(rows):
    def med(vals):
        vals = [v for v in vals if v is not None]
        return f"{statistics.median(vals):.2f}" if vals else "-"

    def p90(vals):
        vals = sorted(v for v in vals if v is not None)
        return f"{vals[int(0.9 * (len(vals) - 1))]:.2f}" if vals else "-"

    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in KEY), []).append(r)
    hdr = KEY + ["n", "err", "ttft_med", "ttft_p90", "prefill_tps", "decode_tps", "agg_tps", "accept"]
    out = [hdr]
    for k, rs in groups.items():
        g = lambda f: [r.get(f) for r in rs if not r.get("error")]
        out.append([str(x) for x in k] + [str(len(rs)), str(sum(1 for r in rs if r.get("error"))),
                                          med(g("ttft_s")), p90(g("ttft_s")), med(g("prefill_tps")),
                                          med(g("decode_tps")), med(g("agg_tps")), med(g("accept_rate"))])
    w = [max(len(row[i]) for row in out) for i in range(len(hdr))]
    for row in out:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(row)))


def selftest():
    chunks = [
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"reasoning_content": "The"}}]}',
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "length"}], "tensorfold": {"prefill_s": 0.1141, "decode_s": 0.393, "drafted": 25, "accepted": 22, "cached": 0}, "usage": {"prompt_tokens": 13, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 0}}}',
        "data: [DONE]",
    ]
    ticks = iter([1.0, 2.0])  # first token at t=1, end at t=2
    m = parse_stream(chunks, 0.0, clock=lambda: next(ticks))
    assert m["ttft_s"] == 1.0 and m["e2e_s"] == 2.0, m
    assert abs(m["decode_tps"] - 30 / 0.393) < 1e-9 and abs(m["prefill_tps"] - 13 / 0.1141) < 1e-9
    assert m["accept_rate"] == 22 / 25 and m["out_tok"] == 30 and m["cached_tok"] == 0
    assert len(filler(100, "x", 1.0).split()) == 102
    names = [c["scenario"] for c in cells("all")]
    assert {"quick", "decode", "prefill", "cache"} == set(names) and "long" not in names
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="quick", choices=["quick", "decode", "prefill", "cache", "long", "all"])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    ratio = tok_ratio(a.url)
    print(f"tokens/word = {ratio:.3f}; warming up")
    one(a.url, "hi", 16, True, False)
    new = not os.path.exists(a.csv)
    rows = []
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for cell in cells(a.preset):
            for rep in range(a.reps):
                wait_idle(a.url)
                rs = run_cell(a.url, cell, rep, ratio)
                w.writerows(rs)
                f.flush()
                rows += rs
                r0 = rs[-1]
                print(f"{cell['scenario']:7} {cell['content']:9} c={cell['conc']} p={cell['prompt_len']:>7} rep={rep} "
                      f"ttft={r0.get('ttft_s') or 0:.2f}s decode={r0.get('decode_tps') or 0:.1f}t/s "
                      f"agg={r0.get('agg_tps') or 0:.1f}t/s {r0.get('error', '')}", flush=True)
    print()
    summarize(rows)


if __name__ == "__main__":
    sys.exit(main())

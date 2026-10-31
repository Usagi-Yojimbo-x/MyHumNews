#!/usr/bin/env python3
"""Live progress for the long-running jobs on this Mac (and the VM crawlers).

    .venv/bin/python progress.py            # one snapshot
    .venv/bin/python progress.py --watch    # refresh every 10s until Ctrl-C
"""
from __future__ import annotations
import argparse, json, os, re, sqlite3, subprocess, sys, time

W = 34

def bar(done, total, width=W):
    if not total:
        return "[" + " " * width + "]    ?"
    f = min(max(done / total, 0.0), 1.0)
    n = int(f * width)
    return "[" + "#" * n + "." * (width - n) + f"] {f*100:5.1f}%"

def pid_alive(pat):
    try:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
        return bool(out.strip())
    except Exception:
        return False

def crawl_row(site, db):
    if not os.path.exists(db):
        return None
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        st = dict(c.execute("select status,count(*) from urls group by status"))
    except sqlite3.Error:
        return None
    done = st.get("done", 0) + st.get("skipped", 0) + st.get("failed", 0)
    total = sum(st.values())
    return site, done, total, st.get("pending", 0)

def index_row(out="rag/index"):
    meta = os.path.join(out, "meta.jsonl")
    if not os.path.exists(meta):
        return None
    done = sum(1 for _ in open(meta, "rb"))
    total = 0
    log = "logs/rag_index.log"
    if os.path.exists(log):
        m = re.findall(r"(\d+) articles to embed", open(log, errors="replace").read())
        if m:
            total = int(m[-1])
    return done, total

def rate_eta(key, done, total, state):
    now = time.time()
    prev = state.get(key)
    state[key] = (now, done)
    if not prev:
        return ""
    dt, dn = now - prev[0], done - prev[1]
    if dt <= 0 or dn <= 0:
        return "   idle"
    r = dn / dt
    left = max(total - done, 0)
    return f" {r:5.2f}/s  eta {left/r/60:5.1f}m"

def render(state):
    lines = []
    lines.append("  LOCAL (this Mac)")
    r = crawl_row("irrawaddy", "data/irrawaddy/crawl.db")
    if r:
        s, d, t, p = r
        live = "running" if pid_alive("run.py irrawaddy") else "STOPPED"
        lines.append(f"    crawl {s:<11} {bar(d,t)}  {d:>6}/{t:<6} {live}"
                     + rate_eta("irr", d, t, state))
    ix = index_row()
    if ix:
        d, t = ix
        live = "running" if pid_alive("rag.build_index") else "STOPPED"
        lines.append(f"    bge-m3 index    {bar(d,t)}  {d:>6}/{t:<6} {live}"
                     + rate_eta("idx", d, t, state))
    lines.append("")
    lines.append("  VM (azure)")
    try:
        out = subprocess.run(
            ["ssh", "-i", "sentinel_key704.pem", "-o", "ConnectTimeout=10",
             "-o", "BatchMode=yes", "azureuser@70.153.9.116",
             "cd ~/MyHumNews && for s in dvb rfa myanmarnow; do "
             "python3 -c \"import sqlite3,sys;c=sqlite3.connect('data/$s/crawl.db');"
             "print('$s',*[f'{k}={v}' for k,v in c.execute('select status,count(*) from urls group by status')])\"; done"],
            capture_output=True, text=True, timeout=25).stdout.strip()
        for ln in out.splitlines():
            parts = ln.split()
            site = parts[0]
            st = dict(x.split("=") for x in parts[1:])
            st = {k: int(v) for k, v in st.items()}
            d = st.get("done", 0) + st.get("skipped", 0) + st.get("failed", 0)
            t = sum(st.values())
            lines.append(f"    crawl {site:<11} {bar(d,t)}  {d:>6}/{t:<6}"
                         + rate_eta(site, d, t, state))
    except Exception as e:
        lines.append(f"    (vm unreachable: {type(e).__name__})")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    state = {}
    if not a.watch:
        print(render(state)); return 0
    try:
        while True:
            out = render(state)
            print("\033[2J\033[H" + time.strftime("  %H:%M:%S") + "\n" + out, flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

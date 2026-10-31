#!/usr/bin/env python3
"""
Hand-label the sampled articles. One keystroke per article.

    python -m classifier.label_cli

Resumable: every label is appended and flushed immediately, so Ctrl-C or a
closed terminal loses nothing and restarting picks up where you stopped. Same
rule the crawler follows.

It never shows you the model's prediction. That is the whole point -- see the
note in sample_for_labelling.py. If you want to know how you did, that is
score_labels.py, afterwards.

Keys
  1 conflict      6 health            m  more of the body text
  2 politics      7 education         b  back / undo last label
  3 crime         8 sport             s  skip (boilerplate, unlabelable)
  4 economy       9 culture           q  save and quit
  5 environment   0 international     ?  this help

Label what the article is ABOUT, by the taxonomy in classifier/taxonomy.py.
Where two apply, RESOLUTION_ORDER says the more specific one wins -- an article
about fighting that is also about politics is conflict. Use 's' freely for
programme announcements and other non-articles; they are counted separately and
do not drag the score down.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))

# Order matches classifier.taxonomy.CLASSES; digits beat mnemonic letters when
# there are ten options and several share a first letter.
KEYS = {
    "1": "conflict", "2": "politics", "3": "crime", "4": "economy",
    "5": "environment", "6": "health", "7": "education", "8": "sport",
    "9": "culture", "0": "international",
}

HELP = """
  1 conflict      6 health            m  more of the body text
  2 politics      7 education         b  back / undo last label
  3 crime         8 sport             s  skip (not an article)
  4 economy       9 culture           a  ambiguous (taxonomy unclear)
  5 environment   0 international     b  undo     q  quit     ? help
"""


def getch() -> str:
    """One keystroke, no Enter. Falls back to line input when piped."""
    if not sys.stdin.isatty():
        ch = sys.stdin.read(1)
        return ch or "q"
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
        return "q"
    return ch


def wrap(s: str, width: int) -> list[str]:
    # break_long_words matters: Burmese has no spaces, so the whole paragraph is
    # one "word" and a word-aware wrapper would emit a single endless line.
    out = []
    for para in s.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.extend(textwrap.wrap(para, width=width,
                                 break_long_words=True,
                                 break_on_hyphens=False) or [""])
    return out


def load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_done(path: str) -> dict[str, dict]:
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[r["url"]] = r
    return done


def rewrite(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=os.path.join(HERE, "data", "label_task.jsonl"))
    p.add_argument("--out", default=os.path.join(HERE, "data", "labels.jsonl"))
    p.add_argument("--chars", type=int, default=700,
                   help="body characters shown before 'm'")
    a = p.parse_args()

    if not os.path.exists(a.task):
        print("no task file -- run: python -m classifier.sample_for_labelling",
              file=sys.stderr)
        return 1

    tasks = load_tasks(a.task)
    done = load_done(a.out)
    order = [t for t in tasks if t["url"] not in done]

    print(f"{len(tasks)} sampled, {len(done)} already labelled, "
          f"{len(order)} to go")
    if not order:
        print("all done -- next: python -m classifier.score_labels")
        return 0
    print(HELP)
    print("  press any key to start, q to quit")
    getch()

    t_start = time.time()
    n_session = 0
    show = a.chars

    i = 0
    while i < len(order):
        t = order[i]
        cols = min(shutil.get_terminal_size((100, 30)).columns - 2, 100)

        os.system("clear")
        n_left = len(order) - i
        rate = (time.time() - t_start) / n_session if n_session else 0
        eta = f"  ~{n_left * rate / 60:.0f} min left" if rate else ""
        print(f"  [{len(done) + 1}/{len(tasks)}]  {n_left} to go{eta}")
        print(f"  {t['published_date'][:10]}   {t['n_chars']} chars"
              + (f"   tags: {', '.join(t['tags'][:6])}" if t["tags"] else ""))
        print("  " + "-" * cols)
        for ln in wrap(t["title"], cols - 2):
            print(f"  {ln}")
        print("  " + "-" * cols)
        body = t["body"][:show]
        for ln in wrap(body, cols - 2)[:40]:
            print(f"  {ln}")
        if len(t["body"]) > show:
            print(f"  ... [{len(t['body']) - show} more chars -- press m]")
        print("  " + "-" * cols)
        print(HELP)

        ch = getch().lower()

        if ch == "q":
            break
        if ch == "?":
            continue
        if ch == "m":
            show = len(t["body"]) if show == a.chars else a.chars
            continue
        if ch == "b":
            if done:
                rows = list(done.values())
                last = rows[-1]
                del done[last["url"]]
                rewrite(a.out, list(done.values()))
                # put it back at the front of the queue
                back = next((x for x in tasks if x["url"] == last["url"]), None)
                if back is not None:
                    order.insert(i, back)
                n_session = max(0, n_session - 1)
            continue

        if ch == "s":
            label, reason = None, "skip"
        elif ch == "a":
            # Not the same as a skip: this IS an article, the taxonomy just
            # does not decide it. Guessing here would be indistinguishable
            # from a model error. See LABELLING_GUIDE.md.
            label, reason = None, "ambiguous"
        elif ch in KEYS:
            label, reason = KEYS[ch], None
        else:
            continue

        row = {"url": t["url"], "i": t["i"], "label": label, "reason": reason,
               "published_date": t["published_date"],
               "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done[t["url"]] = row
        show = a.chars
        n_session += 1
        i += 1

    os.system("clear")
    lab = sum(1 for r in done.values() if r["label"])
    amb = sum(1 for r in done.values() if r.get("reason") == "ambiguous")
    skip = sum(1 for r in done.values()
               if not r["label"] and r.get("reason") != "ambiguous")
    print(f"  labelled {lab}, ambiguous {amb}, skipped {skip}, "
          f"{len(tasks) - len(done)} remaining")
    print(f"  saved to {a.out}")
    if len(done) == len(tasks):
        print("\n  all done -- next: python -m classifier.score_labels")
    else:
        print("\n  resume any time: python -m classifier.label_cli")
    return 0


if __name__ == "__main__":
    sys.exit(main())

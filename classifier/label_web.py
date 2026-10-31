#!/usr/bin/env python3
"""
Browser-based labelling. Same job as label_cli, but readable.

    python -m classifier.label_web

Then open http://127.0.0.1:8765 -- it prints the URL.

WHY A BROWSER AND NOT THE TERMINAL:
terminals do not shape complex scripts. Burmese stacks combining marks above,
below and around its consonants (U+1037 dot-below, U+103A asat, medials, and
so on), and a terminal that places each codepoint in its own fixed cell renders
that as disconnected fragments. You cannot reliably label an article you cannot
read, and a labeller squinting at broken glyphs produces noise, not ground
truth. A browser hands the text to the OS text engine, which shapes it
correctly. This Mac has Noto Sans Myanmar in /System/Library/Fonts, so the page
needs no webfont and no network.

HOSTED MODE (several annotators over the internet):

    python -m classifier.label_web --multi --host 0.0.0.0 --port 8765 \
        --token <shared-secret> --outdir classifier/data/annot \
        --task classifier/data/humset_task.jsonl

Adds what a shared deployment needs and local mode does not:

  --token      every request must carry it (?t=... once, then a cookie).
               Not real authentication -- it stops a stray crawler or a
               mistyped URL, nothing more. Anyone holding the token can label.
  --outdir     one JSONL per annotator, so labels stay attributable and
               agreement is computable. Local mode's single --out is unchanged.
  scoped undo  `b` only removes the caller's own last label. In local mode undo
               was global, which would let one annotator delete another's work.

NO PARTITIONING: every annotator is served the WHOLE task file, in the same
order. Whatever two people both reach is double-labelled, so inter-annotator
agreement is measured on everything they have in common rather than on a
reserved subset -- and that common set grows as they work instead of being
fixed at 80 articles up front.

The cost is duplicated effort: two annotators working through 383 articles
produce 383 double-labelled articles, not 766 singly-labelled ones. That is
the right trade when the labels feed a test set, where consistency matters
more than volume, and the wrong one when the goal is maximum coverage for
training. Partition by giving each annotator a different --task file if the
goal ever changes.

WHY A LOCAL SERVER AND NOT A PUBLISHED PAGE:
the labels must land in classifier/data/labels.jsonl, the same file label_cli
writes and score_labels reads. A local server writes there directly. It also
keeps the corpus text on this machine -- there is no reason to send 200 news
articles to a third party to read them in a nicer font.

Binds 127.0.0.1 only, so nothing outside this machine can reach it.

Identical output to label_cli: append-only JSONL, flushed and fsynced per
label, resumable, 's' for unlabelable. The two are interchangeable -- start in
one, finish in the other.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))

# Imported, not duplicated: a second copy of the class list silently drifts
# the moment the taxonomy changes.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.taxonomy import CLASSES  # noqa: E402
from classifier import humset  # noqa: E402


class Store:
    """Owns the label files. Lock because browsers fire overlapping POSTs."""

    def __init__(self, task_path: str, out_path: str, multi: bool = False,
                 outdir: str | None = None):
        self.out_path = out_path
        self.outdir = outdir          # hosted mode: one file per annotator
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        # Multi-label writes `labels` (a list); single-label writes `label` (a
        # string). Different keys on purpose, so a scorer can tell which kind
        # of file it has rather than guessing from the value type.
        self.multi = multi
        self.classes = humset.LABELS if multi else CLASSES
        self.lock = threading.Lock()
        with open(task_path, encoding="utf-8") as fh:
            self.tasks = [json.loads(l) for l in fh if l.strip()]
        self.order = {t["url"]: i for i, t in enumerate(self.tasks)}
        # done is keyed by annotator: {annotator: {url: row}}. Local mode uses
        # the single annotator "" so the old single-file behaviour is intact.
        self.done: dict[str, dict[str, dict]] = {}
        if outdir:
            # Every labels-<who>.jsonl already on disk is an annotator we know.
            for f in sorted(os.listdir(outdir)):
                if f.startswith("labels-") and f.endswith(".jsonl"):
                    who = f[len("labels-"):-len(".jsonl")]
                    self.done[who] = self._read(os.path.join(outdir, f))
        else:
            self.done[""] = self._read(out_path)

    def _path(self, who: str) -> str:
        if not self.outdir:
            return self.out_path
        safe = "".join(c for c in who if c.isalnum() or c in "-_")[:40] or "anon"
        return os.path.join(self.outdir, f"labels-{safe}.jsonl")

    @staticmethod
    def _read(path: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        out[r["url"]] = r
        return out

    def _rewrite(self, who: str = "") -> None:
        path = self._path(who)
        tmp = path + ".tmp"
        rows = sorted(self.done.get(who, {}).values(),
                      key=lambda r: self.order.get(r["url"], 0))
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    def state(self, who: str = "") -> dict:
        with self.lock:
            done = self.done.get(who, {})
            # No partitioning: everyone is served the whole task file in order.
            nxt = next((t for t in self.tasks if t["url"] not in done), None)
            counts: dict[str, int] = {}
            for r in done.values():
                if self.multi:
                    got = r.get("labels") or []
                    if not got:
                        k = "_" + (r.get("reason") or "skip")
                        counts[k] = counts.get(k, 0) + 1
                    for l in got:
                        counts[l] = counts.get(l, 0) + 1
                else:
                    k = r["label"] or ("_" + (r.get("reason") or "skip"))
                    counts[k] = counts.get(k, 0) + 1
            return {
                "total": len(self.tasks),
                "done": len(done),
                "hosted": bool(self.outdir),
                "who": who,
                "peers": {k: len(v) for k, v in self.done.items() if k != who},
                "skipped": sum(1 for r in done.values()
                               if not self._any(r)
                               and r.get("reason") != "ambiguous"),
                "ambiguous": sum(1 for r in done.values()
                                 if r.get("reason") == "ambiguous"),
                "multi": self.multi,
                "display": (humset.DISPLAY if self.multi
                            else {c: c for c in CLASSES}),
                "definition": (humset.DEFINITION if self.multi else {}),
                "exclusive": humset.EXCLUSIVE if self.multi else None,
                "counts": counts,
                "classes": self.classes,
                "article": nxt,
                "can_undo": bool(done),
            }

    @staticmethod
    def _any(r: dict) -> bool:
        return bool(r.get("labels") if "labels" in r else r.get("label"))

    def label(self, url: str, label, reason: str | None = None,
              who: str = "") -> None:
        """reason distinguishes the two non-labels: "ambiguous" means the
        taxonomy does not decide it (a finding about the taxonomy), "skip"
        means it is not an article (a finding about the corpus). Collapsing
        them would hide which one we have."""
        with self.lock:
            t = next((x for x in self.tasks if x["url"] == url), {})
            row = {"url": url, "i": t.get("i"),
                   "site": t.get("site"),
                   "annotator": who or None,
                   "reason": reason,
                   "published_date": t.get("published_date"),
                   "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if self.multi:
                row["labels"] = humset.validate(label or [])
            else:
                row["label"] = label
            done = self.done.setdefault(who, {})
            fresh = url not in done
            done[url] = row
            path = self._path(who)
            if fresh:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                self._rewrite(who)

    def undo(self, who: str = "") -> None:
        """Only ever the caller's own last label -- never another annotator's."""
        with self.lock:
            done = self.done.get(who, {})
            if not done:
                return
            last = max(done.values(), key=lambda r: self.order.get(r["url"], -1))
            del done[last["url"]]
            self._rewrite(who)


def make_handler(store: Store, page: str, token: str | None,
                 cookie_path: str = "/"):
    class H(http.server.BaseHTTPRequestHandler):
        # --- identity + token ---
        def _cookies(self) -> dict:
            raw = self.headers.get("Cookie") or ""
            out = {}
            for part in raw.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip()
            return out

        def _qs(self) -> dict:
            from urllib.parse import urlparse, parse_qs
            return {k: v[0] for k, v in
                    parse_qs(urlparse(self.path).query).items()}

        def _authed(self) -> bool:
            if not token:
                return True
            return (self._qs().get("t") == token
                    or self._cookies().get("lt") == token)

        @staticmethod
        def _clean(w: str) -> str:
            return "".join(c for c in (w or "") if c.isalnum() or c in "-_")[:40]

        def _who(self) -> str:
            """The annotator's own name, from a cookie they set once."""
            if not store.outdir:
                return ""
            return self._clean(self._cookies().get("lw")
                               or self._qs().get("who") or "")
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            if not self._authed():
                self._send(403, b"add ?t=<token> to the URL", "text/plain")
                return
            if path in ("/", "/index.html"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if token:
                    # Carry the token so the annotator need not keep it in the URL.
                    self.send_header("Set-Cookie",
                                     f"lt={token}; Path={cookie_path}; "
                                     f"Max-Age=2592000; SameSite=Lax")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/state":
                self._json(store.state(self._who()))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._authed():
                self._send(403, b"forbidden", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b"bad json", "text/plain")
                return
            if self.path == "/label":
                lbl = payload.get("labels") if store.multi else payload.get("label")
                try:
                    if store.multi:
                        humset.validate(lbl or [])
                    elif lbl is not None and lbl not in store.classes:
                        raise ValueError(f"unknown label {lbl!r}")
                except ValueError as e:
                    self._send(400, str(e).encode("utf-8"), "text/plain")
                    return
                who = self._who()
                if store.outdir and not who:
                    self._send(409, b"set your name first", "text/plain")
                    return
                store.label(payload["url"], lbl, payload.get("reason"), who)
                self._json(store.state(who))
            elif self.path == "/name":
                who = self._clean(payload.get("name"))
                if not who:
                    self._send(400, b"letters, digits, - and _ only",
                               "text/plain")
                    return
                body = json.dumps(store.state(who), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Set-Cookie",
                                 f"lw={who}; Path={cookie_path}; "
                                 f"Max-Age=7776000; SameSite=Lax")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/undo":
                store.undo(self._who())
                self._json(store.state(self._who()))
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=os.path.join(HERE, "data", "label_task.jsonl"))
    p.add_argument("--out", default=os.path.join(HERE, "data", "labels.jsonl"))
    p.add_argument("--page", default=os.path.join(HERE, "label_web.html"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--multi", action="store_true",
                   help="HumSet multi-label mode (classifier.humset)")
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept connections from other machines")
    p.add_argument("--token", default=None,
                   help="shared secret required on every request (hosted mode). "
                        "Prefer the LABEL_TOKEN env var: a token passed here is "
                        "visible in `ps` to every user on the machine.")
    p.add_argument("--outdir", default=None,
                   help="hosted mode: one labels-<id>.jsonl per annotator here")
    p.add_argument("--cookie-path", default="/",
                   help="set to the proxy mount point (e.g. /label/) so the "
                        "token cookie is not sent to other apps on the host")
    a = p.parse_args()
    # env wins only when --token was not given, so a systemd unit can
    # keep the secret in an EnvironmentFile instead of the command line.
    a.token = a.token or os.environ.get("LABEL_TOKEN") or None

    for path, hint in ((a.task, "python -m classifier.sample_for_labelling"),
                       (a.page, "label_web.html is missing from the repo")):
        if not os.path.exists(path):
            print(f"missing {path}\n  -> {hint}", file=sys.stderr)
            return 1

    store = Store(a.task, a.out, multi=a.multi, outdir=a.outdir)
    page = open(a.page, encoding="utf-8").read()

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    if a.host != "127.0.0.1" and not a.token:
        print("refusing to listen on a public interface without --token.\n"
              "  generate one:  python3 -c \"import secrets;print(secrets.token_urlsafe(16))\"",
              file=sys.stderr)
        return 2

    shown = "127.0.0.1" if a.host in ("127.0.0.1", "localhost") else a.host
    url = f"http://{shown}:{a.port}" + (f"/?t={a.token}" if a.token else "")
    with S((a.host, a.port), make_handler(store, page, a.token, a.cookie_path)) as httpd:
        st = store.state()
        mode = "MULTI-label (HumSet)" if a.multi else "single-label"
        print(f"  {mode}: {len(store.classes)} labels")
        if a.outdir:
            print(f"  hosted: {len(store.tasks)} articles, served WHOLE to "
                  f"every annotator (no partitioning)")
            if store.done:
                for who, d in sorted(store.done.items()):
                    print(f"    {who:20} {len(d)} labelled")
            else:
                print(f"    no annotators yet")
            print(f"  writing to {a.outdir}/labels-<name>.jsonl")
        else:
            print(f"  {st['total']} sampled, {st['done']} already labelled, "
                  f"{st['total'] - st['done']} to go")
            print(f"  writing to {a.out}")
        print(f"\n  open {url}")
        print(f"  Ctrl-C when done -- progress is saved after every label\n")
        if not a.no_open:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            if a.outdir:
                print("\n  stopped.")
                for who, d in sorted(store.done.items()):
                    print(f"    {who:20} {len(d)} labelled")
            else:
                st = store.state()
                print(f"\n  stopped: {st['done']}/{st['total']} labelled "
                      f"({st['skipped']} skipped, {st['ambiguous']} ambiguous)")
            print("  resume with the same command; nothing is lost")
    return 0


if __name__ == "__main__":
    sys.exit(main())

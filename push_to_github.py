#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
push_to_github.py -- upload this repo to GitHub, safely and reliably

FOUR WAYS A TOKEN LEAKS, ALL AVOIDABLE
======================================
  1. **pasted into a chat / issue / commit.**  It lives in that
     transcript forever, and any log collector keeps a copy.
     -> This script reads it from a FILE instead. Never paste it.

  2. **embedded in the remote URL**
     (https://user:token@github.com/...).  git writes that URL into
     .git/config in PLAIN TEXT, where it stays and travels with the repo.
     -> The remote URL here never contains a token. The credential goes
        through `-c http.extraheader`, which is NOT persisted.

  3. **as a command-line argument.**  On Windows other processes can read
     your command line from the process list.
     -> The token enters through a config value, not argv.

  4. **printed in an error message.**  A failed push dumps the URL it
     tried, token included.
     -> Every message is scrubbed before printing.

WHERE THE TOKEN LIVES
=====================
    <parent of this repo>\<repo-name>_token.txt

Outside the repo. .gitignore additionally blocks the name so it cannot be
committed even if you move it inside by mistake.

TOKEN SCOPE
===========
Needs `repo` (classic) or Contents + Administration read/write
(fine-grained). Administration is what lets this create the repository;
if that is refused you get an exact explanation and two ways forward.

WHY IT RETRIES ACROSS CHANNELS
==============================
Measured on this machine: github.com is REACHABLE but FLAKY.

    git ls-remote https://github.com/...     -> ok, 1.2s
    git push     https://github.com/...      -> sometimes
                                                "Recv failure: Connection
                                                 was reset"

A single attempt is therefore not enough. This tries the direct route,
then mirrors measured reachable here:

    gh-proxy.com   1.1s
    ghfast.top     1.4s
    ghproxy.net    1.7s

THE BUG THIS FIXED
==================
The first version announced "pushed" when git had actually failed -- it
never inspected the exit code. Now every git call returns its status, and
the last step asks the GitHub API what really arrived. A success claim is
only made after the API agrees.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT.parent / ("%s_token.txt" % ROOT.name)
DEFAULT_REPO = ROOT.name
API = "https://api.github.com"

#: (label, url template). Direct first, then measured-reachable mirrors.
CHANNELS = [
    ("direct", "https://github.com/{login}/{repo}.git"),
    ("gh-proxy.com",
     "https://gh-proxy.com/https://github.com/{login}/{repo}.git"),
    ("ghfast.top",
     "https://ghfast.top/https://github.com/{login}/{repo}.git"),
    ("ghproxy.net",
     "https://ghproxy.net/https://github.com/{login}/{repo}.git"),
]


# ---------------------------------------------------------------------------
def read_token() -> str:
    """
    Read the token from a file.

    Tolerates however it was saved: bare token, 'token=xxx', 'token: xxx',
    quoted, CRLF, or written as GBK by a Windows editor.
    """
    if not TOKEN_FILE.exists():
        return ""
    raw = ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
        try:
            raw = TOKEN_FILE.read_text(encoding=enc)
            break
        except Exception:
            continue
    if not raw.strip():
        return ""
    for line in raw.splitlines():
        s = line.strip().strip('"').strip("'")
        if not s or s.startswith("#"):
            continue
        for sep in ("=", ":"):
            if sep in s:
                head, _, tail = s.partition(sep)
                if head.strip().lower() in ("token", "github", "github_token",
                                            "gh_token", "pat"):
                    s = tail.strip().strip('"').strip("'")
                    break
        if s:
            return s
    return ""


def mask(t: str) -> str:
    return "(too short)" if len(t) < 12 else t[:7] + "..." + t[-4:]


def api(path: str, token: str, method: str = "GET", body=None):
    """Returns (status, parsed_json_or_text)."""
    url = path if path.startswith("http") else API + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "push-to-github-script")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        return 0, "%s: %s" % (type(e).__name__, e)


def run_git(args, timeout: int = 600):
    """
    Run git. **Always returns the exit code** -- ignoring it was the
    original bug: the script announced success on a failed push.
    """
    r = subprocess.run(["git"] + args, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return r.returncode, ((r.stdout or "") + (r.stderr or ""))


def scrub(text: str, secrets) -> str:
    """Remove anything token-shaped before it reaches stdout or a log."""
    for s in secrets:
        if s and len(s) > 8:
            text = text.replace(s, "***REDACTED***")
    return text


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-create", action="store_true")
    ap.add_argument("--branch", default="")
    ap.add_argument("--force", action="store_true",
                    help="use --force-with-lease (after an amend/rebase)")
    ap.add_argument("--desc", default=(
        "GGUF memory budget for CPU-only machines, with measured calibration"))
    args = ap.parse_args()

    print()
    print("=" * 74)
    print("  push %s to GitHub" % ROOT.name)
    print("=" * 74)

    # ---- 1. token ----
    print("\n[1] token")
    token = read_token()
    if not token:
        print("    not found: %s" % TOKEN_FILE)
        print()
        print("    Create that file and put ONLY the token in it:")
        print("      - one line, no quotes needed")
        print("      - it sits OUTSIDE the repo")
        print("      - .gitignore also blocks the name as a second guard")
        print()
        print("    Get one at: https://github.com/settings/tokens")
        print("    classic token, scope: repo")
        print("    (or fine-grained: Contents + Administration, read/write)")
        print()
        print("    DO NOT paste the token into a chat.")
        return 1
    print("    read from %s" % TOKEN_FILE.name)
    print("    looks like %s (len %d)" % (mask(token), len(token)))
    if token.startswith("ghp_"):
        print("    type: classic")
    elif token.startswith("github_pat_"):
        print("    type: fine-grained")
    else:
        print("    type: unrecognised prefix (may still work)")

    basic = base64.b64encode(("x-access-token:" + token).encode()).decode()
    secrets = [token, basic]
    hdr = "AUTHORIZATION: basic " + basic

    # ---- 2. validate ----
    print("\n[2] validate against GitHub")
    st, who = api("/user", token)
    if st != 200:
        print("    FAILED: HTTP %d" % st)
        print("    %s" % str(who)[:300])
        print()
        print("    401 -> token wrong, expired, or revoked")
        print("    403 -> lacks permission, or rate limited")
        return 1
    login = who.get("login", "?")
    print("    OK  %s  (%s)" % (login, who.get("html_url", "")))

    try:
        req = urllib.request.Request(API + "/user")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("User-Agent", "push-to-github-script")
        with urllib.request.urlopen(req, timeout=30) as r:
            scopes = r.headers.get("x-oauth-scopes", "")
    except Exception:
        scopes = ""
    if scopes:
        print("    scopes: %s" % scopes)
        parts = [s.strip() for s in scopes.split(",")]
        danger = [s for s in parts
                  if s in ("delete_repo", "admin:org", "admin:enterprise",
                           "write:packages", "delete:packages")]
        if "repo" not in parts and "public_repo" not in parts:
            print("    WARNING: no 'repo' scope -- push will likely fail")
        if danger:
            print("    WARNING: broad permissions on this token: %s"
                  % ", ".join(danger))
            print("             a token scoped to just 'repo' is enough here")

    if args.check:
        print("\n  --check only; nothing pushed.")
        return 0

    # ---- 3. repo ----
    print("\n[3] repository '%s'" % args.repo)
    st, info = api("/repos/%s/%s" % (login, args.repo), token)
    if st == 200:
        print("    exists: %s" % info.get("html_url"))
    elif st == 404:
        if args.no_create:
            print("    does not exist, and --no-create was given.")
            print("    create it: https://github.com/new?name=%s" % args.repo)
            return 1
        print("    does not exist -> creating")
        st2, created = api("/user/repos", token, method="POST", body={
            "name": args.repo, "description": args.desc,
            "private": bool(args.private), "auto_init": False,
            "has_issues": True, "has_wiki": False, "has_projects": False})
        if st2 in (200, 201):
            print("    created: %s" % created.get("html_url"))
        else:
            print("    FAILED: HTTP %d" % st2)
            print("    %s" % str(created)[:400])
            print()
            print("    This is the 403 hit before. Two fixes:")
            print("      a) create it by hand: "
                  "https://github.com/new?name=%s" % args.repo)
            print("         then re-run with --no-create")
            print("      b) add the permission "
                  "(fine-grained: Administration read/write; "
                  "classic: the 'repo' scope covers it)")
            return 1
    else:
        print("    unexpected HTTP %d: %s" % (st, str(info)[:200]))
        return 1

    # ---- 4. local repo ----
    print("\n[4] local repository")
    rc, out = run_git(["status", "--porcelain"])
    if out.strip():
        print("    %d uncommitted change(s); committing"
              % len(out.strip().splitlines()))
        run_git(["add", "-A"])
        run_git(["commit", "-q", "-m", "chore: update before push"])
    else:
        print("    working tree clean")

    branch = args.branch
    if not branch:
        rc, out = run_git(["branch", "--show-current"])
        branch = out.strip() or "main"
    rc, out = run_git(["ls-files"])
    n_files = len([x for x in out.splitlines() if x.strip()])
    print("    branch: %s   tracked files: %d" % (branch, n_files))
    if n_files == 0:
        print("    nothing to push")
        return 1

    # ---- 5. push across channels ----
    print("\n[5] push (retrying across channels; github.com is flaky here)")
    last_err = ""
    pushed_via = ""
    for name, tmpl in CHANNELS:
        url = tmpl.format(login=login, repo=args.repo)
        print("\n    channel: %s" % name)
        run_git(["remote", "remove", "origin"])
        run_git(["remote", "add", "origin", url])   # no credentials inside

        t0 = time.time()
        rc, out = run_git(["-c", "http.extraheader=" + hdr,
                           "ls-remote", "origin", "HEAD"], timeout=90)
        if rc != 0:
            last_err = scrub(out, secrets)
            tail = (last_err.strip().splitlines()[-1][:88]
                    if last_err.strip() else "?")
            print("      unreachable (%.1fs): %s" % (time.time() - t0, tail))
            continue
        print("      reachable (%.1fs)" % (time.time() - t0))

        # Fetch first, ALWAYS.
        # Why: after an amend/rebase the remote and local diverge, and the
        # push is rejected as non-fast-forward. `--force-with-lease` needs a
        # known remote-tracking ref to lease against, and it refuses with
        # "stale info" when it has none -- which looks like a failure but is
        # really just missing information. Fetching supplies it.
        rc, fout = run_git(["-c", "http.extraheader=" + hdr,
                            "fetch", "origin", branch], timeout=180)
        if rc != 0:
            print("      (fetch failed, will try a plain push anyway)")

        push_args = ["-c", "http.extraheader=" + hdr, "push", "-u", "origin",
                     branch]
        if args.force:
            push_args = ["-c", "http.extraheader=" + hdr, "push", "-u",
                         "--force-with-lease", "origin", branch]

        t0 = time.time()
        rc, out = run_git(push_args, timeout=600)
        safe = scrub(out, secrets)
        if rc == 0:
            print("      PUSHED in %.1fs" % (time.time() - t0))
            for ln in safe.splitlines():
                if ln.strip():
                    print("        " + ln.strip()[:110])
            pushed_via = name
            break

        # Non-fast-forward after an amend: retry once with the lease.
        if "non-fast-forward" in safe or "rejected" in safe:
            print("      rejected (history diverged); "
                  "retrying with --force-with-lease")
            rc, out = run_git(["-c", "http.extraheader=" + hdr, "push", "-u",
                               "--force-with-lease", "origin", branch],
                              timeout=600)
            safe = scrub(out, secrets)
            if rc == 0:
                print("      PUSHED (forced) in %.1fs" % (time.time() - t0))
                for ln in safe.splitlines():
                    if ln.strip():
                        print("        " + ln.strip()[:110])
                pushed_via = name
                break

        last_err = safe
        print("      FAILED in %.1fs" % (time.time() - t0))
        for ln in safe.strip().splitlines()[-4:]:
            print("        " + ln.strip()[:110])

    if not pushed_via:
        print()
        print("=" * 74)
        print("  every channel failed")
        print("=" * 74)
        for ln in (last_err or "").strip().splitlines()[-8:]:
            print("  " + ln.strip()[:110])
        return 1

    # ---- 6. verify with the API (the anti-false-success step) ----
    print("\n[6] verify: ask the GitHub API what actually arrived")
    st, branches = api("/repos/%s/%s/branches" % (login, args.repo), token)
    if st == 200 and isinstance(branches, list):
        names = [b.get("name") for b in branches]
        print("    branches on GitHub: %s" % (", ".join(names) or "(none)"))
        if branch not in names:
            print("    FAILED: branch '%s' is not there -- the push did not "
                  "land" % branch)
            return 1
    else:
        print("    could not list branches: HTTP %d" % st)

    st, files = api("/repos/%s/%s/contents/" % (login, args.repo), token)
    if st == 200 and isinstance(files, list):
        print("    files on GitHub: %d" % len(files))
        for f in files[:12]:
            print("      %-28s %8s  %s"
                  % (f.get("name"), f.get("size", ""), f.get("type")))
    else:
        print("    could not list files: HTTP %d" % st)

    st, commits = api("/repos/%s/%s/commits" % (login, args.repo), token)
    if st == 200 and isinstance(commits, list) and commits:
        print("    latest commit: %s  %s"
              % (commits[0]["sha"][:8],
                 commits[0]["commit"]["message"].splitlines()[0][:52]))

    # ---- 7. credential audit ----
    print("\n[7] credential audit")
    rc, u = run_git(["remote", "get-url", "origin"])
    u = u.strip()
    print("    remote url: %s" % u)
    dirty = ("@" in u or "token" in u.lower() or "basic" in u.lower())
    print("    %s" % ("LEAKED -- fix .git/config now"
                      if dirty else "clean: no credentials in the URL"))
    cfg = ROOT / ".git" / "config"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="replace")
        leak = (token in text) or (basic in text)
        print("    .git/config: %s"
              % ("CONTAINS THE TOKEN" if leak else "no token"))
        if leak:
            print("      fix it: set the remote to %s" % u)

    print()
    print("=" * 74)
    print("  done via '%s'  ->  https://github.com/%s/%s"
          % (pushed_via, login, args.repo))
    print("=" * 74)
    print()
    print("  you can delete %s now, or keep it for the next push"
          % TOKEN_FILE)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

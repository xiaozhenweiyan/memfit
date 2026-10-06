#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
push_to_github.py -- upload this repo to GitHub, safely

WHY THIS SCRIPT EXISTS (read this before using a token)
=======================================================
Three ways a GitHub token leaks, all avoidable:

  1. **pasted into a chat / issue / commit message.**
     Then it lives in that transcript forever, and any log collector
     keeps a copy. Never paste it anywhere -- write it to a file.

  2. **embedded in the remote URL** (https://user:token@github.com/...).
     git writes that URL into .git/config in PLAIN TEXT, where it sits
     forever and gets copied with the repo. This script never does that.

  3. **in a command line argument.** On Windows, arguments are visible
     to other processes via the process list. This script passes the
     token through the environment instead, and uses a credential-free
     remote URL.

WHERE THE TOKEN GOES
====================
    E:\memfit_token.txt        <- you create this, once

It is OUTSIDE the repo, and .gitignore additionally blocks the name,
so it cannot be committed even by accident.

TOKEN SCOPE
===========
Needs `repo` (classic) or Contents+Administration read/write
(fine-grained). The Administration permission is what lets the script
create the repository for you -- that exact permission is what caused
the 403 you hit last time on githeat, so if creation fails, either add
the permission or create the empty repo on the web and re-run this.

USAGE
=====
    python push_to_github.py                  # check everything, then push
    python push_to_github.py --check           # only validate the token
    python push_to_github.py --repo NAME       # different repo name
    python push_to_github.py --private         # create it private
    python push_to_github.py --no-create       # repo already exists on web
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT.parent / "memfit_token.txt"
DEFAULT_REPO = "memfit"

API = "https://api.github.com"


# ---------------------------------------------------------------------------
def read_token() -> str:
    """
    Read the token from a file.

    Accepts the file however you happened to create it:
    a bare token, 'token=xxx', 'token: xxx', or with quotes.
    Also handles the file being written as GBK by a Windows editor.
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
    # take the first non-comment line that looks like a token
    for line in raw.splitlines():
        s = line.strip().strip('"').strip("'")
        if not s or s.startswith("#"):
            continue
        # strip a "key=value" or "key: value" prefix if present
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


def api(path: str, token: str, method: str = "GET", body=None):
    """Call the GitHub API. Returns (status, parsed_or_text)."""
    url = path if path.startswith("http") else API + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "memfit-push-script")
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
        return 0, f"{type(e).__name__}: {e}"


def mask(t: str) -> str:
    if len(t) < 12:
        return "(too short)"
    return t[:7] + "..." + t[-4:]


def run_git(args, env=None, check=True):
    r = subprocess.run(["git"] + args, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace", env=env)
    if check and r.returncode != 0:
        return False, (r.stdout or "") + (r.stderr or "")
    return True, (r.stdout or "") + (r.stderr or "")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-create", action="store_true")
    ap.add_argument("--desc", default=(
        "GGUF memory budget for CPU-only machines, with measured calibration"))
    args = ap.parse_args()

    print()
    print("=" * 74)
    print("  push memfit to GitHub")
    print("=" * 74)

    # ---- 1. token ----
    print("\n[1] token")
    token = read_token()
    if not token:
        print(f"    not found: {TOKEN_FILE}")
        print()
        print("    create that file and paste ONLY the token into it:")
        print("      - one line, no quotes needed")
        print("      - it is outside the repo, and .gitignore also blocks the name")
        print()
        print("    get a token at: https://github.com/settings/tokens")
        print("    classic token, scope: repo")
        print("    (or fine-grained: Contents + Administration, both read/write)")
        print()
        print("    DO NOT paste the token into this chat.")
        return 1

    print(f"    read from {TOKEN_FILE.name}")
    print(f"    looks like {mask(token)}  (len={len(token)})")

    if token.startswith("github_pat_"):
        kind = "fine-grained"
    elif token.startswith("ghp_"):
        kind = "classic"
    else:
        kind = "unrecognised prefix"
    print(f"    type: {kind}")

    # ---- 2. validate ----
    print("\n[2] check the token against GitHub")
    st, who = api("/user", token)
    if st != 200:
        print(f"    FAILED: HTTP {st}")
        print(f"    {str(who)[:300]}")
        print()
        print("    if 401: the token is wrong, expired, or revoked")
        print("    if 403: it lacks permission, or you hit a rate limit")
        return 1
    login = who.get("login", "?")
    print(f"    OK  logged in as {login}")
    print(f"        {who.get('html_url','')}")

    st2, scopes = api("/", token)
    hdr_scopes = ""
    try:
        req = urllib.request.Request(API + "/user")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("User-Agent", "memfit-push-script")
        with urllib.request.urlopen(req, timeout=30) as r:
            hdr_scopes = r.headers.get("x-oauth-scopes", "")
    except Exception:
        pass
    if hdr_scopes:
        print(f"        scopes: {hdr_scopes}")
        if "repo" not in hdr_scopes and "public_repo" not in hdr_scopes:
            print("        WARNING: no 'repo' scope -- push will likely fail")

    if args.check:
        print("\n  --check only; nothing pushed.")
        return 0

    # ---- 3. repo exists? ----
    print(f"\n[3] repository '{args.repo}'")
    st, info = api(f"/repos/{login}/{args.repo}", token)
    if st == 200:
        print(f"    already exists: {info.get('html_url')}")
        exists = True
    elif st == 404:
        exists = False
        if args.no_create:
            print("    does not exist, and --no-create was given.")
            print(f"    create it at https://github.com/new?name={args.repo}")
            return 1
        print("    does not exist -> creating")
        st2, created = api("/user/repos", token, method="POST", body={
            "name": args.repo,
            "description": args.desc,
            "private": bool(args.private),
            "auto_init": False,
            "has_issues": True, "has_wiki": False, "has_projects": False,
        })
        if st2 in (200, 201):
            print(f"    created: {created.get('html_url')}")
            exists = True
        else:
            print(f"    FAILED to create: HTTP {st2}")
            print(f"    {str(created)[:400]}")
            print()
            print("    This is the 403 you hit before: the token cannot")
            print("    create repositories. Two fixes:")
            print(f"      a) create it manually: https://github.com/new?name={args.repo}")
            print("         then re-run this with --no-create")
            print("      b) add the permission (fine-grained: Administration")
            print("         read/write; classic: the 'repo' scope covers it)")
            return 1
    else:
        print(f"    unexpected HTTP {st}: {str(info)[:200]}")
        return 1

    # ---- 4. git remote (WITHOUT the token in the URL) ----
    print("\n[4] git remote")
    url = f"https://github.com/{login}/{args.repo}.git"
    ok, out = run_git(["remote", "get-url", "origin"], check=False)
    if ok and out.strip():
        run_git(["remote", "set-url", "origin", url])
        print(f"    updated origin -> {url}")
    else:
        run_git(["remote", "add", "origin", url])
        print(f"    added origin -> {url}")
    print("    (the URL contains NO token -- check .git/config yourself)")

    # ---- 5. commit if dirty ----
    print("\n[5] working tree")
    ok, out = run_git(["status", "--porcelain"])
    if out.strip():
        print(f"    {len(out.strip().splitlines())} uncommitted change(s):")
        for ln in out.strip().splitlines()[:10]:
            print("      " + ln)
        print("    committing them")
        run_git(["add", "-A"])
        run_git(["commit", "-q", "-m", "chore: update before push"])
        print("    committed")
    else:
        print("    clean")

    # ---- 6. push ----
    print("\n[6] push")
    branch = "main"
    ok, out = run_git(["branch", "--show-current"])
    if out.strip():
        branch = out.strip()
    print(f"    branch: {branch}")

    # Key point: the token travels in the environment, never in the URL
    # and never as a command-line argument (visible in the process list).
    env = dict(os.environ)
    env["GIT_ASKPASS"] = ""
    env["GIT_TERMINAL_PROMPT"] = "0"

    # Read it from the environment via a one-shot credential helper,
    helper = ("!f() { echo username=x-access-token; "
              "echo password=$GITHUB_TOKEN; }; f")
    ok, out = run_git(
        ["-c", "credential.helper=" + helper, "push", "-u", "origin",
         branch, "--force-with-lease"],
        env={**env, "GITHUB_TOKEN": token}, check=False)

    if ok:
        print("    pushed")
    else:
        # --force-with-lease fails on a brand new remote; retry plainly
        if "stale info" in out or "lease" in out.lower():
            print("    (remote had no matching ref; retrying plainly)")
            ok, out = run_git(
                ["-c", "credential.helper=" + helper, "push", "-u",
                 "origin", branch],
                env={**env, "GITHUB_TOKEN": token}, check=False)
        if ok:
            print("    pushed")
        else:
            # redact anything that looks like the token before printing
            safe = out.replace(token, "***REDACTED***")
            print("    FAILED")
            for ln in safe.splitlines()[-12:]:
                print("      " + ln)
            print()
            print("    common causes:")
            print("      - token lacks 'repo' / Contents write")
            print("      - token expired")
            print("      - the repo is owned by someone else")
            return 1

    print()
    print("=" * 74)
    print(f"  done: {url.replace('.git','')}")
    print("=" * 74)
    print()
    print("  check that no token leaked into git config:")
    print("    git -C " + str(ROOT) + " config --get remote.origin.url")
    print("  it must NOT contain '@' or 'token'.")
    print()
    print(f"  you can now delete {TOKEN_FILE} if you want")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

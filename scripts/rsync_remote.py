#!/usr/bin/env python3
"""Push local reroom source to the Run:ai kernel and run code there.

Local box is a thin client (no torch); all compute is the remote `vla` workload.
This ships edited files into /opt/NeMo/reroom/src via a base64 blob through the
jexec persistent kernel, and runs arbitrary code with a generous timeout.

    python3 scripts/rsync_remote.py push reroom/generative/model.py [reroom/...]
    python3 scripts/rsync_remote.py run  path/to/snippet.py [timeout_s]
    echo 'code' | python3 scripts/rsync_remote.py run - [timeout_s]
"""
import base64, os, sys, textwrap

sys.path.insert(0, os.path.expanduser("~/.local/share/runai-tools"))
import jexec  # noqa: E402

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# project5 lives in its OWN remote directory so pushing here can never clobber
# project4's tree (override with REROOM_REMOTE).
REPO_REMOTE = os.environ.get("REROOM_REMOTE", "/opt/NeMo/reroom/e2e")


def _run(code, timeout=1800):
    jexec.init_xsrf()
    jexec.start_kernel()
    try:
        return jexec.run_code(code, timeout=timeout)
    finally:
        try:
            jexec.stop_kernel()
        except Exception:
            pass


def push(rel_paths):
    blobs = {}
    for rel in rel_paths:
        with open(os.path.join(REPO_LOCAL, rel), "rb") as f:
            blobs[rel] = base64.b64encode(f.read()).decode()
    code = "import base64, os\n"
    code += f"REPO={REPO_REMOTE!r}\n"
    code += f"BLOBS={blobs!r}\n"
    code += textwrap.dedent("""
        for rel, b in BLOBS.items():
            dst = os.path.join(REPO, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(base64.b64decode(b))
            print("wrote", dst, os.path.getsize(dst), "bytes")
    """)
    print(_run(code, timeout=300))


def run(path, timeout=1800):
    code = sys.stdin.read() if path == "-" else open(path).read()
    # ensure remote imports resolve the repo
    prelude = f"import os,sys; os.chdir({REPO_REMOTE!r}); sys.path.insert(0,{REPO_REMOTE!r})\n"
    print(_run(prelude + code, timeout=timeout))


def pull(remote_paths, local_dir):
    """Fetch generated binary files off the remote via a base64 print."""
    import re
    os.makedirs(local_dir, exist_ok=True)
    code = "import base64\n"
    code += f"PATHS={list(remote_paths)!r}\n"
    code += textwrap.dedent("""
        for p in PATHS:
            try:
                with open(p,'rb') as f: b=base64.b64encode(f.read()).decode()
                print('@@FILE@@', p, len(b))
                print(b)
                print('@@END@@')
            except Exception as e:
                print('@@ERR@@', p, e)
    """)
    out = _run(code, timeout=600) or ""
    # parse @@FILE@@ name len \n b64 \n @@END@@
    blocks = re.findall(r"@@FILE@@ (\S+) \d+\n(.*?)\n@@END@@", out, re.S)
    for rp, b in blocks:
        name = os.path.basename(rp)
        with open(os.path.join(local_dir, name), "wb") as f:
            f.write(base64.b64decode(b.strip()))
        print("pulled", name, os.path.getsize(os.path.join(local_dir, name)), "bytes")
    for e in re.findall(r"@@ERR@@ .*", out):
        print(e)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "push":
        push(sys.argv[2:])
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1800)
    elif cmd == "pull":
        # pull <local_dir> <remote_path...>
        pull(sys.argv[3:], sys.argv[2])
    else:
        sys.exit("usage: rsync_remote.py push <rel...> | run <file|-> [timeout] | pull <local_dir> <remote...>")

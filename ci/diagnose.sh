#!/usr/bin/env bash
# PROBE — temporary: report the runner facts behind the 2026-10-13 readiness failures.
set -u

echo "== host =="; uname -a || true
echo "== /etc/hosts =="; cat /etc/hosts || true
echo "== python3 =="; which -a python3 || true; python3 -VV || true
echo "== curl =="; which -a curl || true; curl --version | head -1 || true
echo "== proxy vars =="; env | grep -i proxy || echo "none"

echo "== getfqdn =="
python3 - <<'PY'
import socket, time
t = time.time()
name = socket.getfqdn("127.0.0.1")
print("getfqdn(127.0.0.1):", repr(name), round(time.time() - t, 2), "s", flush=True)
PY

echo "== stand-in server latency (two attempts) =="
python3 - <<'PY'
import os, socket, subprocess, tempfile, time, urllib.request

for attempt in (1, 2):
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "ready"), "w") as f:
        f.write("ok")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t0 = time.time()
    p = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", tmp],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tcp = None
    while time.time() - t0 < 60:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            tcp = time.time() - t0
            break
        except OSError:
            time.sleep(0.05)
    code = None
    while time.time() - t0 < 60:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as r:
                code = r.status
            break
        except Exception:
            time.sleep(0.05)
    print(f"attempt {attempt}: tcp_connect_after={tcp if tcp is None else round(tcp, 2)}s "
          f"http_200_after={round(time.time() - t0, 2)}s code={code} still_alive={p.poll() is None}", flush=True)
    if p.poll() is None:
        p.kill()
    print("server output:", repr(p.communicate(timeout=5)[0].decode()[:400]), flush=True)
PY

echo "== measure.sh start flow =="
port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
tmp=$(mktemp -d); echo ok > "$tmp/ready"
t0=$(date +%s)
( echo $$ > "$tmp/server.pid"; exec python3 -m http.server "$port" --bind 127.0.0.1 --directory "$tmp" ) > "$tmp/server.log" 2>&1 &
pid=$!
code=000
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$port/ready" || true)
  if [ "$code" = 200 ]; then break; fi
  sleep 0.5
done
echo "measure-flow ready: probes=$i code=$code elapsed=$(( $(date +%s) - t0 ))s"
echo "server log:"; cat "$tmp/server.log" || true
echo "== curl verbose vs the same server =="
curl -sv "http://127.0.0.1:$port/ready" -o /dev/null 2>&1 | head -12 || true
kill "$pid" 2>/dev/null || true

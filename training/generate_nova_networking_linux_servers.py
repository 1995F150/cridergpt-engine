"""Generate deterministic, duplicate-safe Nova networking/Linux/server records."""
import argparse
import json
import random
from pathlib import Path

def rec(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

def make(kind, rng, n):
    host = rng.randint(2, 240)
    port = rng.randint(1024, 9000)
    svc = f"nova-service-{n}"

    if kind == "private":
        q = f"For host scenario {n}, is 192.168.{n % 250}.{host} a private IPv4 address?"
        a = "Yes. 192.168.0.0/16 is an RFC 1918 private IPv4 range."
    elif kind == "cidr":
        q = f"For subnet exercise {n}, what network contains 10.{n % 250}.{host}.17/24?"
        a = f"A /24 uses the first three octets as the network prefix, so the network is 10.{n % 250}.{host}.0/24."
    elif kind == "ssh":
        q = f"For SSH exercise {n}, give a command for user nova connecting to 192.168.{n % 250}.{host}."
        a = f"`ssh nova@192.168.{n % 250}.{host}`"
    elif kind == "port":
        q = f"Server scenario {n}: a service should listen on TCP port {port}. What Linux command can check listening TCP sockets?"
        a = f"Use `ss -ltnp` and look for port {port}. Depending on permissions, process details may require elevated privileges."
    elif kind == "systemctl":
        q = f"How do you restart the systemd service `{svc}`?"
        a = f"`sudo systemctl restart {svc}`\nThen check its status or logs."
    elif kind == "journal":
        q = f"How do you view logs for systemd service `{svc}`?"
        a = f"`journalctl -u {svc}` shows its journal entries. Use `-f` to follow new entries."
    elif kind == "enable":
        q = f"How do you enable `{svc}` at boot with systemd?"
        a = f"`sudo systemctl enable {svc}` enables it for boot. Add `--now` if it should also start immediately."
    elif kind == "dns":
        q = f"DNS scenario {n}: a host can reach 1.1.1.1 but cannot resolve `host{n}.example`. What should you investigate first?"
        a = "Investigate DNS/name resolution: configured DNS servers, resolver settings, DNS reachability, and the requested record."
    elif kind == "gateway":
        q = f"Routing scenario {n}: a Linux host can reach its LAN but no Internet IP addresses. Name key things to inspect."
        a = "Inspect its IP configuration, default route, gateway reachability, firewall rules, upstream router/NAT state, and the upstream Internet connection."
    elif kind == "dhcp":
        q = f"In DHCP scenario {n}, what information does DHCP normally provide to a LAN client?"
        a = "DHCP commonly provides an IP address plus configuration such as subnet prefix or mask, default gateway, DNS servers, and lease information."
    elif kind == "tcpudp":
        q = f"For networking exercise {n}, summarize the main difference between TCP and UDP."
        a = "TCP provides a connection-oriented reliable ordered byte stream. UDP sends independent datagrams without TCP's delivery, ordering, or connection guarantees."
    elif kind == "disk":
        q = f"For Linux server {n}, what command gives human-readable filesystem free/used space?"
        a = "`df -h` reports filesystem disk-space usage in human-readable units."
    elif kind == "memory":
        q = f"For Linux server {n}, what command gives human-readable RAM and swap usage?"
        a = "`free -h` reports memory and swap usage in human-readable units."
    elif kind == "process":
        q = f"For Linux host {n}, what command gives an interactive live view of processes and CPU/memory use?"
        a = "`top` provides an interactive, continuously updated process and resource view."
    elif kind == "linklocal":
        q = f"Interface scenario {n}: a Linux interface has address 169.254.{n % 250}.{host}. What does that indicate?"
        a = "169.254.0.0/16 is IPv4 link-local space. It can indicate the host did not obtain its expected configured or DHCP address; diagnose DHCP, link, and interface configuration rather than assuming one cause."
    else:
        raise ValueError(kind)

    return q, a

KINDS = (
    "private", "cidr", "ssh", "port", "systemctl", "journal", "enable",
    "dns", "gateway", "dhcp", "tcpudp", "disk", "memory", "process", "linklocal",
)

def generate(count, seed):
    rng = random.Random(seed)
    rows = []
    seen = set()
    for index in range(count):
        q, a = make(KINDS[index % len(KINDS)], rng, index + 1)
        key = " ".join(q.lower().split())
        if key in seen:
            raise RuntimeError(f"duplicate prompt generated at record {index + 1}: {q}")
        seen.add(key)
        rows.append(rec(q, a))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--seed", type=int, default=212)
    p.add_argument("--output", type=Path, default=Path("data/training/cridergpt21/generated_networking_linux_servers.jsonl"))
    args = p.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    rows = generate(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} Nova networking/Linux/server records across {len(KINDS)} template families to {args.output}")

if __name__ == "__main__":
    main()

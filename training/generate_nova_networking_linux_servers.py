"""Generate deterministic Nova networking/Linux/server instruction records."""
import argparse,json,random
from pathlib import Path
def rec(q,a):return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}
def make(k,r,n):
    host=r.randint(2,240);port=r.randint(1024,9000);svc=f"nova-service-{n}"
    if k=="private": return f"Is 192.168.{n%250}.{host} a private IPv4 address?","Yes. 192.168.0.0/16 is an RFC 1918 private IPv4 range."
    if k=="cidr": return f"What network contains 10.{n%250}.{host}.17/24?",f"A /24 uses the first three octets as the network prefix, so the network is 10.{n%250}.{host}.0/24."
    if k=="ssh": return f"Give an SSH command for user nova connecting to 192.168.{n%250}.{host}.",f"\`ssh nova@192.168.{n%250}.{host}\`"
    if k=="port": return f"A service should listen on TCP port {port}. What Linux command can help check listening TCP sockets?",f"Use \`ss -ltnp\` and look for port {port}. Depending on permissions, process details may require elevated privileges."
    if k=="systemctl": return f"How do you restart the systemd service \`{svc}\`?",f"\`sudo systemctl restart {svc}\`\nThen check its status or logs."
    if k=="journal": return f"How do you view logs for systemd service \`{svc}\`?",f"\`journalctl -u {svc}\` shows its journal entries. Use \`-f\` to follow new entries."
    if k=="enable": return f"How do you enable \`{svc}\` at boot with systemd?",f"\`sudo systemctl enable {svc}\` enables it for boot. Add \`--now\` if it should also start immediately."
    if k=="dns": return f"A host can reach 1.1.1.1 but cannot resolve \`host{n}.example\`. What should you investigate first?","Investigate DNS/name resolution: configured DNS servers, resolver settings, DNS reachability, and the requested record."
    if k=="gateway": return f"A Linux host can reach its LAN but no Internet IP addresses. Name key things to inspect.","Inspect its IP configuration, default route, gateway reachability, firewall rules, upstream router/NAT state, and the upstream Internet connection."
    if k=="dhcp": return f"What does DHCP normally provide to client {n} on a LAN?","DHCP commonly provides an IP address plus configuration such as subnet prefix or mask, default gateway, DNS servers, and lease information."
    if k=="tcpudp": return f"For networking example {n}, summarize the main difference between TCP and UDP.","TCP provides a connection-oriented reliable ordered byte stream. UDP sends independent datagrams without TCP's delivery, ordering, or connection guarantees."
    if k=="disk": return f"What Linux command gives human-readable filesystem free/used space for server {n}?","\`df -h\` reports filesystem disk-space usage in human-readable units."
    if k=="memory": return f"What Linux command gives human-readable RAM and swap usage for server {n}?","\`free -h\` reports memory and swap usage in human-readable units."
    if k=="process": return f"What command can give an interactive live view of processes and CPU/memory use on Linux host {n}?","\`top\` provides an interactive, continuously updated process and resource view."
    if k=="linklocal": return f"A Linux interface has address 169.254.{n%250}.{host}. What does that indicate?","169.254.0.0/16 is IPv4 link-local space. It can indicate the host did not obtain its expected configured or DHCP address; diagnose DHCP, link, and interface configuration rather than assuming one cause."
    raise ValueError(k)
KINDS=("private","cidr","ssh","port","systemctl","journal","enable","dns","gateway","dhcp","tcpudp","disk","memory","process","linklocal")
def generate(count,seed):
    r=random.Random(seed);return [rec(*make(KINDS[n%len(KINDS)],r,n+1)) for n in range(count)]
def main():
    p=argparse.ArgumentParser();p.add_argument("--count",type=int,default=10000);p.add_argument("--seed",type=int,default=212);p.add_argument("--output",type=Path,default=Path("data/training/cridergpt21/generated_networking_linux_servers.jsonl"));a=p.parse_args()
    if a.count<1:raise SystemExit("--count must be at least 1")
    rows=generate(a.count,a.seed);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8") as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print(f"Wrote {len(rows)} Nova networking/Linux/server records across {len(KINDS)} template families to {a.output}")
if __name__=="__main__":main()

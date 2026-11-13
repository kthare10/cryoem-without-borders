#!/usr/bin/env python3
"""CryoEM Without Borders — slice bring-up (the weave).

Provisions a multi-GPU processing VM ("cryoem") with the ``cryoem-without-borders``
CephFS project volume (the Globus collection) as its INPUT, installs MotionCor3 +
CTFFIND4 (+ RELION 5.0), and adds a downstream "receiver" VM at FIU (stand-in for
CAPE) joined by a dedicated site-to-site L2 link (``XFER``, private subnet) used
only for the result transfer — not the Ceph network. It sets up SSH trust
cryoem -> receiver, writes ``results/slice_info.json`` and EXITS. It waits for the slice once
(``submit(wait=True)``) and never polls after that. The experiment itself is
``cryoem_experiment.ipynb``.

GPU placement is discovered at submit time from per-host availability: the node
lands on a host that has ``NUM_GPUS`` free GPUs of one model (best model first:
A40 > A30 > RTX6000 > T4), unless ``SITE`` / ``GPU_MODEL`` pin it.

Actions:
    python3 provision.py start  SLICE_NAME          # provision + configure
    python3 provision.py status SLICE_NAME          # one-shot health check
    python3 provision.py extend SLICE_NAME [DAYS]   # renew the lease
    python3 provision.py stop   SLICE_NAME          # delete the slice

Configuration comes from environment variables set by weave.sh (see weave.json).
"""

from __future__ import annotations

import json
import os
import sys
import time

from fabric_env import (CEPH_CLUSTER, STORAGE_VOLUME, VOLUME_GLOB, data_dir, token_project,
                        WEAVE_DIR, get_fablib)

import ipaddress

IMAGE = "default_ubuntu_22"
NODE_NAME = "cryoem"            # must match fabric_env.NODE_NAME
RECEIVER_NAME = "receiver"
XFER_NET = "XFER"
RESULTS_DIR = os.path.join(WEAVE_DIR, "results")
TOOLS_DIR = os.path.join(WEAVE_DIR, "tools")

# list_hosts availability field -> add_component model, best first.
GPU_FIELDS = [
    ("a40_available", "GPU_A40"),
    ("a30_available", "GPU_A30"),
    ("rtx6000_available", "GPU_RTX6000"),
    ("tesla_t4_available", "GPU_TeslaT4"),
]

# XFER_NIC value -> list_hosts availability field.
NIC_FIELDS = {
    "NIC_Basic": "nic_basic_available",
    "NIC_ConnectX_5": "nic_connectx_5_available",
    "NIC_ConnectX_6": "nic_connectx_6_available",
    "NIC_ConnectX_7_100": "nic_connectx_7_100_available",
    "NIC_ConnectX_7_400": "nic_connectx_7_400_available",
}


def host_sites(res) -> dict:
    """host name -> site name."""
    out = {}
    for site in res.get_site_names():
        try:
            for h in res.get_hosts_by_site(site):
                out[h] = site
        except Exception:
            pass
    return out


def nics_free(row: dict, cfg: dict) -> bool:
    """The host has the XFER NIC (and, for the GPU node, a basic NIC for Ceph) free."""
    basic = int(row.get("nic_basic_available", 0) or 0)
    if not cfg["receiver_site"]:
        return basic >= 1
    if cfg["xfer_nic"] == "NIC_Basic":
        return basic >= 2
    return basic >= 1 and int(row.get(NIC_FIELDS[cfg["xfer_nic"]], 0) or 0) >= 1


def load_config() -> dict:
    env = os.environ.get
    return {
        "site": env("SITE", "").strip().upper() or None,
        "gpu_model": env("GPU_MODEL", "auto").strip(),
        "num_gpus": int(env("NUM_GPUS", "2")),
        "cores": int(env("CORES", "16")),
        "ram": int(env("RAM", "64")),
        "disk": int(env("DISK", "500")),
        "cuda_version": env("CUDA_VERSION", "12.4"),
        "install_relion": env("INSTALL_RELION", "true").lower() == "true",
        "lease_days": int(env("LEASE_DAYS", "14")),
        "keep_slice": env("KEEP_SLICE", "true").lower() == "true",
        "exclude_sites": {s.strip().upper() for s in env("EXCLUDE_SITES", "").split(",") if s.strip()},
        # Downstream site (FIU now; CAPE later). Blank = no receiver node.
        "receiver_site": env("RECEIVER_SITE", "FIU").strip().upper() or None,
        "receiver_cores": int(env("RECEIVER_CORES", "8")),
        "receiver_ram": int(env("RECEIVER_RAM", "32")),
        "receiver_disk": int(env("RECEIVER_DISK", "500")),
        "xfer_subnet": env("XFER_SUBNET", "192.168.100.0/24"),
        # NIC on both ends of the XFER link: shared NIC_Basic, or a dedicated ConnectX.
        "xfer_nic": env("XFER_NIC", "NIC_Basic").strip() or "NIC_Basic",
        # Pin vCPUs to the XFER NIC's NUMA node (CPU pinning only, no restart).
        "pin_cpu": env("PIN_CPU", "true").lower() == "true",
    }


# --- placement ----------------------------------------------------------------
def find_gpu_host(res, cfg: dict):
    """Return (site, host, model, available) for the best host, or None."""
    fields = [(f, m) for f, m in GPU_FIELDS
              if cfg["gpu_model"].lower() in ("", "auto", "any") or m == cfg["gpu_model"]]
    host_site = host_sites(res)
    try:
        rows = res.list_hosts(output="list", quiet=True) or []
    except Exception as exc:
        print(f"### PROGRESS: WARNING — list_hosts failed ({exc})")
        return None

    best = None
    for row in rows:
        host = row.get("name")
        site = host_site.get(host)
        if not site or site in cfg["exclude_sites"] or (cfg["site"] and site != cfg["site"]):
            continue
        if not cfg["site"] and site == cfg["receiver_site"]:   # transfer must cross sites
            continue
        if str(row.get("state", "")).lower() != "active":
            continue
        if (int(row.get("cores_available", 0) or 0) < cfg["cores"]
                or int(row.get("ram_available", 0) or 0) < cfg["ram"]
                or int(row.get("disk_available", 0) or 0) < cfg["disk"]
                or not nics_free(row, cfg)):
            continue
        for rank, (field, model) in enumerate(fields):
            avail = int(row.get(field, 0) or 0)
            if avail >= cfg["num_gpus"]:
                key = (rank, -avail, site)
                if best is None or key < best[0]:
                    best = (key, (site, host, model, avail))
                break
    return best[1] if best else None


def find_receiver_host(res, cfg: dict):
    """Host at RECEIVER_SITE with the receiver VM size and the XFER NIC free, or None."""
    host_site = host_sites(res)
    try:
        rows = res.list_hosts(output="list", quiet=True) or []
    except Exception as exc:
        print(f"### PROGRESS: WARNING — list_hosts failed ({exc})")
        return None
    for row in rows:
        if (host_site.get(row.get("name")) == cfg["receiver_site"]
                and str(row.get("state", "")).lower() == "active"
                and int(row.get("cores_available", 0) or 0) >= cfg["receiver_cores"]
                and int(row.get("ram_available", 0) or 0) >= cfg["receiver_ram"]
                and int(row.get("disk_available", 0) or 0) >= cfg["receiver_disk"]
                and int(row.get(NIC_FIELDS[cfg["xfer_nic"]], 0) or 0) >= 1):
            return row.get("name")
    return None


def project_tags(fablib) -> set:
    """Permission tags of the project in the token fablib uses (the user's default)."""
    tags = set(token_project(fablib).get("tags") or [])
    if not tags:
        print("### PROGRESS: WARNING — could not read project tags from the token; skipping the tag check")
    return tags


def project_name(fablib) -> str:
    p = token_project(fablib)
    return f"{p.get('name', '?')} ({p.get('uuid') or fablib.get_project_id()})"


def preflight(cfg: dict, fablib) -> None:
    """Fail fast on a bad XFER_NIC or when the project lacks the tags this slice needs
    (FABRIC default: 2c/10G/10G, no GPU, no SmartNIC)."""
    if cfg["xfer_nic"] not in NIC_FIELDS:
        print(f"### PROGRESS: ERROR — XFER_NIC={cfg['xfer_nic']} is not one of {', '.join(NIC_FIELDS)}.")
        sys.exit(2)
    tags = project_tags(fablib)
    if not tags:
        return
    unlimited = "VM.NoLimit" in tags
    if cfg["receiver_site"] and "Slice.Multisite" not in tags:
        print("### PROGRESS: ERROR — the FIU receiver needs the Slice.Multisite project tag.")
        sys.exit(2)
    need = []
    if max(cfg["cores"], cfg["receiver_cores"]) > 2 and not (unlimited or "VM.NoLimitCPU" in tags):
        need.append(f"VM.NoLimitCPU (CORES={cfg['cores']} > 2)")
    if max(cfg["ram"], cfg["receiver_ram"]) > 10 and not (unlimited or "VM.NoLimitRAM" in tags):
        need.append(f"VM.NoLimitRAM (RAM={cfg['ram']} > 10 GB)")
    if max(cfg["disk"], cfg["receiver_disk"]) > 10 and not (unlimited or "VM.NoLimitDisk" in tags):
        need.append(f"VM.NoLimitDisk (DISK={cfg['disk']} > 10 GB)")
    if cfg["num_gpus"] > 0 and "Component.GPU" not in tags:
        need.append("Component.GPU")
    if cfg["receiver_site"] and cfg["xfer_nic"] != "NIC_Basic":
        # FABRIC grants SmartNICs per model (Component.SmartNIC_ConnectX_5, ...), or all at once.
        model_tag = "Component.SmartNIC_" + cfg["xfer_nic"][len("NIC_"):]
        if "Component.SmartNIC" not in tags and not any(t.startswith(model_tag) for t in tags):
            need.append(f"{model_tag} (XFER_NIC={cfg['xfer_nic']})")
    if need:
        print(f"### PROGRESS: ERROR — project {project_name(fablib)} has tags {sorted(tags)}; "
              f"this slice also needs: {', '.join(need)}.")
        print("### PROGRESS: Ask FABRIC support / the project lead to add them, then download a "
              "fresh id_token.json (tags are baked into the token).")
        sys.exit(2)


def extend_lease(slice_obj, days: int) -> bool:
    """Renew the lease to ``days`` from now (FABRIC caps ~14). Never raises."""
    try:
        slice_obj.renew(days=int(days))
        print(f"### PROGRESS: Lease now ends {slice_obj.get_lease_end()}")
        return True
    except Exception as exc:
        print(f"### PROGRESS: WARNING — lease extension failed: {exc}")
        return False


# --- node setup ---------------------------------------------------------------
def run_setup(node, cfg: dict) -> int:
    """Run tools/setup-node.sh, streaming its output. Returns its exit code."""
    cmd = (f"timeout 7200 bash ~/tools/setup-node.sh {cfg['cuda_version']} "
           f"{'true' if cfg['install_relion'] else 'false'} 2>&1 | tee ~/setup-node.log; "
           f"echo SETUP_RC=${{PIPESTATUS[0]}}")
    out, _ = node.execute(cmd, quiet=False)
    for line in reversed(out.splitlines()):
        if line.startswith("SETUP_RC="):
            return int(line.split("=", 1)[1])
    return 1


def reboot_and_remount(slice_obj, node) -> None:
    """Reboot once for the NVIDIA driver, then re-mount the CephFS volume."""
    print("### PROGRESS: Rebooting node to load the NVIDIA driver...")
    try:
        node.execute("sudo systemctl reboot", quiet=True, timeout=20)
    except Exception:
        pass                                  # the SSH session drops; expected
    time.sleep(60)
    slice_obj.wait_ssh(timeout=900, interval=15, progress=True)
    remount_ceph(node)


def remount_ceph(node) -> None:
    node.execute("for s in ~/.ceph/*/mount_*.sh; do [ -f \"$s\" ] && sudo bash \"$s\"; done; true",
                 quiet=True)


def pin_to_nic(slice_obj, cfg: dict) -> None:
    """Pin each node's vCPUs to host cores on the NUMA node of its XFER NIC.

    CPU pinning only: memory is not NUMA-tuned, so no VM restart is needed. Runs
    once per node (marker file). Failures are reported and skipped: pinning is a
    tuning step, not a requirement.
    """
    for n in slice_obj.get_nodes():
        if n.get_name() not in (NODE_NAME, RECEIVER_NAME):
            continue
        out, _ = n.execute("test -f ~/.cryoem_pinned && echo PINNED", quiet=True)
        if "PINNED" in out:
            continue
        try:
            numa = n.get_component(name="xfer_nic").get_numa_node()
            print(f"### PROGRESS: Pinning {n.get_name()} vCPUs to NUMA node {numa} "
                  f"(the {cfg['xfer_nic']} NUMA node)...")
            n.pin_cpu(component_name="xfer_nic")
            n.execute("touch ~/.cryoem_pinned", quiet=True)
        except Exception as exc:
            print(f"### PROGRESS: WARNING — CPU pinning on {n.get_name()} skipped: {exc}")


# --- downstream receiver + dedicated transfer link ----------------------------
def add_receiver(slice_obj, node, cfg: dict, host: str | None = None) -> None:
    """Add the receiver VM and a dedicated site-to-site L2 link (XFER) to the topology.

    The link is separate from the FABNetv4 interface the GPU node uses for Ceph,
    so the demo transfer has its own path and its own private subnet. Both ends use
    XFER_NIC: the shared NIC_Basic, or a dedicated ConnectX (port 1) for more bandwidth.
    To swap the FIU VM for CAPE later, replace the receiver interface with a CAPE
    facility port on this same L2 network.
    """
    rx = slice_obj.add_node(name=RECEIVER_NAME, site=cfg["receiver_site"], host=host,
                            cores=cfg["receiver_cores"], ram=cfg["receiver_ram"],
                            disk=cfg["receiver_disk"], image=IMAGE)
    a = node.add_component(model=cfg["xfer_nic"], name="xfer_nic").get_interfaces()[0]
    b = rx.add_component(model=cfg["xfer_nic"], name="xfer_nic").get_interfaces()[0]
    slice_obj.add_l2network(name=XFER_NET, interfaces=[a, b],
                            subnet=ipaddress.IPv4Network(cfg["xfer_subnet"]))
    a.set_mode("auto")
    b.set_mode("auto")


def xfer_ip(n) -> str:
    return str(n.get_interface(network_name=XFER_NET).get_ip_addr())


# VM-to-VM SSH runs on the dataplane: destination AND source are XFER addresses,
# never the management network.
SSH_XFER_CONFIG = """Host {name}
    HostName {ip}
    BindAddress {src}
    User {user}
    IdentityFile ~/.ssh/xfer_ed25519
    StrictHostKeyChecking accept-new
"""


def configure_transfer(node, rx) -> dict:
    """Install transfer tools, tune TCP + MTU on the XFER link, set up SSH cryoem -> receiver."""
    src_ip, dst_ip = xfer_ip(node), xfer_ip(rx)
    print(f"### PROGRESS: Transfer link {XFER_NET}: {NODE_NAME} {src_ip} <-> {RECEIVER_NAME} {dst_ip}")
    rx.upload_directory(TOOLS_DIR, ".")
    rx.execute("bash ~/tools/setup-receiver.sh", quiet=True)

    mtu = 9000
    for n, ip in ((node, src_ip), (rx, dst_ip)):
        n.execute(f"bash ~/tools/net-tune.sh {ip} {mtu}", quiet=True)
    out, _ = node.execute(f"ping -c 3 -W 2 -M do -s 8972 {dst_ip} >/dev/null && echo JUMBO_OK", quiet=True)
    if "JUMBO_OK" not in out:
        mtu = 1500
        for n, ip in ((node, src_ip), (rx, dst_ip)):
            n.execute(f"bash ~/tools/net-tune.sh {ip} {mtu}", quiet=True)
    rtt, _ = node.execute(f"ping -c 5 -q {dst_ip} | tail -1", quiet=True)
    print(f"### PROGRESS: XFER MTU {mtu}; RTT {rtt.strip()}")

    # SSH trust: a dedicated key on cryoem, authorized on the receiver.
    pub, _ = node.execute("test -f ~/.ssh/xfer_ed25519 || ssh-keygen -q -t ed25519 -N '' "
                          "-f ~/.ssh/xfer_ed25519; cat ~/.ssh/xfer_ed25519.pub", quiet=True)
    pub = pub.strip().splitlines()[-1]
    rx.execute(f"mkdir -p ~/.ssh && (grep -qxF '{pub}' ~/.ssh/authorized_keys 2>/dev/null "
               f"|| echo '{pub}' >> ~/.ssh/authorized_keys)", quiet=True)
    cfg_text = SSH_XFER_CONFIG.format(name=RECEIVER_NAME, ip=dst_ip, src=src_ip, user=rx.get_username())
    # Prepend the Include (works when ~/.ssh/config is missing or empty, unlike `sed 1i`).
    node.execute(f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                 f"printf '%s' '{cfg_text}' > ~/.ssh/config.d-xfer && "
                 f"(grep -qs config.d-xfer ~/.ssh/config || "
                 f"{{ {{ echo 'Include ~/.ssh/config.d-xfer'; cat ~/.ssh/config 2>/dev/null || true; }} "
                 f"> ~/.ssh/config.new && mv ~/.ssh/config.new ~/.ssh/config; }}) && "
                 f"chmod 600 ~/.ssh/config ~/.ssh/config.d-xfer", quiet=True)
    # Prove the session is on the dataplane: $SSH_CONNECTION = client_ip client_port server_ip server_port.
    out, _ = node.execute(f"ssh -o BatchMode=yes -o ConnectTimeout=10 {RECEIVER_NAME} "
                          "'echo SSHCONN $SSH_CONNECTION'", quiet=True)
    conn = next((l.split()[1:] for l in out.splitlines() if l.startswith("SSHCONN ")), [])
    if len(conn) != 4 or conn[0] != src_ip or conn[2] != dst_ip:
        raise RuntimeError(f"ssh {NODE_NAME} -> {RECEIVER_NAME} is not on the {XFER_NET} dataplane "
                           f"(expected {src_ip} -> {dst_ip}, got {' '.join(conn) or out.strip()})")
    print(f"### PROGRESS: SSH {NODE_NAME} -> {RECEIVER_NAME} verified on dataplane {conn[0]} -> {conn[2]}")
    return {"name": RECEIVER_NAME, "site": rx.get_site(), "host": rx.get_host(),
            "xfer_ip": dst_ip, "source_xfer_ip": src_ip, "mtu": mtu, "rtt": rtt.strip(),
            "inbox": "~/cryoem-inbox", "ssh": rx.get_ssh_command()}


def start(slice_name: str) -> None:
    cfg = load_config()
    fablib = get_fablib()
    print(f"### PROGRESS: FABRIC project: {project_name(fablib)} (your default FABRIC config)")

    try:
        existing = fablib.get_slice(name=slice_name)
        if existing and "Dead" not in str(existing.get_state()) and "Closing" not in str(existing.get_state()):
            print(f"### PROGRESS: Slice '{slice_name}' already exists ({existing.get_state()}); "
                  "re-running node setup only.")
            slice_obj = existing
        else:
            slice_obj = None
    except Exception:
        slice_obj = None

    if slice_obj is None:
        preflight(cfg, fablib)
        print(f"### PROGRESS: Searching for a host with {cfg['num_gpus']}x GPU "
              f"({cfg['gpu_model']}), {cfg['cores']}c/{cfg['ram']}G/{cfg['disk']}G"
              f"{' and a free ' + cfg['xfer_nic'] if cfg['receiver_site'] else ''}...")
        res = fablib.get_resources(update=True)
        rx_host = None
        if cfg["receiver_site"]:
            rx_host = find_receiver_host(res, cfg)
            if rx_host is None:
                print(f"### PROGRESS: ERROR — no host at RECEIVER_SITE={cfg['receiver_site']} has a "
                      f"free {cfg['xfer_nic']} and {cfg['receiver_cores']}c/{cfg['receiver_ram']}G/"
                      f"{cfg['receiver_disk']}G. Try another XFER_NIC (FIU has ConnectX-5, not -6) "
                      "or RECEIVER_SITE.")
                sys.exit(2)
        pick = find_gpu_host(res, cfg)
        if pick is None:
            print(f"### PROGRESS: ERROR — no host has {cfg['num_gpus']} free GPU(s) matching "
                  f"GPU_MODEL={cfg['gpu_model']} SITE={cfg['site'] or 'any'}"
                  f"{' with a free ' + cfg['xfer_nic'] if cfg['receiver_site'] else ''}. "
                  "Try NUM_GPUS=1, a different GPU_MODEL/SITE, or XFER_NIC.")
            sys.exit(2)
        site, host, model, avail = pick
        print(f"### PROGRESS: Placing on {site} / {host}: {cfg['num_gpus']}x {model} "
              f"({avail} free)")

        slice_obj = fablib.new_slice(name=slice_name)
        node = slice_obj.add_node(name=NODE_NAME, site=site, host=host, cores=cfg["cores"],
                                  ram=cfg["ram"], disk=cfg["disk"], image=IMAGE)
        for i in range(cfg["num_gpus"]):
            node.add_component(model=model, name=f"gpu{i + 1}")
        # CephFS project volume: adds a FABNetv4 link and mounts at post-boot.
        node.enable_storage(cluster=CEPH_CLUSTER)
        if cfg["receiver_site"]:
            print(f"### PROGRESS: Receiver at {cfg['receiver_site']} / {rx_host} "
                  f"({cfg['receiver_cores']}c/{cfg['receiver_ram']}G/{cfg['receiver_disk']}G) "
                  f"on dedicated L2 link {XFER_NET} {cfg['xfer_subnet']} via {cfg['xfer_nic']}")
            add_receiver(slice_obj, node, cfg, host=rx_host)

        print("### PROGRESS: Submitting slice and waiting for StableOK (10-20 min)...")
        try:
            slice_obj.submit(wait=True, progress=True)
        except Exception as exc:
            print(f"### PROGRESS: ERROR — submit failed: {exc}")
            if not cfg["keep_slice"]:
                try:
                    fablib.get_slice(name=slice_name).delete()
                except Exception:
                    pass
            sys.exit(1)
        print("### PROGRESS: Slice is StableOK — no further polling.")
        slice_obj = fablib.get_slice(name=slice_name)

    node = slice_obj.get_node(name=NODE_NAME)

    # Existing slice from before the receiver was added: modify it in place.
    names = {n.get_name() for n in slice_obj.get_nodes()}
    if cfg["receiver_site"] and RECEIVER_NAME not in names:
        print(f"### PROGRESS: Adding receiver at {cfg['receiver_site']} to the existing slice...")
        add_receiver(slice_obj, node, cfg,
                     host=find_receiver_host(fablib.get_resources(update=True), cfg))
        slice_obj.submit(wait=True, progress=True)
        slice_obj = fablib.get_slice(name=slice_name)
        node = slice_obj.get_node(name=NODE_NAME)

    if cfg["pin_cpu"] and cfg["receiver_site"]:
        pin_to_nic(slice_obj, cfg)
        slice_obj = fablib.get_slice(name=slice_name)
        node = slice_obj.get_node(name=NODE_NAME)

    print("### PROGRESS: Installing CUDA, MotionCor3, CTFFIND4"
          f"{', RELION 5.0' if cfg['install_relion'] else ''} (20-45 min)...")
    node.upload_directory(TOOLS_DIR, ".")
    rc = run_setup(node, cfg)
    if rc == 3:
        reboot_and_remount(slice_obj, node)
        rc = run_setup(node, cfg)
    if rc != 0:
        print(f"### PROGRESS: ERROR — node setup failed (rc={rc}). See ~/setup-node.log on "
              f"the node. Re-run the weave to retry setup on the existing slice.")
        sys.exit(1)

    gpus, _ = node.execute("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
                           quiet=True)
    try:
        vol = data_dir(node)
    except RuntimeError as exc:
        print(f"### PROGRESS: ERROR — {exc}. Re-run the weave to retry setup on the existing slice.")
        sys.exit(1)

    receiver = None
    if cfg["receiver_site"]:
        try:
            receiver = configure_transfer(node, slice_obj.get_node(name=RECEIVER_NAME))
        except Exception as exc:
            print(f"### PROGRESS: ERROR — transfer setup failed: {exc}. "
                  "Re-run the weave to retry on the existing slice.")
            sys.exit(1)

    extend_lease(slice_obj, cfg["lease_days"])

    # Demo web app (optional: the notebook works without it).
    app_ok = False
    try:
        out, _ = node.execute("bash ~/tools/setup-app.sh 8080 2>&1 | tail -3", quiet=True)
        app_ok = "demo app up" in out
        if not app_ok:
            print(f"### PROGRESS: WARNING — demo app not installed: {out.strip()[-300:]}")
    except Exception as exc:
        print(f"### PROGRESS: WARNING — demo app not installed: {exc}")

    info = {
        "slice_name": slice_name,
        "project": project_name(fablib),
        "node": NODE_NAME,
        "site": node.get_site(),
        "host": node.get_host(),
        "mgmt_ip": str(node.get_management_ip()),
        "ssh": node.get_ssh_command(),
        "gpus": [g.strip() for g in gpus.strip().splitlines()],
        "storage_volume": STORAGE_VOLUME,
        "ceph_cluster": CEPH_CLUSTER,
        "data_dir": vol,
        "receiver": receiver,
        "xfer_nic": cfg["xfer_nic"] if receiver else None,
        "cpu_pinned": bool(cfg["pin_cpu"] and receiver),
        "lease_end": slice_obj.get_lease_end(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "slice_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    print("### PROGRESS: ---------------------------------------------")
    print(f"### PROGRESS: Node:    {NODE_NAME} @ {info['site']} ({info['host']})")
    for g in info["gpus"]:
        print(f"### PROGRESS: GPU:     {g}")
    print(f"### PROGRESS: Storage: {STORAGE_VOLUME} mounted at {vol}")
    print(f"### PROGRESS: SSH:     {info['ssh']}")
    if receiver:
        print(f"### PROGRESS: Receiver: {RECEIVER_NAME} @ {receiver['site']} — {XFER_NET} "
              f"{receiver['source_xfer_ip']} -> {receiver['xfer_ip']} (MTU {receiver['mtu']}, "
              f"{cfg['xfer_nic']}{', CPU pinned' if cfg['pin_cpu'] else ''}), inbox {receiver['inbox']}")
    if app_ok:
        print(f"### PROGRESS: Demo app: LoomAI Tunnels > slice {slice_name}, node {NODE_NAME}, port 8080 "
              "(prepare it once with notebook §7c).")
    print(f"### PROGRESS: READY — open cryoem_experiment.ipynb (slice '{slice_name}').")


def status(slice_name: str) -> None:
    """One-shot health check (not a polling loop)."""
    try:
        slice_obj = get_fablib().get_slice(name=slice_name)
        node = slice_obj.get_node(name=NODE_NAME)
        out, _ = node.execute(
            f"nvidia-smi -L; which MotionCor3 ctffind; ls {VOLUME_GLOB}/ | head", quiet=True)
        print(f"### PROGRESS: {slice_obj.get_state()}, lease ends {slice_obj.get_lease_end()}")
        print(out)
        if RECEIVER_NAME in {n.get_name() for n in slice_obj.get_nodes()}:
            out, _ = node.execute(f"ssh -o BatchMode=yes -o ConnectTimeout=10 {RECEIVER_NAME} "
                                  "'hostname; df -h ~/cryoem-inbox | tail -1'", quiet=True)
            print(f"receiver via {XFER_NET}:\n{out}")
    except Exception as exc:
        print(f"### PROGRESS: ERROR — {exc}")
        sys.exit(1)


def extend(slice_name: str, days: int | None = None) -> None:
    days = days or int(os.environ.get("LEASE_DAYS", "14"))
    ok = extend_lease(get_fablib().get_slice(name=slice_name), days)
    sys.exit(0 if ok else 1)


def stop(slice_name: str) -> None:
    print(f"### PROGRESS: Deleting slice '{slice_name}'...")
    try:
        get_fablib().get_slice(name=slice_name).delete()
        print("### PROGRESS: Slice deleted.")
    except Exception as exc:
        print(f"### PROGRESS: Could not delete (may already be gone): {exc}")


if __name__ == "__main__":
    actions = {"start": start, "status": status, "extend": extend, "stop": stop}
    args = sys.argv[1:]
    if args and args[0] in actions:
        action, args = args[0], args[1:]
    else:
        action = "start"
    name = args[0] if args else os.environ.get("SLICE_NAME", "cryoem-demo")
    if action == "extend":
        extend(name, int(args[1]) if len(args) > 1 else None)
    else:
        actions[action](name)

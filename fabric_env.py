"""Shared FABRIC connection settings for the CryoEM Without Borders weave.

Both ``provision.py`` (the weave) and ``cryoem_experiment.ipynb`` import this so
they talk to FABRIC the same way: with the user's default FABRIC config (``fabric_rc``,
LoomAI's Configure view), its token and its project. Nothing is overridden here, so
the weave works in whatever project the user has selected. (Passing a project_id would
make fablib refresh the token into that project.)
"""

from __future__ import annotations

import os

WEAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# CephFS project volume (the Globus collection is backed by this subvolume).
CEPH_CLUSTER = os.environ.get("CEPH_CLUSTER", "west")
STORAGE_VOLUME = os.environ.get("STORAGE_VOLUME", "cryoem-without-borders")
# fablib mounts the volume under /mnt/cephfs/<cluster>/<bastion user>/<project uuid>_<volume>.
# The user part differs per account, so the path is discovered on the node (see data_dir()).
VOLUME_GLOB = f"/mnt/cephfs/{CEPH_CLUSTER}/*/*_{STORAGE_VOLUME}"
REMOTE_WORK = "~/cryoem-work"
REMOTE_PY = "~/cryoem-venv/bin/python"
NODE_NAME = "cryoem"
SLICE_INFO = os.path.join(WEAVE_DIR, "results", "slice_info.json")


def data_dir(node) -> str:
    """Mount point of the cryoem-without-borders volume on ``node`` (under /mnt).

    Raises if the volume is not mounted.
    """
    out, _ = node.execute(
        f"for d in {VOLUME_GLOB}; do mountpoint -q \"$d\" && echo \"$d\" && break; done",
        quiet=True)
    path = out.strip().splitlines()[0] if out.strip() else ""
    if not path:
        raise RuntimeError(f"Storage volume '{STORAGE_VOLUME}' is not mounted on "
                           f"{node.get_name()} (expected {VOLUME_GLOB})")
    return path


def resolve_slice_name(fablib) -> str:
    """Slice to use: $SLICE_NAME, else results/slice_info.json, else the only live
    slice in the project that has a ``cryoem`` node."""
    import json

    if os.environ.get("SLICE_NAME"):
        return os.environ["SLICE_NAME"]
    if os.path.exists(SLICE_INFO):
        with open(SLICE_INFO) as fh:
            return json.load(fh)["slice_name"]
    live = []
    for s in fablib.get_slices():
        try:
            if any(n.get_name() == NODE_NAME for n in s.get_nodes()):
                live.append(s.get_name())
        except Exception:
            pass
    if len(live) == 1:
        return live[0]
    raise RuntimeError(
        "Cannot pick the slice automatically "
        f"({'none found' if not live else 'several: ' + ', '.join(live)}). "
        "Set SLICE_NAME (env) or run the weave so it writes results/slice_info.json.")


def get_fablib():
    """FablibManager with the default FABRIC config: its token and its project."""
    from fabrictestbed_extensions.fablib.fablib import FablibManager

    return FablibManager()


def token_project(fablib) -> dict:
    """{'name', 'uuid', 'tags'} of the project in the token FABRIC checks (best effort)."""
    import base64
    import json

    try:
        with open(os.path.expanduser(fablib.get_token_location())) as fh:
            tok = json.load(fh)["id_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4)))
        projects = claims.get("projects") or []
        return projects[0] if projects else {}
    except Exception:
        return {}

"""Shared FABRIC connection settings for the CryoEM Without Borders weave.

Both ``provision.py`` (the weave) and ``cryoem_experiment.ipynb`` import this so
they talk to FABRIC with the same token and project:

- Token: the user's default FABRIC token (``FABRIC_TOKEN_LOCATION`` / ``fabric_rc``,
  normally ``~/work/fabric_config/id_token.json``). Set ``CRYOEM_TOKEN`` to use another file.
- Project: "CryoEM Without Borders" (override with ``CRYOEM_PROJECT_ID``).

Everything else (bastion user/key, slice SSH keys) comes from the normal
``fabric_rc`` (``~/work/fabric_config/fabric_rc``). The project is passed
explicitly because values in ``fabric_rc`` win over environment variables.
"""

from __future__ import annotations

import os

WEAVE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_NAME = "CryoEM Without Borders"
PROJECT_ID = os.environ.get("CRYOEM_PROJECT_ID", "c93be0c1-696c-4c36-9dbd-03b14be196af")

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
    """FablibManager bound to the CryoEM project, using the default token unless
    ``CRYOEM_TOKEN`` names another one."""
    from fabrictestbed_extensions.fablib.fablib import FablibManager

    token = os.environ.get("CRYOEM_TOKEN")
    if token:
        return FablibManager(project_id=PROJECT_ID, token_location=os.path.expanduser(token))
    return FablibManager(project_id=PROJECT_ID)

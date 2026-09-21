"""Content receipts for repeated admissions of an immutable local checkpoint.

Receipts are trusted local artifacts, not signatures for untrusted checkpoints.
Reuse requires the same file inventory and filesystem identities, including ctime.
Moving or modifying a snapshot requires another complete content verification.
"""

import hashlib
import json
from pathlib import Path


def file_identity(path):
    stat = path.stat()
    return dict(
        name=path.name,
        bytes=stat.st_size,
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def checkpoint_identity(directory, *, receipt=None, output=None):
    root = Path(directory).resolve()
    files = sorted(
        (*root.glob("*.safetensors"), *root.glob("*.json")), key=lambda p: p.name
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("checkpoint directory has no safetensors shards")
    identities = [file_identity(p) for p in files]
    if receipt is not None:
        document = json.loads(Path(receipt).read_text())
        if (
            document.get("schema") != 1
            or document.get("root") != str(root)
            or document.get("files") != identities
        ):
            raise ValueError(
                "checkpoint content receipt is stale; verify immutable files again"
            )
        fingerprint = document["fingerprint"]
        if (
            len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
            or set(document["sha256"]) != {p.name for p in files}
        ):
            raise ValueError("invalid checkpoint content receipt")
        return document
    digest, hashes = hashlib.sha256(), {}
    for path, identity in zip(files, identities, strict=True):
        digest.update(path.name.encode() + b"\0")
        digest.update(identity["bytes"].to_bytes(8, "little"))
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
                file_hash.update(block)
        hashes[path.name] = file_hash.hexdigest()
    if [file_identity(p) for p in files] != identities:
        raise ValueError("checkpoint changed during content verification")
    document = dict(
        schema=1,
        root=str(root),
        files=identities,
        fingerprint=digest.hexdigest(),
        sha256=hashes,
    )
    if output is not None:
        destination = Path(output)
        if destination.resolve().parent == root:
            raise ValueError("content receipt must be outside the checkpoint inventory")
        with destination.open("x") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
    return document

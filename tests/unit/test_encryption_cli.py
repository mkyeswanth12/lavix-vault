from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "enc" / "encryption.py"


def _write_keys(directory: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    unlock = directory / "unlock"
    unlock.mkdir()
    (unlock / "private_4096.pem").write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (directory / "public_4096.pem").write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _run(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=directory,
        env={
            **os.environ,
            "PUBLIC_KEY": str(directory / "public_4096.pem"),
            "PRIVATE_KEY": str(directory / "unlock" / "private_4096.pem"),
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_encryption_cli_round_trip(tmp_path: Path) -> None:
    _write_keys(tmp_path)
    source = tmp_path / "proof.txt"
    content = b"Lavix encrypted round-trip proof\n" * 32
    source.write_bytes(content)

    encrypted = _run(tmp_path, "encrypt", str(source), "--delete")
    assert encrypted.returncode == 0, encrypted.stderr
    assert not source.exists()

    decrypted = _run(tmp_path, "decrypt", str(tmp_path / "proof.txt.enc"))
    assert decrypted.returncode == 0, decrypted.stderr
    assert source.read_bytes() == content


def test_decryption_rejects_unbounded_ciphertext_chunk_length(tmp_path: Path) -> None:
    _write_keys(tmp_path)
    source = tmp_path / "tampered.bin"
    source.write_bytes(b"safe payload")
    encrypted = _run(tmp_path, "encrypt", str(source), "--delete")
    assert encrypted.returncode == 0, encrypted.stderr

    encrypted_path = tmp_path / "tampered.bin.enc"
    payload = bytearray(encrypted_path.read_bytes())
    payload[12:16] = (2**32 - 1).to_bytes(4, "big")
    encrypted_path.write_bytes(payload)

    result = _run(tmp_path, "decrypt", str(encrypted_path))
    assert result.returncode != 0
    assert "invalid chunk length" in result.stderr

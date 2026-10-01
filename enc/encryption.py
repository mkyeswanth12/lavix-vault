#!/usr/bin/env python3
"""
Lavix Vault – Unified Encryption / Decryption Script
---------------------------------------------------
Encrypt:
  encryption.py encrypt <file> [--delete]

Decrypt:
  encryption.py decrypt <file>.enc

RSA is MANDATORY.
"""

import hashlib
import os
import pathlib
import secrets
import struct
import subprocess
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

CHUNK_SIZE = 64 * 1024 * 1024
PUBLIC_KEY = pathlib.Path(os.environ.get("PUBLIC_KEY", "/app/enc/public_4096.pem"))
PRIVATE_KEY = pathlib.Path(os.environ.get("PRIVATE_KEY", "/run/secrets/private_key"))


# =========================
# HELPERS
# =========================
def die(msg):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()


# =========================
# ENCRYPT
# =========================
def encrypt_file(path: pathlib.Path):
    if not path.exists():
        die(f"File not found: {path}")

    if not PUBLIC_KEY.exists():
        die(f"Missing public key: {PUBLIC_KEY}")

    print(f"\n🔐 Encrypting: {path}")

    aes_key = secrets.token_bytes(32)
    nonce_base = secrets.token_bytes(12)

    enc_path = path.with_suffix(path.suffix + ".enc")
    key_path = path.with_suffix(path.suffix + ".key")
    rsa_path = path.with_suffix(path.suffix + ".key.rsa4096")
    sha_path = path.with_suffix(path.suffix + ".sha256")

    aesgcm = AESGCM(aes_key)

    with open(path, "rb") as src, open(enc_path, "wb") as dst:
        dst.write(nonce_base)
        chunks = 0
        while data := src.read(CHUNK_SIZE):
            nonce = (int.from_bytes(nonce_base, "big") + chunks).to_bytes(12, "big")
            ct = aesgcm.encrypt(nonce, data, None)
            dst.write(len(ct).to_bytes(4, "big"))
            dst.write(ct)
            chunks += 1

    # write temp key file
    key_path.write_text(f"AES_KEY={aes_key.hex()}\nNONCE={nonce_base.hex()}\nCHUNKS={chunks}\n")

    # RSA wrap (MANDATORY)
    print("🔑 Wrapping AES key with RSA-4096")
    r = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-encrypt",
            "-pubin",
            "-inkey",
            str(PUBLIC_KEY),
            "-in",
            str(key_path),
            "-out",
            str(rsa_path),
        ],
        capture_output=True,
        text=True,
    )

    if r.returncode != 0:
        die("RSA encryption failed:\n" + r.stderr)

    os.remove(key_path)

    sha = sha256_file(path)
    sha_path.write_text(f"{sha}\n")

    print("✅ Encryption complete")
    print("  ", enc_path.name)
    print("  ", rsa_path.name)
    print("  ", sha_path.name)


# =========================
# DECRYPT
# =========================
def decrypt_file(enc_path: pathlib.Path):
    if not enc_path.exists():
        die(f"File not found: {enc_path}")

    if not PRIVATE_KEY.exists():
        die(f"Missing private key: {PRIVATE_KEY}")

    rsa_path = enc_path.with_suffix(".key.rsa4096")
    sha_path = enc_path.with_suffix(".sha256")
    out_path = enc_path.with_suffix("")

    if not rsa_path.exists():
        die(f"Missing RSA key file: {rsa_path}")

    print(f"\n🔓 Decrypting: {enc_path}")

    # unwrap AES key
    r = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-decrypt",
            "-inkey",
            str(PRIVATE_KEY),
            "-in",
            str(rsa_path),
        ],
        capture_output=True,
        text=True,
    )

    if r.returncode != 0:
        die("RSA decrypt failed:\n" + r.stderr)

    lines = r.stdout.strip().splitlines()
    vals = dict(line.split("=", 1) for line in lines)

    aes_key = bytes.fromhex(vals["AES_KEY"])
    nonce_base = bytes.fromhex(vals["NONCE"])
    chunks = int(vals["CHUNKS"])

    aesgcm = AESGCM(aes_key)

    with open(enc_path, "rb") as fin, open(out_path, "wb") as fout:
        fin.seek(12)
        for i in range(chunks):
            header = fin.read(4)
            if len(header) != 4:
                die("Encrypted file is truncated before a chunk header")
            ln = struct.unpack(">I", header)[0]
            if ln < 16 or ln > CHUNK_SIZE + 16:
                die("Encrypted file contains an invalid chunk length")
            ct = fin.read(ln)
            if len(ct) != ln:
                die("Encrypted file is truncated inside a chunk")
            nonce = (int.from_bytes(nonce_base, "big") + i).to_bytes(12, "big")
            fout.write(aesgcm.decrypt(nonce, ct, None))
        if fin.read(1):
            die("Encrypted file contains trailing data")

    if sha_path.exists():
        expected = sha_path.read_text().strip()
        actual = sha256_file(out_path)
        if expected != actual:
            die("SHA-256 verification FAILED")
        print("✅ SHA-256 verified")

    print("🎉 Decryption complete:", out_path.name)


# =========================
# MAIN
# =========================
if len(sys.argv) < 3:
    print("""
Usage:
  encryption.py encrypt <file> [--delete]
  encryption.py decrypt <file>.enc
""")
    sys.exit(1)

mode = sys.argv[1]
path = pathlib.Path(sys.argv[2]).resolve()
delete_original = "--delete" in sys.argv

if mode == "encrypt":
    encrypt_file(path)
    if delete_original:
        path.unlink()
        print("🗑️ Original file deleted:", path.name)

elif mode == "decrypt":
    decrypt_file(path)
else:
    die("Invalid mode")

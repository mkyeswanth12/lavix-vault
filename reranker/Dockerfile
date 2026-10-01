# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM michaelf34/infinity:0.0.77-cpu@sha256:94db836145f609c508f3f296f3b9bd0e78ae088025e879664d073dfd9f929e96 AS runtime

# Reranker model preset, chosen at build time:
#   docker compose build --build-arg RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 reranker
# (or set RERANKER_MODEL in docker-compose.yaml x-settings; compose passes it
# through automatically). Only the presets below are supported: each pins an
# exact Hugging Face revision plus per-file SHA-256 hashes, so a typo or an
# unvetted model fails the build with a clear message instead of baking an
# unknown artifact. Runtime stays offline (HF_HUB_OFFLINE=1); the download
# happens only here, at build time.
ARG RERANKER_MODEL=BAAI/bge-reranker-base

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

# Keep only the files required by the Torch cross-encoder. The Xet chunk
# cache is a BuildKit cache mount, so it never duplicates the model in the
# resulting image.
RUN --mount=type=cache,target=/tmp/hf-xet,sharing=locked \
    HF_XET_CACHE=/tmp/hf-xet python - <<PY
import os
import time

from huggingface_hub import snapshot_download

# repo_id: (revision, allow_patterns, {filename: sha256}).
# Every preset bakes into the SAME /models/reranker directory, and the CMD
# below serves it with --served-model-name set to the baked repo id, so the
# compose service carries no command block and the app's requested model
# always matches the baked artifact.
# Hashes were computed from real downloads; do not invent new rows without
# downloading the revision and recording `sha256sum` output.
PRESETS = {
    "BAAI/bge-reranker-base": (
        "2cfc18c9415c912f9d8155881c133215df768a70",
        [
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        {
            "config.json": "289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708",
            "model.safetensors": "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd",
            "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            "special_tokens_map.json": "d5469a60db23249c7f8945013d78df30b44b6bf686c6bb4740f4223f77b1b535",
            "tokenizer.json": "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e",
            "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
        },
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2": (
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
        {
            "config.json": "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
            "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
            "special_tokens_map.json": "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
            "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "tokenizer_config.json": "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
            "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        },
    ),
}

wanted = os.environ["RERANKER_MODEL"]
if wanted not in PRESETS:
    raise SystemExit(
        f"Unsupported RERANKER_MODEL {wanted!r}. Supported presets: "
        + ", ".join(sorted(PRESETS))
    )
revision, patterns, hashes = PRESETS[wanted]
local_dir = "/models/reranker"

for attempt in range(6):
    try:
        snapshot_download(
            repo_id=wanted,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=patterns,
            max_workers=1,
        )
        break
    except Exception:
        if attempt == 5:
            raise
        time.sleep(min(2**attempt, 16))

import hashlib
import pathlib

for filename, expected in hashes.items():
    digest = hashlib.sha256((pathlib.Path(local_dir) / filename).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Hash mismatch for {filename}: got {digest}, want {expected}")
print(f"baked {wanted} at {revision} (hashes verified)")
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    HOME=/tmp \
    RERANKER_MODEL=${RERANKER_MODEL} \
    INFINITY_ANONYMOUS_USAGE_STATS=0 \
    HF_HUB_DISABLE_TELEMETRY=1

LABEL org.lavix.model.revision.pinned="true"

# Serve the baked preset. A shell-form CMD would hand `-c` to the base
# image's entrypoint, so the serve line lives in an explicit entrypoint
# script instead (same flags the compose file used to carry).
COPY --chown=10003:10003 reranker/serve.sh /serve.sh
ENTRYPOINT ["/serve.sh"]

USER 10003:10003

#!/bin/sh
# Serve the baked reranker preset (fails closed on an unknown model name).
set -eu
exec infinity_emb v2 --model-id /models/reranker --served-model-name "${RERANKER_MODEL:?RERANKER_MODEL is not set}" --engine torch --device cpu --batch-size 8 --no-trust-remote-code --no-compile --no-bettertransformer --port 7997

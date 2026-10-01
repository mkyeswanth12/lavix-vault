"""Run with ``python -m agent_runtime`` inside the isolated image."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("agent_runtime.main:app", host="0.0.0.0", port=8090)

# Concepts

Who this page is for: first-time users. One short paragraph per term, no jargon left undefined.

**Docker.** Software that runs programs in isolated boxes on your machine. Like
apartments in one building: same foundation, separate homes.

**Container.** One running box. Lavix Vault is a set of containers that talk to
each other. If one crashes, the others keep running.

**Docker Compose.** A tool that starts all the containers together from one
recipe file (`docker-compose.yaml`). You run `docker compose up -d` once
instead of starting fourteen things by hand.

**Image.** The frozen template a container starts from, like a class photo
every copy is printed from. Images are downloaded once, then reused.

**Volume.** A container's notebook that survives restarts. Your database,
uploaded files, and settings live in volumes; deleting a volume deletes that data.

**Ollama.** A separate program that runs language models on your own hardware.
Lavix Vault does not include it — you install it and tell Lavix where it is
(`OLLAMA_URL`).

**Language model.** A program that reads and writes text: it answers questions,
rewrites queries, and summarizes. Lavix uses small local ones (for example
`llama3.2:3b`; the `3b` means about 3 billion parameters, i.e. a small,
fast model).

**Embeddings.** Turning a piece of text into a list of numbers (a vector) that
captures its meaning, so "car" and "automobile" end up close together. Your
Ollama server computes these; the numbers are stored next to your file chunks.

**Vector search.** Finding file chunks by meaning instead of exact words: your
question is embedded too, and the database returns the chunks whose numbers
are closest. Think of it as "find me the paragraph that feels most similar".

**Reranker.** A second, pickier judge. Vector search returns candidates quickly;
the reranker re-reads the top ones with the actual question and keeps the best.
This is what makes answers cite the right paragraph.

**RAG ("answers from your documents").** Retrieval-Augmented Generation: the
system first retrieves passages from *your* files, then generates an answer
*only* from those passages, with source cards you can click. If nothing
relevant is found, it says so instead of guessing.

**Agent.** The program that plans how to answer: it rewrites your question
into searches, gathers evidence, checks the draft, and writes the final
answer. Lavix runs it in a separate sandboxed container (built on IBM's
open-source CUGA framework) that holds no passwords or keys — each question
hands it a 300-second signed pass listing exactly the files it may touch.

**Encryption keys.** Two secrets protect each file: a random per-file key that
scrambles the contents (AES-256-GCM), and your RSA-4096 keypair that locks that
per-file key. The private key lives in `secrets/private_4096.pem`. Lose it and
your files can never be decrypted — there is no spare copy.

**Admin vs user.** A user uploads files and chats. An admin additionally manages
people (roles, quotas, password resets), models, registration, and system
settings. The first registered user should be promoted to admin, then
registration should be turned off. Promotion also grants file deletion,
which is denied by default for new accounts.

**Graph memory.** An optional extra index for relationship memory: instead of
only looking up facts ("you like concise answers"), it can also follow
connections between people, projects, and preferences — like a corkboard with
strings between related notes, instead of a flat list. It lives in an optional
Neo4j database, on unless you opt out; PostgreSQL alone handles all memory
otherwise.

Next steps: [Install](install.md), [Configuration](configuration.md).

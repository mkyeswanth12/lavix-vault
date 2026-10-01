# User guide

Who this page is for: anyone using Lavix Vault day to day.

## Sign in and settings

Open `http://localhost:3005`. Sign in with username (or email) + password, or
register if the admin left registration on. If your session expires you land
back at Sign In — sign in again, nothing is lost. In Settings you get Profile
(display name, username, password change), Preferences (theme, language),
Interface (font size, send-on-Enter, timestamps, list/grid, thumbnail
regrouping), and AI Memory (see below).

## Files: upload, folders, trash

<img src="images/2026-10-02-182734_hyprshot.png" alt="Document grid with thumbnails and status pills" width="100%">

- **Upload:** Files > Upload, drag-and-drop (single files, many files, or a
  whole folder — the folder tree is mirrored) or the Upload Files button.
  Pick the target folder or create one. Progress shows in the background bar;
  Upload Results lists what landed. Limits: 512 MB per file; PDFs, Office
  (DOCX/XLSX/PPTX, plus DOC/XLS/PPT via conversion), HTML, CSV, TXT and common
  code/text files, and common images are indexed. Audio/video can be stored but
  are never indexed. Password-protected files wait for the password.
- **Folders:** sidebar tree; New Folder, rename, move/copy (bulk supported),
  paths look like `/Root/...`.
- **Trash:** deleted files wait here; restore them or Empty Trash. Permanent
  delete cannot be undone.

Tip: sort by Newest/Largest/Name; search filters by name.

## AI access: grant and revoke

Files are unreadable to the AI until you grant access (the `AI-READY` pill on
each card). Click **Enable AI** on a card, in bulk from the bulk bar, or from
chat when it asks (`Indexing Access Required` → `Grant & Ask`). The AI-Ready
box shows `INDEXING N/M P%` while work runs (click it to manage or cancel),
`AI-READY` when searchable, `INDEX FAILED` if every file failed (open the row:
`Failed — retry`, or `Locked — needs file password`). **Revoke AI access**
removes the index — the file must be indexed again before the AI can use it.

## Chat

<img src="images/2026-09-25-003903_hyprshot.png" alt="Answer with cited sources" width="100%">

Open AI Chat and type. Answers cite `Sources` (click a `V1` chip to preview the
passage; `N%` is the match score). Use `@` to tag files or folders
(`Scoped to N files: a, b…`); follow-ups keep the scope; clear it to search
everything. Two modes: `Casual` (short, fast) and `Expert` (long, thorough).
Toggles: `MODEL` (active chat model), `FILE SEARCH` (your documents),
`Web Search` (the web, off by default), `AI Persona Prompts`. `Suggested
Files` offers unindexed matches with `Process Selected`/`Skip`. If nothing
relevant exists you get `No verifiable evidence` — rephrase, tag files, or
turn Web on. Copy/Regen sit on hover; token counts (`↑ in · ↓ out`) under each
answer.

<img src="images/2026-09-29-234310_hyprshot.png" alt="File-scoped question with code answer" width="100%">

## Memory (About Me)

<img src="images/2026-09-30-032946_hyprshot.png" alt="Memory review with approval controls" width="100%">

Settings > AI Memory. `Safe Automatic memory` learns preferences from chat
(you approve them); `Pin a preference` adds one by hand (`I prefer concise
answers…`); `About Me` summarizes what is known with counts, retention, and
next expiry; expiry is 30/90/365 days (default 90). `Clear relationship
memory` forgets learned items but keeps pinned preferences. Memory never
leaves your machine; it only personalizes your own answers. The optional
Neo4j graph projection (on by default; admins can opt out) adds
connection-following recall; everything above works without it.

Next steps: [Admin guide](admin-guide.md), [Troubleshooting](troubleshooting.md).

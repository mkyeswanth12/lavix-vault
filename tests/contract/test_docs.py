"""Docs consistency: links resolve, settings/scripts/errors are documented."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _md_files() -> list[Path]:
    return [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]


def test_docs_internal_links_resolve():
    missing: list[str] = []
    for page in _md_files():
        for match in re.finditer(r"\]\(([^)\"'#]+)(?:#[^)\"]*)?\)", page.read_text(encoding="utf-8")):
            target = match.group(1)
            if re.match(r"https?://|mailto:|#", target):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.is_file():
                missing.append(f"{page.name} -> {target}")
    assert missing == []


def test_every_x_settings_key_appears_in_configuration():
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    block = compose.split("x-settings:", 1)[1].split("x-hardened:", 1)[0]
    keys = re.findall(r"^\s{2}([A-Z][A-Z0-9_]+):", block, re.MULTILINE)
    assert keys, "no x-settings keys found"
    configuration = (DOCS / "configuration.md").read_text(encoding="utf-8")
    absent = [key for key in dict.fromkeys(keys) if key not in configuration]
    assert absent == []


def test_every_script_appears_in_reference():
    scripts = sorted(
        path.name
        for path in (ROOT / "scripts").iterdir()
        if path.is_file()
        and path.suffix in {".sh", ".py", ".mjs"}
        and not path.name.startswith(".")
    )
    assert scripts, "no scripts found"
    reference = (DOCS / "reference.md").read_text(encoding="utf-8")
    absent = [name for name in scripts if name not in reference]
    assert absent == []


def test_every_gate_error_string_appears_in_troubleshooting():
    gate = (ROOT / "scripts" / "preflight-gate.sh").read_text(encoding="utf-8")
    templates = re.findall(r'fail\s+"([^"]+)"', gate)
    assert templates, "no gate FAIL strings found"
    troubleshooting = (DOCS / "troubleshooting.md").read_text(encoding="utf-8").lower()
    absent: list[str] = []
    for template in templates:
        # Static words of the template must all be documented; variables,
        # quoting, and parenthetical details may differ.
        words = sorted(
            {
                word.strip("—:;'\"().`")
                for word in re.sub(r"\$+[\w#{}()]+|\$[A-Za-z_#]+", " ", template).lower().split()
                if len(word.strip("—:;'\"().`")) >= 4
            }
        )
        missing = [word for word in words if word not in troubleshooting]
        if missing:
            absent.append(f"{template[:60]}… missing {missing}")
    assert absent == []


def test_no_doc_or_script_mentions_the_removed_graph_overlay():
    offenders: list[str] = []
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "scripts"]
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.md")) + sorted(root.rglob("*.sh"))
        for path in files:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if "docker-compose.graph.yaml" in text:
                offenders.append(f"{path.name}: graph overlay file")
            if re.search(r"--graph\b", text):
                offenders.append(f"{path.name}: --graph flag")
    assert offenders == []


def test_docs_referenced_images_exist():
    missing: list[str] = []
    for page in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = page.read_text(encoding="utf-8")
        targets = re.findall(r"!\[[^\]]*\]\(([^)\"'#]+)\)", text)
        targets += re.findall(r'<img\s[^>]*src="([^"#]+)"', text)
        for target in targets:
            if re.match(r"https?://|mailto:", target):
                continue
            if not (page.parent / target).is_file():
                missing.append(f"{page.name} -> {target}")
    assert missing == []

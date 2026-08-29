"""Parse the CoEST iTrust dataset into requirements, code artefacts, and gold links."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET


# Point this at the extracted dataset folder.
DATA_DIR = Path("data/raw/iTrust")


@dataclass
class Dataset:
    requirements: dict[str, str] = field(default_factory=dict)  # req_id -> text
    code: dict[str, str] = field(default_factory=dict)          # code_id -> source text
    gold_links: set[tuple[str, str]] = field(default_factory=set)  # (req_id, code_id)


def _read_artifact_collection(xml_path: Path, base_dir: Path, read_external: bool) -> dict[str, str]:
    """Parse an artifacts_collection XML into an id -> text dict.

    If read_external is True, <content> is a relative path to a text file
    whose contents become the value. Otherwise <content> is stored as-is.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result: dict[str, str] = {}

    for artifact in root.iter("artifact"):
        art_id = artifact.findtext("id", default="").strip()
        content = artifact.findtext("content", default="").strip()
        if not art_id:
            continue

        if read_external:
            text_path = base_dir / content
            try:
                text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            except FileNotFoundError:
                text = ""
        else:
            text = content

        result[art_id] = text

    return result

def _read_code_collection(xml_path: Path, base_dir: Path) -> dict[str, str]:
    """Parse the code artifacts_collection, reading each source file's text.

    <content> is a relative path to a .java file. The value is the file's
    source code (or an empty string if the file is missing).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result: dict[str, str] = {}

    for artifact in root.iter("artifact"):
        art_id = artifact.findtext("id", default="").strip()
        content = artifact.findtext("content", default="").strip()
        if not art_id:
            continue

        source_path = base_dir / content
        try:
            source = source_path.read_text(encoding="utf-8", errors="ignore").strip()
        except FileNotFoundError:
            source = ""

        result[art_id] = source

    return result

def _read_gold_links(xml_path: Path) -> set[tuple[str, str]]:
    """Parse the answer matrix into a set of (req_id, code_id) links.

    Using a set de-duplicates the repeated links present in the file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    links: set[tuple[str, str]] = set()

    for link in root.iter("link"):
        src = link.findtext("source_artifact_id", default="").strip()
        tgt = link.findtext("target_artifact_id", default="").strip()
        if src and tgt:
            links.add((src, tgt))

    return links

def load_itrust(data_dir: Path = DATA_DIR) -> Dataset:
    """Load the iTrust dataset: requirements, Java code, and Java-only gold links."""
    requirements = _read_artifact_collection(
        data_dir / "source_req.xml", data_dir, read_external=True
    )
    code = _read_code_collection(data_dir / "target_JavaCode.xml", data_dir)
    all_links = _read_gold_links(data_dir / "answer_req_code.xml")

    # Keep only links whose target is a known Java artefact. This drops the
    # JSP-path targets and any links to code outside our collection, so the
    # gold set is consistent with the artefacts we actually evaluate against.
    gold_links = {
        (req, tgt)
        for (req, tgt) in all_links
        if req in requirements and tgt in code
    }

    return Dataset(requirements=requirements, code=code, gold_links=gold_links)
"""Skill URL importer — GitHub path parsing."""
import io
import zipfile

import pytest

from services.memory.skill_importer import (
    ResolvedSource,
    SkillImportError,
    _assert_github_url,
    _fetch_bytes,
    _list_github_dir,
    extract_skill_bundles_from_zip,
    list_repo_skill_dirs,
    list_repo_top_dirs,
    parse_skill_source,
)


def test_parse_github_blob_skill_md():
    src = parse_skill_source(
        "https://github.com/anthropics/skills/blob/main/skills/pdf/SKILL.md"
    )
    assert src.owner == "anthropics"
    assert src.repo == "skills"
    assert src.ref == "main"
    assert src.path.endswith("skills/pdf/SKILL.md")


def test_parse_github_tree_directory():
    src = parse_skill_source(
        "https://github.com/example/my-skills/tree/develop/caveman-skill"
    )
    assert src.owner == "example"
    assert src.repo == "my-skills"
    assert src.ref == "develop"
    assert src.path == "caveman-skill"


def test_parse_raw_github():
    src = parse_skill_source(
        "https://raw.githubusercontent.com/o/r/main/path/SKILL.md"
    )
    assert src.owner == "o"
    assert src.repo == "r"
    assert src.ref == "main"
    assert src.path == "path/SKILL.md"


def test_rejects_non_github():
    with pytest.raises(SkillImportError):
        parse_skill_source("https://example.com/skill.md")


def test_fetch_bytes_rejects_cross_host_redirect(monkeypatch):
    class _Resp:
        url = "https://evil.example/secret"
        status_code = 200
        content = b"x"

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr("services.memory.skill_importer.httpx.Client", _Client)
    monkeypatch.setattr(
        "services.memory.skill_importer.check_outbound_url",
        lambda url: (True, ""),
    )
    with pytest.raises(SkillImportError, match="redirect target"):
        _fetch_bytes("https://raw.githubusercontent.com/o/r/main/SKILL.md")


def test_assert_github_url_allows_api_host():
    _assert_github_url(
        "https://api.github.com/repos/o/r/contents?ref=main",
        context="redirect target",
    )


def test_list_github_dir_accepts_api_github_response(monkeypatch):
    monkeypatch.setattr(
        "services.memory.skill_importer._fetch_text",
        lambda url: "# skill\n",
    )
    monkeypatch.setattr(
        "services.memory.skill_importer.check_outbound_url",
        lambda url: (True, ""),
    )

    class _Resp:
        url = "https://api.github.com/repos/o/r/contents?ref=main"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{
                "name": "SKILL.md",
                "type": "file",
                "download_url": "https://raw.githubusercontent.com/o/r/main/SKILL.md",
            }]

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr("services.memory.skill_importer.httpx.Client", _Client)

    out = {}
    src = ResolvedSource(owner="o", repo="r", ref="main", path="")
    _list_github_dir(src, "", out)
    assert "SKILL.md" in out


def _mock_httpx_client(monkeypatch, response):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return response

    monkeypatch.setattr("services.memory.skill_importer.httpx.Client", _Client)
    monkeypatch.setattr(
        "services.memory.skill_importer.check_outbound_url",
        lambda url: (True, ""),
    )


def test_list_github_dir_surfaces_rate_limit(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/contents?ref=main"
        status_code = 403

        def json(self):
            return {"message": "API rate limit exceeded for 203.0.113.1"}

    _mock_httpx_client(monkeypatch, _Resp())
    src = ResolvedSource(owner="o", repo="r", ref="main", path="")
    with pytest.raises(SkillImportError, match="rate limit"):
        _list_github_dir(src, "", {})


def test_fetch_bytes_surfaces_github_error_detail(monkeypatch):
    class _Resp:
        url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        status_code = 403
        content = b""

        def json(self):
            return {"message": "Forbidden"}

    _mock_httpx_client(monkeypatch, _Resp())
    with pytest.raises(SkillImportError, match="GitHub request failed \\(403\\): Forbidden"):
        _fetch_bytes("https://raw.githubusercontent.com/o/r/main/SKILL.md")


def test_list_repo_top_dirs_returns_only_directories(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/contents?ref=main"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"name": "theme-factory", "type": "dir"},
                {"name": "document-skills", "type": "dir"},
                {"name": "README.md", "type": "file"},
                {"name": ".github", "type": "dir"},
            ]

    _mock_httpx_client(monkeypatch, _Resp())
    src, dirs = list_repo_top_dirs("https://github.com/ComposioHQ/awesome-claude-skills")
    assert src.owner == "ComposioHQ"
    assert src.repo == "awesome-claude-skills"
    assert dirs == ["theme-factory", "document-skills", ".github"]


def test_list_repo_top_dirs_rejects_non_directory_listing(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/contents/SKILL.md?ref=main"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"name": "SKILL.md", "type": "file"}

    _mock_httpx_client(monkeypatch, _Resp())
    with pytest.raises(SkillImportError, match="expected a directory"):
        list_repo_top_dirs("https://github.com/o/r/blob/main/SKILL.md")


def test_list_repo_skill_dirs_flat_layout(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/git/trees/main?recursive=1"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "sha": "abc", "truncated": False,
                "tree": [
                    {"path": "README.md", "type": "blob"},
                    {"path": "theme-factory", "type": "tree"},
                    {"path": "theme-factory/SKILL.md", "type": "blob"},
                    {"path": "theme-factory/scripts/render.py", "type": "blob"},
                    {"path": "document-skills/SKILL.md", "type": "blob"},
                ],
            }

    _mock_httpx_client(monkeypatch, _Resp())
    src, dirs = list_repo_skill_dirs("https://github.com/ComposioHQ/awesome-claude-skills")
    assert src.owner == "ComposioHQ"
    assert dirs == ["theme-factory", "document-skills"]


def test_list_repo_skill_dirs_deeply_nested_layout(monkeypatch):
    """alirezarezvani/claude-skills shape: SKILL.md several levels deep
    under category/plugin/skills/<name>/SKILL.md, not one-per-top-folder."""
    class _Resp:
        url = "https://api.github.com/repos/o/r/git/trees/main?recursive=1"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "sha": "abc", "truncated": False,
                "tree": [
                    {"path": "engineering", "type": "tree"},
                    {"path": "engineering/agenthub", "type": "tree"},
                    {"path": "engineering/agenthub/skills/board/SKILL.md", "type": "blob"},
                    {"path": "marketing-skill/skills/aeo/SKILL.md", "type": "blob"},
                    {"path": "README.md", "type": "blob"},
                ],
            }

    _mock_httpx_client(monkeypatch, _Resp())
    src, dirs = list_repo_skill_dirs("https://github.com/alirezarezvani/claude-skills")
    assert dirs == ["engineering/agenthub/skills/board", "marketing-skill/skills/aeo"]


def test_list_repo_skill_dirs_scopes_to_subpath(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/git/trees/main?recursive=1"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "sha": "abc", "truncated": False,
                "tree": [
                    {"path": "keep/skill-a/SKILL.md", "type": "blob"},
                    {"path": "other/skill-b/SKILL.md", "type": "blob"},
                ],
            }

    _mock_httpx_client(monkeypatch, _Resp())
    src, dirs = list_repo_skill_dirs("https://github.com/o/r/tree/main/keep")
    assert dirs == ["keep/skill-a"]


def test_list_repo_skill_dirs_rejects_non_tree_response(monkeypatch):
    class _Resp:
        url = "https://api.github.com/repos/o/r/git/trees/main?recursive=1"
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "Not Found"}

    _mock_httpx_client(monkeypatch, _Resp())
    with pytest.raises(SkillImportError, match="expected a repository tree"):
        list_repo_skill_dirs("https://github.com/o/r")


def _make_zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in entries.items():
            zf.writestr(path, content)
    return buf.getvalue()


def test_extract_skill_bundles_from_zip_finds_real_skills_and_siblings():
    data = _make_zip({
        "repo-main/marketing-skill/skills/aeo/SKILL.md": "---\nname: aeo\ndescription: x\n---\nbody",
        "repo-main/marketing-skill/skills/aeo/scripts/run.py": "print('hi')",
        "repo-main/engineering/skills/board/SKILL.md": "---\nname: board\ndescription: y\n---\nbody",
        "repo-main/README.md": "not a skill",
    })
    bundles = extract_skill_bundles_from_zip(data)
    labels = sorted(label for label, _ in bundles)
    assert labels == ["engineering/skills/board", "marketing-skill/skills/aeo"]

    aeo_files = dict(bundles)["marketing-skill/skills/aeo"]
    assert "SKILL.md" in aeo_files
    assert "scripts/run.py" in aeo_files
    assert aeo_files["SKILL.md"].startswith("---")


def test_extract_skill_bundles_from_zip_skips_dotdir_mirrors_and_non_frontmatter():
    data = _make_zip({
        # Real skill.
        "repo-main/marketing-skill/skills/aeo/SKILL.md": "---\nname: aeo\n---\nbody",
        # Broken-symlink-style mirror under a dot-prefixed tool dir — must be skipped.
        "repo-main/.gemini/skills/aeo/SKILL.md": "../../../marketing-skill/skills/aeo/SKILL.md",
        # A SKILL.md-named fixture with no real frontmatter — must be skipped.
        "repo-main/engineering/skills/skill-tester/assets/sample-skill/SKILL.md": "# Sample Text Processor\n\nNot real frontmatter",
    })
    bundles = extract_skill_bundles_from_zip(data)
    labels = [label for label, _ in bundles]
    assert labels == ["marketing-skill/skills/aeo"]


def test_extract_skill_bundles_from_zip_respects_max_skills():
    entries = {}
    for i in range(5):
        entries[f"repo-main/cat/skills/s{i}/SKILL.md"] = f"---\nname: s{i}\n---\nbody"
    data = _make_zip(entries)
    bundles = extract_skill_bundles_from_zip(data, max_skills=2)
    assert len(bundles) == 2


def test_extract_skill_bundles_from_zip_rejects_bad_zip():
    with pytest.raises(SkillImportError, match="not a valid .zip"):
        extract_skill_bundles_from_zip(b"not a zip file")

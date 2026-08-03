from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_verifier() -> ModuleType:
    path = ROOT / "scripts" / "verify_wandb_release_download.py"
    spec = importlib.util.spec_from_file_location("barun_wandb_download_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_records(files: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(files.items())
    }


def artifact_record(
    *, name: str, artifact_type: str, source_root: str, files: dict[str, bytes]
) -> dict[str, Any]:
    return {
        "file_count": len(files),
        "files": file_records(files),
        "name": name,
        "source_root": source_root,
        "total_bytes": sum(len(payload) for payload in files.values()),
        "type": artifact_type,
    }


def release_records(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, dict[str, bytes]]]:
    repository = tmp_path / "repo"
    records_root = repository / "release-records"
    records_root.mkdir(parents=True)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    download_root = external_parent / "fresh-download"
    verification_path = records_root / "wandb-redownload-verification.json"

    float_files = {
        "LICENSE": b"license\n",
        "MODEL_CARD.md": b"# BarunAction-35M float\n",
        "NOTICE": b"notice\n",
        "barun_config.json": b"{}\n",
        "checkpoint_manifest.json": b'{"schema_version":"fixture"}\n',
        "model.safetensors": b"float checkpoint bytes\n",
        "tokenizer.json": b'{"version":"fixture"}\n',
    }
    int8_files = {
        "LICENSE": b"license\n",
        "MODEL_CARD.md": b"# BarunAction-35M int8\n",
        "NOTICE": b"notice\n",
        "barun_config.json": b"{}\n",
        "model.int8.pt": b"int8 checkpoint bytes\n",
        "quantization_manifest.json": b'{"schema_version":"fixture"}\n',
        "tokenizer.json": b'{"version":"fixture"}\n',
    }
    evidence_files = {
        "mobile/result.json": b'{"ast_exact":0.7962962962962963}\n',
        "presto/result.json": b'{"ast_exact":0.7432810750279956}\n',
    }
    retention_files = {
        "artifact-sha256.json": b'{"schema_version":"fixture"}\n',
        "paired-samples.jsonl": b'{"sample_id":"fixture"}\n',
        "protocol.json": b'{"protocol_id":"fixture"}\n',
        "result.json": b'{"gate":{"passed":true}}\n',
    }
    manifest = {
        "artifacts": {
            "float": artifact_record(
                name="barunaction-35m-candidate-v2-float",
                artifact_type="model",
                source_root="release-artifacts/float",
                files=float_files,
            ),
            "int8_darwin_arm64_qnnpack": artifact_record(
                name="barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
                artifact_type="model",
                source_root="release-artifacts/int8",
                files=int8_files,
            ),
            "evidence": artifact_record(
                name="barunaction-35m-candidate-v2-evidence",
                artifact_type="evaluation",
                source_root="release-artifacts/evidence",
                files=evidence_files,
            ),
        },
        "int8_retention": {
            "disposition": "retained-release",
            "file_count": len(retention_files),
            "files": file_records(retention_files),
            "gate": {"passed": True},
            "source_root": "release-artifacts/int8-retention",
            "total_bytes": sum(len(payload) for payload in retention_files.values()),
        },
        "local_manifest_authoritative": True,
        "project": "barunaction-35m",
        "run": {},
        "schema_version": "barunaction-wandb-release-manifest-v1",
        "security": {
            "credential_markers_found": 0,
            "symlinks_found": 0,
            "unlisted_source_entries_found": 0,
        },
        "spec": {},
        "summary_metrics": {},
        "wandb_plan": {
            "automatic_code_capture": False,
            "automatic_git_capture": False,
            "evidence_includes_local_manifest_as": "release-manifest.json",
            "external_wandb_dir_required": True,
            "package_version": "0.28.1",
            "upload_requires_explicit_opt_in": True,
        },
    }
    manifest_path = records_root / "release-manifest.json"
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    identities = {
        "float": (
            "barunaction-35m-candidate-v2-float",
            "model",
            "v3",
            "digest-float",
        ),
        "int8_darwin_arm64_qnnpack": (
            "barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack",
            "model",
            "v4",
            "digest-int8",
        ),
        "evidence": (
            "barunaction-35m-candidate-v2-evidence",
            "evaluation",
            "v5",
            "digest-evidence",
        ),
    }
    receipt_artifacts = {}
    remote_files: dict[str, dict[str, bytes]] = {
        "float": float_files,
        "int8_darwin_arm64_qnnpack": int8_files,
        "evidence": {
            **evidence_files,
            **{f"int8-retention/{name}": payload for name, payload in retention_files.items()},
            "release-manifest.json": manifest_bytes,
        },
    }
    for kind, (name, _, version, digest) in identities.items():
        receipt_artifacts[kind] = {
            "digest": digest,
            "name": name,
            "qualified_name": f"test-entity/barunaction-35m/{name}:{version}",
            "version": version,
        }
    upload_receipt = {
        "artifacts": receipt_artifacts,
        "entity": "test-entity",
        "local_manifest": {
            "path": manifest_path.relative_to(repository).as_posix(),
            "sha256": sha256_bytes(manifest_bytes),
        },
        "project": "barunaction-35m",
        "run_id": "release-run-17",
        "schema_version": "barunaction-wandb-release-receipt-v1",
    }
    upload_receipt_path = records_root / "wandb-receipt.json"
    upload_receipt_path.write_text(
        json.dumps(upload_receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return (
        repository,
        manifest_path,
        upload_receipt_path,
        download_root,
        verification_path,
        remote_files,
    )


class FakeArtifact:
    def __init__(
        self,
        *,
        qualified_name: str,
        digest: str,
        files: dict[str, bytes],
        mutate: Callable[[Path], None] | None = None,
    ) -> None:
        self.qualified_name = qualified_name
        self.digest = digest
        self.files = files
        self.mutate = mutate

    def download(self, *, root: str) -> str:
        destination = Path(root)
        assert not destination.exists()
        destination.mkdir()
        for relative, payload in self.files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if self.mutate is not None:
            self.mutate(destination)
        return str(destination)


class FakeApi:
    def __init__(self, artifacts: dict[str, FakeArtifact]) -> None:
        self.artifacts = artifacts
        self.calls: list[tuple[str, str]] = []

    def artifact(self, qualified_name: str, *, type: str) -> FakeArtifact:
        self.calls.append((qualified_name, type))
        return self.artifacts[qualified_name]


def fake_wandb(
    module: ModuleType,
    remote_files: dict[str, dict[str, bytes]],
    *,
    version: str = "0.28.1",
    mutations: dict[str, Callable[[Path], None]] | None = None,
    digest_override: dict[str, str] | None = None,
) -> tuple[Any, FakeApi]:
    mutations = mutations or {}
    digest_override = digest_override or {}
    qualified = {
        "float": "test-entity/barunaction-35m/barunaction-35m-candidate-v2-float:v3",
        "int8_darwin_arm64_qnnpack": (
            "test-entity/barunaction-35m/barunaction-35m-candidate-v2-int8-darwin-arm64-qnnpack:v4"
        ),
        "evidence": "test-entity/barunaction-35m/barunaction-35m-candidate-v2-evidence:v5",
    }
    digests = {
        "float": "digest-float",
        "int8_darwin_arm64_qnnpack": "digest-int8",
        "evidence": "digest-evidence",
    }
    api = FakeApi(
        {
            qualified[kind]: FakeArtifact(
                qualified_name=qualified[kind],
                digest=digest_override.get(kind, digests[kind]),
                files=files,
                mutate=mutations.get(kind),
            )
            for kind, files in remote_files.items()
        }
    )
    return SimpleNamespace(__version__=version, Api=lambda: api), api


def execute_fixture(
    module: ModuleType,
    fixture: tuple[Path, Path, Path, Path, Path, dict[str, dict[str, bytes]]],
    *,
    download: bool,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    repository, manifest, receipt, download_root, verification, _ = fixture
    return module.execute(
        repository_root=repository,
        manifest_path=manifest,
        wandb_receipt_path=receipt,
        download_root=download_root,
        verification_receipt_path=verification,
        download=download,
        wandb_module=wandb_module,
    )


def test_dry_validation_never_imports_wandb_or_writes(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_verifier()
    fixture = release_records(tmp_path)
    repository, _, _, download_root, verification, _ = fixture

    def forbidden_import(name: str) -> None:
        raise AssertionError(f"dry validation imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    result = execute_fixture(module, fixture, download=False)

    assert result["mode"] == "dry-run"
    assert len(result["artifact_qualified_names"]) == 3
    assert result["download_root"] == str(download_root)
    assert not download_root.exists()
    assert not verification.exists()
    assert repository.is_dir()


def test_explicit_download_verifies_all_bytes_and_writes_one_immutable_receipt(
    tmp_path: Path,
) -> None:
    module = load_verifier()
    fixture = release_records(tmp_path)
    repository, manifest, upload_receipt, download_root, verification, remote_files = fixture
    wandb, api = fake_wandb(module, remote_files)

    result = execute_fixture(module, fixture, download=True, wandb_module=wandb)

    assert result["mode"] == "download"
    assert result["verification_receipt"] == verification.relative_to(repository).as_posix()
    verified = json.loads(verification.read_text())
    assert verified["package_version"] == "0.28.1"
    assert verified["schema_version"] == "barunaction-wandb-download-verification-v1"
    assert set(verified["artifacts"]) == set(module.ARTIFACT_CONTRACTS)
    assert (
        verified["sources"]["release_manifest"]["sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert (
        verified["sources"]["wandb_upload_receipt"]["sha256"]
        == hashlib.sha256(upload_receipt.read_bytes()).hexdigest()
    )
    assert set(verified["artifacts"]["evidence"]["files"]) == set(remote_files["evidence"])
    assert {call[0] for call in api.calls} == {
        record["qualified_name"]
        for record in json.loads(upload_receipt.read_text())["artifacts"].values()
    }
    assert all(call[0].endswith((":v3", ":v4", ":v5")) for call in api.calls)
    assert all("latest" not in call[0] for call in api.calls)
    assert sorted(path.name for path in download_root.iterdir()) == sorted(
        module.ARTIFACT_CONTRACTS
    )
    with pytest.raises(module.ReleaseVerificationError, match="fresh and absent"):
        execute_fixture(module, fixture, download=True, wandb_module=wandb)


def test_mutable_alias_and_digest_mismatch_fail_before_receipt(tmp_path: Path) -> None:
    module = load_verifier()
    fixture = release_records(tmp_path)
    _, _, upload_receipt, download_root, verification, remote_files = fixture
    receipt = json.loads(upload_receipt.read_text())
    receipt["artifacts"]["float"]["version"] = "latest"
    receipt["artifacts"]["float"]["qualified_name"] = (
        "test-entity/barunaction-35m/barunaction-35m-candidate-v2-float:latest"
    )
    upload_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(module.ReleaseVerificationError, match="invalid value"):
        execute_fixture(module, fixture, download=False)
    assert not download_root.exists()
    assert not verification.exists()

    fixture = release_records(tmp_path / "digest")
    _, _, _, _, verification, remote_files = fixture
    wandb, _ = fake_wandb(module, remote_files, digest_override={"float": "wrong-digest"})
    with pytest.raises(module.ReleaseVerificationError, match="digest differs"):
        execute_fixture(module, fixture, download=True, wandb_module=wandb)
    assert not verification.exists()


@pytest.mark.parametrize("failure", ["extra", "token", "symlink", "hash"])
def test_download_rejects_extra_symlink_credential_and_changed_bytes(
    tmp_path: Path, failure: str
) -> None:
    module = load_verifier()
    fixture = release_records(tmp_path)
    _, _, _, _, verification, remote_files = fixture

    def mutate(destination: Path) -> None:
        if failure == "extra":
            (destination / "unlisted.txt").write_text("extra\n")
        elif failure == "token":
            (destination / "mobile/result.json").write_text("wandb_v1_" + "X" * 32)
        elif failure == "symlink":
            target = destination / "outside.txt"
            target.write_text("outside\n")
            (destination / "mobile/result.json").unlink()
            (destination / "mobile/result.json").symlink_to(target)
        else:
            (destination / "mobile/result.json").write_text("changed\n")

    wandb, _ = fake_wandb(module, remote_files, mutations={"evidence": mutate})
    expected = {
        "extra": "file set differs",
        "token": "credential token marker",
        "symlink": "symlink",
        "hash": "differs from the manifest",
    }[failure]
    with pytest.raises(module.ReleaseVerificationError, match=expected):
        execute_fixture(module, fixture, download=True, wandb_module=wandb)
    assert not verification.exists()


def test_wrong_wandb_version_and_preexisting_destinations_fail_closed(tmp_path: Path) -> None:
    module = load_verifier()
    fixture = release_records(tmp_path)
    _, _, _, download_root, verification, remote_files = fixture
    wrong_wandb, _ = fake_wandb(module, remote_files, version="0.28.2")
    with pytest.raises(module.ReleaseVerificationError, match="exactly wandb==0.28.1"):
        execute_fixture(module, fixture, download=True, wandb_module=wrong_wandb)
    assert not download_root.exists()
    assert not verification.exists()

    fixture = release_records(tmp_path / "existing")
    _, _, _, download_root, _, _ = fixture
    download_root.mkdir()
    with pytest.raises(module.ReleaseVerificationError, match="fresh and absent"):
        execute_fixture(module, fixture, download=False)

    fixture = release_records(tmp_path / "receipt")
    _, _, _, _, verification, _ = fixture
    verification.write_text("do not overwrite\n")
    with pytest.raises(module.ReleaseVerificationError, match="refusing to overwrite"):
        execute_fixture(module, fixture, download=False)
    assert verification.read_text() == "do not overwrite\n"


def test_verifier_has_no_login_or_credential_cli_surface(capsys: Any) -> None:
    module = load_verifier()
    source = (ROOT / "scripts" / "verify_wandb_release_download.py").read_text()
    tree = ast.parse(source)
    assert not [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "login"
    ]
    assert not [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.Import) and any(alias.name == "wandb" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "wandb")
        )
    ]
    option_strings = {
        option for action in module.build_parser()._actions for option in action.option_strings
    }
    assert not any(
        "key" in option or "token" in option or "password" in option for option in option_strings
    )
    marker = "wandb_v1_" + "Z" * 32
    assert module.main(["--api-key", marker]) == 2
    captured = capsys.readouterr()
    assert "credential-like CLI argument" in captured.err
    assert marker not in captured.err
    assert "wandb==0.28.1" in (ROOT / "requirements/wandb-release.txt").read_text()

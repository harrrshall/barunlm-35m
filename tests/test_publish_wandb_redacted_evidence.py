from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_publisher() -> ModuleType:
    path = ROOT / "scripts" / "publish_wandb_redacted_evidence.py"
    spec = importlib.util.spec_from_file_location("barun_redacted_evidence_publisher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLISHER = _load_publisher()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_sha256(files: dict[str, dict[str, Any]]) -> str:
    payload = "".join(
        f"{record['output_sha256']}  {record['output_bytes']}  {relative}\n"
        for relative, record in sorted(files.items())
    ).encode()
    return _sha256(payload)


def _fixture(tmp_path: Path, *, override: dict[str, bytes] | None = None) -> dict[str, Any]:
    repository = tmp_path / "repo"
    records = repository / "records"
    records.mkdir(parents=True)
    external = tmp_path / "external"
    source = external / "redacted-view"
    source.mkdir(parents=True)
    derived = {
        "metrics.json": b'{"ast_exact":602,"denominator":756}\n',
        "nested/predictions.jsonl": b'{"id":"sample-1","prediction":"ABSTAIN"}\n',
    }
    if override:
        derived.update(override)
    file_records: dict[str, dict[str, Any]] = {}
    for relative, payload in derived.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = _sha256(payload)
        file_records[relative] = {
            "changed": False,
            "output_bytes": len(payload),
            "output_sha256": digest,
            "redactions": [],
            "source_bytes": len(payload),
            "source_sha256": digest,
        }
    tree_hash = _tree_sha256(file_records)
    receipt = {
        "classification": {
            "authoritative_evidence": False,
            "distribution_view": "redacted",
            "source_public_artifact_mutated": False,
            "warning": "fixture nonauthoritative view",
        },
        "files": file_records,
        "output_tree": {
            "derived_file_count": len(file_records),
            "receipt_excluded_from_self_hash": "redaction-receipt.json",
            "sha256": tree_hash,
        },
        "policy": {},
        "redactions": {
            "changed_files": 0,
            "distinct_local_paths": 0,
            "occurrences_by_kind": {},
            "total_occurrences": 0,
        },
        "schema_version": "barunaction-wandb-redacted-evidence-view-v1",
        "security": {
            "credential_markers_found": 0,
            "network_operations_performed": False,
            "source_manifest_hash_verified": True,
            "source_symlinks_found": 0,
            "source_unlisted_files_found": 0,
            "uploads_performed": False,
        },
        "semantic_preservation": {},
        "source": {
            "artifact": {
                "name": "barunaction-35m-candidate-v2-evidence",
                "public_file_count": len(file_records),
                "public_total_bytes": 135_926_068,
                "source_root": "fixture/evidence",
                "staging_file_count": len(file_records),
                "staging_total_bytes": sum(len(payload) for payload in derived.values()),
                "type": "evaluation",
                "version": "v0",
                "wandb_digest": "b5e2cb35698e36bc1a97095c6c2d8565",
            },
            "release_manifest": {
                "bytes": 1,
                "path": "fixture/release-manifest.json",
                "sha256": ("cbb29c4921855031bfeee1c1f5e9ed1a33a932b902a875c21f255fac793c2165"),
            },
        },
    }
    receipt_payload = (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (source / "redaction-receipt.json").write_bytes(receipt_payload)
    return {
        "repository": repository,
        "records": records,
        "external": external,
        "source": source,
        "receipt_hash": _sha256(receipt_payload),
        "tree_hash": tree_hash,
        "derived_count": len(file_records),
        "upload_count": len(file_records) + 1,
        "derived": derived,
    }


def _execute(
    fixture: dict[str, Any],
    *,
    receipt_name: str = "publication.json",
    wandb_name: str = "wandb-runtime",
    upload: bool = False,
    download_verify: bool = False,
    publication_receipt: Path | None = None,
    download_destination: Path | None = None,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    return PUBLISHER._execute_with_contract(
        repository_root=fixture["repository"],
        source_root=fixture["source"],
        receipt_path=fixture["records"] / receipt_name,
        wandb_dir=fixture["external"] / wandb_name,
        entity="test-entity",
        upload=upload,
        download_verify=download_verify,
        publication_receipt_path=publication_receipt,
        download_destination=download_destination,
        wandb_module=wandb_module,
        expected_receipt_sha256=fixture["receipt_hash"],
        expected_tree_sha256=fixture["tree_hash"],
        expected_derived_count=fixture["derived_count"],
        expected_upload_count=fixture["upload_count"],
    )


def test_public_upload_entrypoint_is_terminally_closed(tmp_path: Path) -> None:
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="is closed"):
        PUBLISHER.execute(
            repository_root=tmp_path,
            source_root=tmp_path / "missing-source",
            receipt_path=tmp_path / "missing-receipt.json",
            wandb_dir=tmp_path / "missing-wandb",
            entity="test-entity",
            upload=True,
            download_verify=False,
        )


class FakePendingArtifact:
    def __init__(self, owner: FakeWandb, *, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.owner = owner
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.version = ""
        self.digest = ""
        self.qualified_name = ""
        self.state = ""

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))

    def wait(self) -> FakePendingArtifact:
        return self


class FakeRemoteArtifact:
    def __init__(self, owner: FakeWandb) -> None:
        self.owner = owner
        self.name = "barunaction-35m-candidate-v2-evidence"
        self.type = "evaluation"
        self.version = owner.api_version_override or "v1"
        self.digest = owner.api_digest_override or owner.committed_digest
        self.qualified_name = f"test-entity/barunaction-35m/{self.name}:{self.version}"
        self.state = owner.api_state_override or "COMMITTED"
        assert owner.run is not None and owner.run.logged
        self.metadata = owner.run.logged[0].metadata

    def download(self, *, root: str) -> str:
        destination = Path(root)
        assert destination.is_dir()
        for relative, payload in self.owner.remote_files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if self.owner.download_mutation is not None:
            self.owner.download_mutation(destination)
        return str(destination)


class FakeSourceArtifact:
    def __init__(self, owner: FakeWandb) -> None:
        self.name = owner.latest_name
        self.type = owner.latest_type
        self.version = owner.latest_version
        self.digest = owner.latest_digest
        self.qualified_name = owner.latest_qualified_name_override or (
            f"test-entity/barunaction-35m/{self.name}:{self.version}"
        )
        self.state = owner.latest_state


class FakeApi:
    def __init__(self, owner: FakeWandb) -> None:
        self.owner = owner

    def artifact(
        self, qualified_name: str, *, type: str
    ) -> FakeRemoteArtifact | FakeSourceArtifact | None:
        self.owner.api_calls.append((qualified_name, type))
        if qualified_name.endswith(":latest"):
            if self.owner.latest_absent:
                return None
            return FakeSourceArtifact(self.owner)
        return FakeRemoteArtifact(self.owner)


class FakeRun:
    def __init__(self, owner: FakeWandb, *, entity: str, project: str) -> None:
        self.owner = owner
        self.entity = entity
        self.project = project
        self.id = "redacted-release-test-run"
        self.finish_codes: list[int] = []
        self.logged: list[FakePendingArtifact] = []

    def log_artifact(self, artifact: FakePendingArtifact) -> FakePendingArtifact:
        self.owner.log_calls += 1
        artifact.version = self.owner.committed_version
        artifact.digest = self.owner.committed_digest
        artifact.qualified_name = f"{self.entity}/{self.project}/{artifact.name}:{artifact.version}"
        artifact.state = self.owner.committed_state
        self.owner.remote_files = {
            relative: Path(path).read_bytes() for path, relative in artifact.files
        }
        self.logged.append(artifact)
        return artifact

    def finish(self, *, exit_code: int) -> None:
        self.finish_codes.append(exit_code)


class FakeWandb:
    def __init__(self, *, version: str = "0.28.1") -> None:
        self.__version__ = version
        self.committed_version = "v1"
        self.committed_digest = "fixture-wandb-digest-v1"
        self.committed_state = "COMMITTED"
        self.latest_version = "v0"
        self.latest_digest = "b5e2cb35698e36bc1a97095c6c2d8565"
        self.latest_state = "COMMITTED"
        self.latest_name = "barunaction-35m-candidate-v2-evidence"
        self.latest_type = "evaluation"
        self.latest_qualified_name_override: str | None = None
        self.latest_absent = False
        self.api_version_override: str | None = None
        self.api_digest_override: str | None = None
        self.api_state_override: str | None = None
        self.download_mutation: Callable[[Path], None] | None = None
        self.remote_files: dict[str, bytes] = {}
        self.api_calls: list[tuple[str, str]] = []
        self.init_calls: list[dict[str, Any]] = []
        self.log_calls = 0
        self.run: FakeRun | None = None

    def Settings(self, **kwargs: Any) -> Any:
        return SimpleNamespace(kwargs=kwargs)

    def Artifact(self, *, name: str, type: str, metadata: dict[str, Any]) -> FakePendingArtifact:
        return FakePendingArtifact(self, name=name, type=type, metadata=metadata)

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_calls.append(kwargs)
        self.run = FakeRun(self, entity=kwargs["entity"], project=kwargs["project"])
        return self.run

    def Api(self) -> FakeApi:
        return FakeApi(self)


def test_dry_run_validates_every_byte_without_import_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    source_hashes = {
        str(path.relative_to(fixture["source"])): _sha256(path.read_bytes())
        for path in fixture["source"].rglob("*")
        if path.is_file()
    }

    def forbidden_import(name: str) -> None:
        raise AssertionError(f"dry-run imported {name}")

    monkeypatch.setattr(PUBLISHER.importlib, "import_module", forbidden_import)
    result = _execute(fixture)

    assert result == {
        "proposed_artifact_qualified_name": (
            "test-entity/barunaction-35m/barunaction-35m-candidate-v2-evidence:v1"
        ),
        "mode": "dry-run",
        "receipt": "records/publication.json",
        "source_receipt_sha256": fixture["receipt_hash"],
        "upload_file_count": fixture["upload_count"],
    }
    assert not (fixture["records"] / "publication.json").exists()
    assert not (fixture["external"] / "wandb-runtime").exists()
    assert source_hashes == {
        str(path.relative_to(fixture["source"])): _sha256(path.read_bytes())
        for path in fixture["source"].rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("failure", ["drift", "extra", "symlink", "credential"])
def test_source_drift_extra_symlink_and_credential_fail_before_any_write(
    tmp_path: Path, failure: str
) -> None:
    override = None
    if failure == "credential":
        marker = b"wandb" + b"_v1_" + b"X" * 32
        override = {"metrics.json": b'{"value":"' + marker + b'"}\n'}
    fixture = _fixture(tmp_path, override=override)
    if failure == "drift":
        (fixture["source"] / "metrics.json").write_text("{}\n")
    elif failure == "extra":
        (fixture["source"] / "extra.json").write_text("{}\n")
    elif failure == "symlink":
        target = fixture["source"] / "metrics.json"
        link = fixture["source"] / "nested/predictions.jsonl"
        link.unlink()
        link.symlink_to(target)

    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError):
        _execute(fixture)
    assert not (fixture["records"] / "publication.json").exists()
    assert not (fixture["external"] / "wandb-runtime").exists()


@pytest.mark.parametrize("operation", ["upload", "download"])
def test_every_remote_executor_boundary_is_terminally_closed(
    tmp_path: Path, operation: str
) -> None:
    fixture = _fixture(tmp_path)
    wandb = FakeWandb()
    kwargs: dict[str, Any] = {"wandb_module": wandb}
    if operation == "upload":
        kwargs["upload"] = True
    else:
        kwargs.update(
            {
                "download_verify": True,
                "publication_receipt": fixture["records"] / "absent-publication.json",
                "download_destination": fixture["external"] / "absent-download",
            }
        )

    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="is closed"):
        _execute(fixture, **kwargs)

    assert wandb.api_calls == []
    assert wandb.init_calls == []
    assert wandb.log_calls == 0
    assert wandb.run is None
    assert wandb.remote_files == {}
    assert not (fixture["records"] / "publication.json").exists()


def test_lowest_remote_helpers_contain_no_mutation_or_download_implementation() -> None:
    with pytest.raises(
        PUBLISHER.RedactedEvidencePublicationError, match="implementation was removed"
    ):
        PUBLISHER._upload(
            wandb_module=None,
            view=None,
            entity="unused",
            repository_root=None,
            receipt_path=None,
            receipt_relative="unused",
            wandb_dir=None,
            validation_contract={},
        )
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="no remote v1"):
        PUBLISHER._download_verify(
            wandb_module=None,
            view=None,
            identity={},
            publication_receipt_relative="unused",
            publication_receipt_sha256="unused",
            entity="unused",
            repository_root=None,
            receipt_path=None,
            receipt_relative="unused",
            wandb_dir=None,
            download_root=None,
            validation_contract={},
        )

    tree = ast.parse((ROOT / "scripts/publish_wandb_redacted_evidence.py").read_text())
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "init" not in attributes
    assert "log_artifact" not in attributes
    assert "download" not in attributes


def test_overwrite_and_runtime_destination_refusals_are_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture["records"] / "publication.json"
    receipt.write_text("do not overwrite\n")
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="overwrite"):
        _execute(fixture)
    assert receipt.read_text() == "do not overwrite\n"

    fixture = _fixture(tmp_path / "wandb")
    (fixture["external"] / "wandb-runtime").mkdir()
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="fresh and absent"):
        _execute(fixture)

    fixture = _fixture(tmp_path / "download")
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="is closed"):
        _execute(fixture, download_verify=True)


def test_source_must_be_outside_repository(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(PUBLISHER.RedactedEvidencePublicationError, match="outside"):
        PUBLISHER._execute_with_contract(
            repository_root=fixture["repository"],
            source_root=fixture["repository"],
            receipt_path=fixture["records"] / "publication.json",
            wandb_dir=fixture["external"] / "wandb-runtime",
            entity="test-entity",
            upload=False,
            download_verify=False,
            expected_receipt_sha256=fixture["receipt_hash"],
            expected_tree_sha256=fixture["tree_hash"],
            expected_derived_count=fixture["derived_count"],
            expected_upload_count=fixture["upload_count"],
        )


def test_cli_has_no_credentials_login_or_implicit_network_surface(capsys: Any) -> None:
    source = (ROOT / "scripts" / "publish_wandb_redacted_evidence.py").read_text()
    tree = ast.parse(source)
    assert not [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "login"
    ]
    assert not [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            isinstance(node, ast.Import)
            and any(alias.name == "wandb" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module == "wandb"
        )
    ]
    options = {
        option for action in PUBLISHER.build_parser()._actions for option in action.option_strings
    }
    assert {"--upload", "--download-verify"} <= options
    assert "--download" not in options
    assert not any(
        word in option for option in options for word in ("api-key", "password", "secret", "token")
    )
    marker = "wandb" + "_v1_" + "Z" * 32
    assert PUBLISHER.main(["--api-key", marker]) == 2
    captured = capsys.readouterr()
    assert "credential-like CLI argument" in captured.err
    assert marker not in captured.err
    assert "wandb==0.28.1" in (ROOT / "requirements/wandb-release.txt").read_text()

    builder = ast.parse((ROOT / "scripts/build_wandb_redacted_evidence.py").read_text())
    builder_imports = {
        alias.name.split(".")[0]
        for node in ast.walk(builder)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert builder_imports.isdisjoint(
        {"httpx", "requests", "socket", "subprocess", "urllib", "wandb"}
    )

"""The Docker deliverable: entry point, healthcheck, compose file.

None of this is exercised by the application's own tests, and all of it is
what a new user meets first. The bugs these cover were both real: a documented
entry-point command that had been dead since Phase 3, and a generator
container that reported itself unhealthy for its entire life.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "deploy" / "entrypoint.sh"
HEALTHCHECK = ROOT / "scripts" / "healthcheck.sh"
COMPOSE = ROOT / "deploy" / "docker-compose.yml"
DOCKERFILE = ROOT / "deploy" / "Dockerfile"


def _documented_commands() -> set[str]:
    """The commands listed in the entry point's header comment."""
    names = set()
    for line in ENTRYPOINT.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        match = re.match(r"#\s{3,}([a-z]+)\s{2,}\S", line)
        if match:
            names.add(match.group(1))
    return names


def _handled_commands() -> set[str]:
    """The labels of the entry point's case statement, minus the catch-all."""
    body = ENTRYPOINT.read_text(encoding="utf-8")
    body = body[body.index('case "$command" in') :]
    return {m.group(1) for m in re.finditer(r"^  ([a-z]+)\)$", body, re.MULTILINE)}


def test_every_documented_command_is_handled():
    """That header is the only documentation these commands have, and it drifted:
    `streamer` was still advertised as "arrives in Phase 3" long after Phase 3
    shipped, and running it printed that and exited 64."""
    documented = _documented_commands()
    assert documented, "the header comment stopped listing commands"
    missing = documented - _handled_commands()
    assert not missing, f"documented but not handled: {sorted(missing)}"


def test_no_command_is_handled_without_being_documented():
    undocumented = _handled_commands() - _documented_commands()
    assert not undocumented, f"handled but not documented: {sorted(undocumented)}"


def test_the_entry_point_does_not_still_promise_future_phases():
    """Phase markers are build-time scaffolding. Shipping them tells a user
    that a finished feature has not been written yet."""
    body = ENTRYPOINT.read_text(encoding="utf-8")
    assert not re.search(r"Phase \d", body), "a phase marker survived into the entry point"


def test_the_entry_point_records_its_role_for_the_healthcheck():
    """The healthcheck is a separate `docker exec`: it inherits the image's
    environment, not anything the entry point exports. Without the marker file
    it cannot tell a generator container from a web one."""
    assert "/tmp/tad-role" in ENTRYPOINT.read_text(encoding="utf-8")


def test_the_healthcheck_knows_the_generator_has_no_web_port():
    """A generator container started from this image has no socket to answer
    on, so a healthcheck that only knows how to curl reports it unhealthy for
    ever — and an unhealthy container is what restart policies act on."""
    body = HEALTHCHECK.read_text(encoding="utf-8")
    assert "/tmp/tad-role" in body
    assert "generator)" in body
    assert "app.services.visual.worker" in body.replace("\\", "")


@pytest.mark.parametrize("script", [ENTRYPOINT, HEALTHCHECK])
def test_shipped_scripts_are_executable(script: Path):
    assert script.exists(), f"{script.name} is missing"
    assert script.stat().st_mode & stat.S_IXUSR, f"{script.name} is not executable"


def test_compose_and_dockerfile_agree_on_the_healthcheck():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["autodj"]
    referenced = service["healthcheck"]["test"][-1]
    assert referenced.endswith("healthcheck.sh")
    assert referenced in DOCKERFILE.read_text(encoding="utf-8")


def test_compose_keeps_every_mutable_path_in_one_volume():
    """A container that loses its database, secret store or block pool on
    recreation is not deployable, and `docker compose down` removes anything
    not in a named volume."""
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = compose["services"]["autodj"]
    mounts = [v.split(":")[1] for v in service["volumes"] if not v.strip().startswith("#")]
    assert "/data" in mounts
    assert compose["volumes"], "the data volume is not declared"


def test_compose_publishes_the_dashboard():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    ports = compose["services"]["autodj"]["ports"]
    assert any(str(p).startswith("8080:") for p in ports)


def test_the_generator_is_a_service_and_not_a_one_off_run():
    """It used to be documented as `docker compose run --rm -d autodj
    generator`. That container is not part of the project: `docker compose
    down` leaves it running, and while it lives it holds the data volume open,
    so `down -v` keeps the volume too and the next "fresh" start is not fresh.
    """
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    generator = compose["services"]["generator"]
    assert generator["command"] == ["generator"]
    assert generator["profiles"], "the generator must be opt-in, not always on"
    assert "autodj-data:/data" in generator["volumes"]


def test_the_generator_is_the_one_the_kernel_should_kill():
    """A dead generator costs one block. A dead broadcast costs the broadcast."""
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = compose["services"]
    assert services["generator"]["oom_score_adj"] > services["autodj"]["oom_score_adj"]


def test_the_readme_does_not_still_document_the_one_off_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "compose run --rm -d autodj generator" not in readme


def test_the_compose_restart_policy_has_something_to_restart_into():
    """`restart: unless-stopped` promises the container comes back. It used to
    come back silent and off air, because nothing restored what was running."""
    from app.core.config import load_config

    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert compose["services"]["autodj"]["restart"] == "unless-stopped"
    assert load_config(None, environ={}).app.resume_on_start is True


def _units() -> list[Path]:
    return sorted((ROOT / "deploy" / "systemd").glob("*.service"))


def test_every_systemd_unit_starts_something_that_exists():
    """`trance-autodj-streamer.service` ran `app.services.stream.worker`, a
    module that was never written: the broadcast is owned by the web service.
    On a native install that unit crashlooped from every boot, and the only
    place it showed was the journal.
    """
    import re

    for unit in _units():
        text = unit.read_text(encoding="utf-8")
        match = re.search(r"^ExecStart=.*?-m\s+(\S+)", text, re.MULTILINE)
        if not match:
            continue
        module = match.group(1)
        path = ROOT / (module.replace(".", "/") + ".py")
        package = ROOT / module.replace(".", "/") / "__init__.py"
        assert path.exists() or package.exists(), f"{unit.name} runs {module}, which is missing"


def test_the_units_come_up_on_their_own_after_a_reboot():
    """The whole point of a 24/7 station: nobody logs in to start it."""
    for unit in _units():
        text = unit.read_text(encoding="utf-8")
        assert "WantedBy=multi-user.target" in text, unit.name
        assert "Restart=" in text, unit.name


def test_the_nas_compose_needs_nothing_but_itself():
    """Pasted into Container Station, this file is the whole deployment: no
    build step, no source tree, an image the NAS can pull for its own chip."""
    import yaml

    text = (ROOT / "deploy" / "docker-compose.nas.yml").read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    service = doc["services"]["autodj"]
    assert service["image"].startswith("ghcr.io/themechanic-dev/trance-autodj")
    assert "build" not in service, "a NAS has no source tree to build from"
    assert "generator" not in doc["services"], "block building does not belong on a NAS"
    assert service["restart"] == "unless-stopped"
    # QTS itself listens on 8080; a container asking for it fails to start
    # after the image has already downloaded.
    assert not any(str(p).startswith("8080:") for p in service["ports"])
    assert "autodj-data" in doc["volumes"]
    # Container Station refuses a compose file that carries resource limits —
    # it wants them set in its own Advanced Settings — so they must not be here.
    for key in ("mem_limit", "cpus", "deploy", "cpu_shares", "mem_reservation"):
        assert key not in service, f"{key} makes Container Station reject the file"

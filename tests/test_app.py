"""The application must boot and answer, on a machine with nothing installed."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import db as db_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAD_APP__LOG_FORMAT", "console")
    # These tests exercise the API, not the login form.
    monkeypatch.setenv("TAD_AUTH__ENABLED", "false")
    monkeypatch.setenv("TAD_CONFIG_FILE", str(tmp_path / "no-such-config.yaml"))
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    db_module.dispose()


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_renders(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Trance AutoDJ" in response.text
    assert "ON AIR" in response.text


def test_every_page_renders(client: TestClient):
    for path in ("/", "/audio", "/visuals", "/stream", "/settings", "/logs", "/preflight"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"
        assert "Trance AutoDJ" in response.text


def test_capabilities_endpoint(client: TestClient):
    payload = client.get("/api/system/capabilities").json()
    assert payload["cpu"]["count"] >= 1
    assert payload["encoder"]["resolved"] in {"libx264", "h264_nvenc"}
    assert payload["ai_backend"]["resolved"] in {"cuda", "sdcpp", "openvino", "none"}
    # Every resolution explains itself; "why is it using x264" is answerable.
    assert payload["encoder"]["reason"]


def test_metrics_endpoint(client: TestClient):
    payload = client.get("/api/system/metrics").json()
    assert payload["cpu_count"] >= 1
    assert payload["disk_total_gb"] > 0


def test_preflight_endpoint(client: TestClient):
    payload = client.get("/api/system/preflight").json()
    assert payload.get("checks")


def test_data_directories_exist_after_boot(client: TestClient, tmp_path: Path):
    root = tmp_path / "data"
    for name in ("music", "playlists", "blocks", "state", "logs"):
        assert (root / name).is_dir()


def test_no_stream_key_value_is_leaked_by_any_endpoint(tmp_path: Path, monkeypatch):
    """The secret itself must not appear anywhere a browser can reach.

    Checking for the *word* "stream_key" would be theatre — it is a field
    name, and it legitimately appears in documentation. What matters is the
    value, so a real one is configured and then hunted for.
    """
    secret = "qwer-tyui-opas-dfgh"
    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAD_CONFIG_FILE", str(tmp_path / "no-such-config.yaml"))
    monkeypatch.setenv("TAD_STREAM__YOUTUBE__STREAM_KEY", secret)
    monkeypatch.setenv("TAD_AUTH__ENABLED", "false")
    from app.main import create_app

    with TestClient(create_app()) as c:
        for path in (
            "/",
            "/api/system/capabilities",
            "/api/system/preflight",
            "/api/system/metrics",
        ):
            assert secret not in c.get(path).text, f"leaked by {path}"
    db_module.dispose()


def test_the_log_file_never_contains_the_stream_key(tmp_path: Path, monkeypatch):
    secret = "zxcv-bnml-qazw-sxed"
    monkeypatch.setenv("TAD_APP__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAD_CONFIG_FILE", str(tmp_path / "no-such-config.yaml"))
    monkeypatch.setenv("TAD_STREAM__YOUTUBE__STREAM_KEY", secret)
    monkeypatch.setenv("TAD_AUTH__ENABLED", "false")
    from app.main import create_app

    with TestClient(create_app()) as c:
        c.get("/")
        logging.getLogger("test").error("pretending ffmpeg printed %s", secret)

    for log_file in (tmp_path / "data" / "logs").glob("*.log"):
        assert secret not in log_file.read_text(encoding="utf-8", errors="replace")
    db_module.dispose()


def test_the_audio_page_can_actually_upload(client: TestClient):
    """The README promised uploading "from the Audio page" while the page only
    told you to copy files into the volume by hand: the endpoint existed and
    nothing reached it."""
    page = client.get("/audio").text
    assert 'id="drop"' in page
    assert 'type="file"' in page
    assert "/api/audio/tracks/upload" in page


def test_upload_limits_are_published(client: TestClient):
    """So the page can refuse a file before spending the upload on it."""
    body = client.get("/api/audio/upload/limits").json()
    assert body["max_bytes"] > 0
    assert ".mp3" in body["suffixes"]


def test_the_stream_page_has_a_monitor(client: TestClient):
    """Seeing what will go out, before it goes out."""
    page = client.get("/stream").text
    assert "/api/stream/monitor/live.mp4" in page
    assert "<video" in page


def test_the_monitor_says_why_it_cannot_run(client: TestClient):
    """An empty pool or a live broadcast are both reasons; neither should be a
    blank player that never starts."""
    body = client.get("/api/stream/monitor/status").json()
    assert body["ready"] is False
    assert body["reason"]


def test_the_monitor_refuses_rather_than_streaming_nothing(client: TestClient):
    assert client.get("/api/stream/monitor/live.mp4").status_code == 409


def test_the_stream_key_never_travels_in_a_query_string(client: TestClient):
    """It is a credential: query strings reach access logs, proxy logs and
    browser history, which would undo the encrypted store and the redactor."""
    assert "?key=" not in client.get("/stream").text
    assert client.put("/api/stream/key", params={"key": "abcdefghij"}).status_code == 422
    assert client.put("/api/stream/key", json={"key": "abcdefghij"}).status_code == 200


def test_the_library_can_be_managed_without_a_shell(client: TestClient):
    """Adding music, removing it, and picking up what was copied in by hand
    are all things an operator does repeatedly. Each needs a button."""
    page = client.get("/audio").text
    assert 'id="drop"' in page  # add
    assert 'class="mini danger del"' in page  # remove one
    assert 'id="purge"' in page  # forget the ones whose files are gone
    assert 'id="scan"' in page  # pick up what was copied in


def _upload(client: TestClient, name: str, playlist: str = "", fill: bytes = b"a"):
    return client.post(
        "/api/audio/tracks/upload",
        files=[("files", (name, fill * 4000, "audio/mpeg"))],
        data={"playlist": playlist},
    ).json()


def test_the_whole_library_can_be_emptied_in_one_go(client: TestClient):
    """Deleting eighty-nine tracks one button at a time is not a workflow, and
    replacing the entire repertoire at once is an ordinary thing to need."""
    _upload(client, "one.mp3", fill=b"a")
    _upload(client, "two.mp3", fill=b"b")
    from app.core.paths import Paths

    paths = Paths.from_config(client.app.state.runtime.config)
    assert len(list(paths.music.glob("*.mp3"))) == 2

    response = client.request("DELETE", "/api/audio/tracks", json={"confirm_count": 2})
    assert response.status_code == 200
    assert response.json()["deleted"] == 2
    assert client.get("/api/audio/tracks").json()["tracks"] == []
    assert list(paths.music.glob("*.mp3")) == [], "the rows went but the files stayed"


def test_emptying_refuses_when_the_library_changed_under_you(client: TestClient):
    """The dashboard sends back the number it showed. If an upload landed in
    between, the operator agreed to destroy a different library than the one
    that is there."""
    _upload(client, "one.mp3", fill=b"a")
    _upload(client, "two.mp3", fill=b"b")

    response = client.request("DELETE", "/api/audio/tracks", json={"confirm_count": 1})
    assert response.status_code == 409
    assert "2 tracks" in response.json()["detail"]
    assert len(client.get("/api/audio/tracks").json()["tracks"]) == 2, "it deleted anyway"


def test_a_playlist_can_take_its_music_with_it(client: TestClient):
    _upload(client, "one.mp3", playlist="Album", fill=b"a")
    _upload(client, "two.mp3", playlist="Album", fill=b"b")
    album = next(
        p for p in client.get("/api/audio/playlists").json()["playlists"] if p["name"] == "Album"
    )
    client.post("/api/audio/playlists", params={"name": "Somewhere else"})
    client.post(f"/api/audio/playlists/{album['id']}/activate")
    other = next(
        p
        for p in client.get("/api/audio/playlists").json()["playlists"]
        if p["name"] == "Somewhere else"
    )
    client.post(f"/api/audio/playlists/{other['id']}/activate")

    body = client.request(
        "DELETE", f"/api/audio/playlists/{album['id']}", params={"with_tracks": True}
    ).json()
    assert body["tracks_deleted"] == 2
    assert body["tracks_kept"] == 0

    from app.core.paths import Paths

    paths = Paths.from_config(client.app.state.runtime.config)
    assert list(paths.music.glob("*.mp3")) == []


def test_music_another_playlist_also_holds_is_kept(client: TestClient):
    """Deleting the file would quietly empty the other list too."""
    _upload(client, "shared.mp3", playlist="Album", fill=b"a")
    _upload(client, "only-here.mp3", playlist="Album", fill=b"b")
    lists = {p["name"]: p for p in client.get("/api/audio/playlists").json()["playlists"]}
    shared_id = next(
        t["id"]
        for t in client.get("/api/audio/tracks").json()["tracks"]
        if "shared" in t["relpath"]
    )

    keeper = client.post("/api/audio/playlists", params={"name": "Keeper"}).json()
    added = client.put(f"/api/audio/playlists/{keeper['id']}/tracks", json=[shared_id])
    assert added.status_code == 200, added.text
    client.post(f"/api/audio/playlists/{keeper['id']}/activate")

    body = client.request(
        "DELETE", f"/api/audio/playlists/{lists['Album']['id']}", params={"with_tracks": True}
    ).json()
    assert body["tracks_deleted"] == 1
    assert body["tracks_kept"] == 1

    left = [t["relpath"] for t in client.get("/api/audio/tracks").json()["tracks"]]
    assert any("shared" in name for name in left)
    assert not any("only-here" in name for name in left)


def test_the_automatic_playlist_does_not_count_as_somewhere_else(client: TestClient):
    """ "All tracks" mirrors the library, so counting it as another list would
    make every track look shared and nothing would ever be deletable."""
    _upload(client, "one.mp3", playlist="Album", fill=b"a")
    lists = {p["name"]: p for p in client.get("/api/audio/playlists").json()["playlists"]}
    assert "All tracks" in lists, "the fixture is not testing what it claims"
    client.post(f"/api/audio/playlists/{lists['All tracks']['id']}/activate")

    body = client.request(
        "DELETE", f"/api/audio/playlists/{lists['Album']['id']}", params={"with_tracks": True}
    ).json()
    assert body["tracks_deleted"] == 1, "the automatic list blocked the deletion"


def test_deleting_a_playlist_still_leaves_the_music_alone_by_default(client: TestClient):
    _upload(client, "one.mp3", playlist="Album", fill=b"a")
    lists = {p["name"]: p for p in client.get("/api/audio/playlists").json()["playlists"]}
    client.post(f"/api/audio/playlists/{lists['All tracks']['id']}/activate")

    body = client.request("DELETE", f"/api/audio/playlists/{lists['Album']['id']}").json()
    assert body["tracks_deleted"] == 0
    assert len(client.get("/api/audio/tracks").json()["tracks"]) == 1


def test_playlists_left_empty_can_be_cleared_in_one_go(client: TestClient):
    """Emptying the library used to leave the lists behind, holding nothing
    and still sitting in the rotation — the very tidying the bulk deletes
    were added to stop."""
    _upload(client, "one.mp3", playlist="Album", fill=b"a")
    _upload(client, "two.mp3", playlist="Later", fill=b"b")
    lists = {p["name"]: p for p in client.get("/api/audio/playlists").json()["playlists"]}
    client.put(
        "/api/audio/playlists/rotation",
        json={"playlist_ids": [lists["Album"]["id"], lists["Later"]["id"]]},
    )

    client.request("DELETE", "/api/audio/tracks", json={"confirm_count": 2})
    after = client.get("/api/audio/playlists").json()["playlists"]
    assert {p["name"] for p in after} >= {"Album", "Later"}, "the fixture proves nothing"
    assert all(p["track_count"] == 0 for p in after)

    body = client.post("/api/audio/playlists/prune-empty").json()
    assert body["removed"] == 2
    assert sorted(body["names"]) == ["Album", "Later"]

    left = [p["name"] for p in client.get("/api/audio/playlists").json()["playlists"]]
    assert left == ["All tracks"], "the automatic list must survive as the fallback"
    assert client.get("/api/audio/playlists/rotation").json()["playlist_ids"] == []


def test_pruning_leaves_a_playlist_that_still_has_music(client: TestClient):
    _upload(client, "one.mp3", playlist="Keep me", fill=b"a")
    client.post("/api/audio/playlists", params={"name": "Empty one"})

    body = client.post("/api/audio/playlists/prune-empty").json()
    assert body["names"] == ["Empty one"]
    left = [p["name"] for p in client.get("/api/audio/playlists").json()["playlists"]]
    assert "Keep me" in left


def test_something_is_always_active_after_pruning(client: TestClient):
    """The active list is not spared when it is empty — there is nothing in it
    to play — so the automatic one has to come back on air by itself."""
    client.post("/api/audio/playlists", params={"name": "Empty one"})
    lists = {p["name"]: p for p in client.get("/api/audio/playlists").json()["playlists"]}
    client.post(f"/api/audio/playlists/{lists['Empty one']['id']}/activate")

    client.post("/api/audio/playlists/prune-empty")
    after = client.get("/api/audio/playlists").json()["playlists"]
    assert [p["name"] for p in after if p["is_active"]] == ["All tracks"]


def test_an_empty_library_still_offers_to_clear_the_lists(client: TestClient):
    """ "The library is already empty" and nothing else is a dead end: the
    playlists left holding no music are exactly where the operator still is,
    and the button that clears them is the one they just pressed."""
    page = client.get("/audio").text
    handler = page[page.index("$('eraselib')") :]
    handler = handler[: handler.index("$('purge')")]
    assert "prunePlaylists" in handler, "the empty-library path dead-ends"
    assert "already empty" in handler


def test_music_can_be_composed_from_the_dashboard(client: TestClient):
    """The station generates its own visuals; generating its own music closes
    the loop — nothing downloaded, nothing anybody else's, nothing to claim."""
    page = client.get("/audio").text
    assert 'id="gengo"' in page
    assert 'id="gencount"' in page

    body = client.post(
        "/api/audio/tracks/generate", json={"count": 2, "minutes": 1.0, "playlist": "Generated"}
    ).json()
    assert body["queued"] == 2
    assert len(set(body["seeds"])) == 2, "two tracks with one seed would be one track twice"


def test_the_styles_are_offered_and_an_unknown_one_is_refused(client: TestClient):
    """The page fills its dropdown from the server, so adding a style is one
    place, not two — and a typo in the request is a 422, not a silent
    fallback to something else."""
    body = client.get("/api/audio/tracks/generate/styles").json()
    names = {s["name"] for s in body["styles"]}
    assert {"uplifting", "progressive", "psy", "tech"} <= names
    assert all(s["blurb"] and len(s["bpm"]) == 2 for s in body["styles"])
    assert 'id="genstyle"' in client.get("/audio").text

    bad = client.post("/api/audio/tracks/generate", json={"count": 1, "style": "hardstyle"})
    assert bad.status_code == 422
    good = client.post("/api/audio/tracks/generate", json={"count": 1, "style": "psy"})
    assert good.status_code == 200


def test_a_given_seed_is_the_seed_used(client: TestClient):
    body = client.post("/api/audio/tracks/generate", json={"count": 1, "seed": 4242}).json()
    assert body["seeds"] == [4242]


def test_generation_progress_is_visible(client: TestClient):
    body = client.get("/api/audio/tracks/generate/status").json()
    assert body["running"] == 0
    assert body["jobs"] == []


def test_the_play_log_answers_the_question_a_claim_asks(client: TestClient):
    from datetime import UTC, datetime

    from app.core.db import session_scope
    from app.models.entities import PlayLogEntry

    page = client.get("/audio").text
    assert 'id="logwhen"' in page
    assert 'id="logask"' in page

    with session_scope() as session:
        session.add(
            PlayLogEntry(
                relpath="generated/claimed.mp3",
                title="Claimed",
                artist="Trance AutoDJ",
                started_at=datetime(2026, 9, 12, 20, 0, tzinfo=UTC),
                ended_at=datetime(2026, 9, 12, 20, 6, tzinfo=UTC),
            )
        )

    # Typed as the operator reads it off the screen: station time, no zone.
    body = client.get("/api/audio/playlog/at", params={"when": "2026-09-12T23:03"}).json()
    assert body["entry"] is not None, "23:03 Athens is 20:03 UTC and should have matched"
    assert body["entry"]["title"] == "Claimed"

    empty = client.get("/api/audio/playlog/at", params={"when": "2026-09-12T19:00"}).json()
    assert empty["entry"] is None


def test_a_logged_time_says_which_clock_it_is_on(client: TestClient):
    """SQLite keeps no offset, so what comes back out of the column is a naive
    wall clock. Serialised bare, every browser reads it as local time and the
    log answers a claim with a track from three hours earlier — confidently.
    """
    from datetime import UTC, datetime

    from app.core.db import session_scope
    from app.models.entities import PlayLogEntry

    with session_scope() as session:
        session.add(
            PlayLogEntry(
                relpath="a.mp3",
                title="A",
                started_at=datetime(2026, 9, 12, 4, 12, 21, tzinfo=UTC),
                ended_at=datetime(2026, 9, 12, 4, 18, 0, tzinfo=UTC),
            )
        )

    entry = client.get("/api/audio/playlog").json()["entries"][0]
    assert entry["started_at"].endswith("+00:00"), entry["started_at"]
    assert entry["ended_at"].endswith("+00:00"), entry["ended_at"]
    assert datetime.fromisoformat(entry["started_at"]).tzinfo is not None


def test_an_unreadable_time_is_refused_not_guessed(client: TestClient):
    response = client.get("/api/audio/playlog/at", params={"when": "sometime tuesday"})
    assert response.status_code == 422


def test_a_track_can_be_flagged_without_being_removed(client: TestClient):
    """A claim lands while the station is on air, and pulling the file out
    from under Liquidsoap mid-broadcast is how you get silence."""
    _upload(client, "suspect.mp3", fill=b"a")
    track = client.get("/api/audio/tracks").json()["tracks"][0]

    flagged = client.post(
        "/api/audio/tracks/flagged",
        params={"track_id": track["id"]},
        json={"note": "claimed 12/09"},
    ).json()
    assert flagged["flagged"] is True
    assert flagged["flag_note"] == "claimed 12/09"
    assert len(client.get("/api/audio/tracks").json()["tracks"]) == 1, "it was removed, not flagged"

    body = client.request("DELETE", "/api/audio/tracks/flagged", params={"remove": True}).json()
    assert body["deleted"] == 1
    assert client.get("/api/audio/tracks").json()["tracks"] == []


def test_flags_can_be_lifted_without_losing_the_music(client: TestClient):
    _upload(client, "suspect.mp3", fill=b"a")
    track = client.get("/api/audio/tracks").json()["tracks"][0]
    client.post("/api/audio/tracks/flagged", params={"track_id": track["id"]}, json={"note": ""})

    body = client.request("DELETE", "/api/audio/tracks/flagged").json()
    assert body == {"cleared": 1, "deleted": 0}
    assert client.get("/api/audio/tracks").json()["tracks"][0]["flagged"] is False


def test_every_tab_has_an_icon(client: TestClient):
    """A browser tab with no icon is a grey page among fifty others. And the
    browser asks for /favicon.ico whether or not the page says where the icon
    is — on the login page, on a 404 — so that has to answer as well."""
    page = client.get("/").text
    assert 'rel="icon"' in page
    assert "/static/img/icon.svg" in page
    assert 'rel="apple-touch-icon"' in page

    for path, kind in (
        ("/favicon.ico", "image/x-icon"),
        ("/static/img/icon.svg", "image/svg+xml"),
        ("/static/img/icon-32.png", "image/png"),
        ("/static/site.webmanifest", None),
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        if kind:
            assert response.headers["content-type"].startswith(kind), (path, response.headers)
        assert len(response.content) > 200, f"{path} is empty"


def test_the_settings_page_carries_the_makers_mark(client: TestClient):
    """The public projects sign as the mechanic. Not a real name — that is the
    point of a handle — and never any other name."""
    for path in ("/", "/audio", "/stream", "/settings"):
        page = client.get(path).text
        assert "HOME LAB" in page and "by the mechanic" in page, path


def test_both_bulk_deletions_have_a_button(client: TestClient):
    page = client.get("/audio").text
    assert 'id="eraselib"' in page
    assert 'id="prunepl"' in page
    assert "delplmus" in page


def test_purging_missing_tracks_is_available(client: TestClient):
    body = client.post("/api/audio/tracks/purge-missing").json()
    assert body["purged"] == 0


def test_the_monitor_starts_with_sound(client: TestClient):
    """You press it to hear whether the station sounds right. Starting muted
    also makes it "video-only background media", which Chrome pauses to save
    power as soon as the tab is not the visible one — so the picture stopped
    too. The button press is the gesture that permits audio."""
    page = client.get("/stream").text
    assert "video.muted = false" in page
    assert "video.muted = true" in page, "there must still be a fallback if sound is refused"


def test_resetting_a_setting_that_would_break_the_config_changes_nothing(client: TestClient):
    """Removing an override can invalidate the configuration just as easily as
    adding one: clip.min_duration_s back to its default of 20 while
    clip.max_duration_s is still pinned at 8 is enough.

    The reset used to commit first and validate afterwards, so the caller got
    a 500 about a change that had in fact already been made — and the running
    process kept the old configuration, which then disagreed with the
    database.
    """
    client.put("/api/settings", json={"visual.clip.min_duration_s": 6.0})
    client.put("/api/settings", json={"visual.clip.max_duration_s": 8.0})

    response = client.delete("/api/settings/visual.clip.min_duration_s")
    assert response.status_code == 409, response.text
    assert "max_duration_s" in response.text

    # And nothing moved: the override is still there, still 6.
    rows = {
        r["path"]: r
        for group in client.get("/api/settings").json()["groups"].values()
        for r in group
    }
    assert rows["visual.clip.min_duration_s"]["value"] == 6.0
    assert rows["visual.clip.min_duration_s"]["overridden"] is True


def test_resetting_in_a_workable_order_succeeds(client: TestClient):
    client.put("/api/settings", json={"visual.clip.min_duration_s": 6.0})
    client.put("/api/settings", json={"visual.clip.max_duration_s": 8.0})
    assert client.delete("/api/settings/visual.clip.max_duration_s").status_code == 200
    assert client.delete("/api/settings/visual.clip.min_duration_s").status_code == 200
    rows = {
        r["path"]: r
        for group in client.get("/api/settings").json()["groups"].values()
        for r in group
    }
    assert rows["visual.clip.min_duration_s"]["overridden"] is False
    assert rows["visual.clip.max_duration_s"]["overridden"] is False


def test_the_playlists_page_can_build_and_order_lists(client: TestClient):
    """Creating lists, choosing their tracks and arranging the order they play
    in were all reachable only from the API — the page showed a list of names
    and an activate button."""
    page = client.get("/audio").text
    for control in (
        'id="newlist"',
        'id="rotation"',
        'class="mini rot-up"',
        'class="mini edit"',
        'id="mode-rotation"',
        'id="editsave"',
    ):
        assert control in page, control


def test_the_rotation_survives_a_round_trip(client: TestClient):
    first = client.post("/api/audio/playlists", params={"name": "Opening"}).json()
    second = client.post("/api/audio/playlists", params={"name": "Peak"}).json()

    client.put("/api/audio/playlists/rotation", json={"playlist_ids": [second["id"], first["id"]]})
    body = client.get("/api/audio/playlists/rotation").json()
    assert body["playlist_ids"] == [second["id"], first["id"]]
    assert [p["name"] for p in body["playlists"]] == ["Peak", "Opening"]

    client.put("/api/audio/playlists/rotation", json={"playlist_ids": [first["id"]]})
    assert client.get("/api/audio/playlists/rotation").json()["playlist_ids"] == [first["id"]]


def test_the_playback_switch_is_a_setting_like_any_other(client: TestClient):
    assert client.get("/api/audio/playlists/rotation").json()["playback"] == "single"
    client.put("/api/settings", json={"audio.playlist_playback": "rotation"})
    assert client.get("/api/audio/playlists/rotation").json()["playback"] == "rotation"


def test_switching_to_the_rotation_rewrites_what_is_playing(client: TestClient, tmp_path: Path):
    """The switch regenerated the Liquidsoap script and left it reading a file
    that still held the single list, so flipping it appeared to do nothing."""
    from app.core.paths import Paths

    first = client.post("/api/audio/playlists", params={"name": "Opening"}).json()
    second = client.post("/api/audio/playlists", params={"name": "Peak"}).json()
    client.put("/api/audio/playlists/rotation", json={"playlist_ids": [second["id"], first["id"]]})

    client.put("/api/settings", json={"audio.playlist_playback": "rotation"})

    paths = Paths.from_config(client.app.state.runtime.config)
    header = paths.active_playlist.read_text(encoding="utf-8").splitlines()[1]
    assert "rotation:" in header
    assert "Peak" in header and "Opening" in header
    assert header.index("Peak") < header.index("Opening"), "the order the operator chose"


def test_a_folder_becomes_a_playlist(client: TestClient):
    """The point of the feature: the operator keeps albums in folders, and the
    folder name is already the name of the list they would have typed."""
    files = [
        ("files", ("one.mp3", b"a" * 4000, "audio/mpeg")),
        ("files", ("two.mp3", b"b" * 4000, "audio/mpeg")),
    ]
    body = client.post(
        "/api/audio/tracks/upload", files=files, data={"playlist": "Goa 2004"}
    ).json()
    assert body["accepted"] == ["one.mp3", "two.mp3"]
    assert body["playlist"]["name"] == "Goa 2004"
    assert body["playlist"]["track_count"] == 2

    names = [p["name"] for p in client.get("/api/audio/playlists").json()["playlists"]]
    assert "Goa 2004" in names


def test_uploading_the_rest_of_a_folder_tops_the_list_up(client: TestClient):
    client.post(
        "/api/audio/tracks/upload",
        files=[("files", ("one.mp3", b"a" * 4000, "audio/mpeg"))],
        data={"playlist": "Goa 2004"},
    )
    body = client.post(
        "/api/audio/tracks/upload",
        files=[("files", ("two.mp3", b"b" * 4000, "audio/mpeg"))],
        data={"playlist": "Goa 2004"},
    ).json()
    assert body["playlist"]["track_count"] == 2, "the second upload replaced the list"


def test_the_same_folder_twice_does_not_double_the_library(client: TestClient):
    """Re-importing an album must not leave "track (1).mp3" beside every file,
    with no way to tell which copy the playlist points at."""
    payload = [("files", ("one.mp3", b"a" * 4000, "audio/mpeg"))]
    client.post("/api/audio/tracks/upload", files=payload, data={"playlist": "Goa 2004"})
    body = client.post(
        "/api/audio/tracks/upload",
        files=[("files", ("one.mp3", b"a" * 4000, "audio/mpeg"))],
        data={"playlist": "Goa 2004"},
    ).json()

    assert body["duplicates"] == ["one.mp3"]
    assert client.get("/api/audio/tracks").json()["total"] == 1
    assert body["playlist"]["track_count"] == 1


def test_uploading_without_a_playlist_still_just_imports(client: TestClient):
    body = client.post(
        "/api/audio/tracks/upload",
        files=[("files", ("loose.mp3", b"c" * 4000, "audio/mpeg"))],
    ).json()
    assert body["accepted"] == ["loose.mp3"]
    assert body["playlist"] is None


def test_the_page_offers_a_folder_picker_that_skips_everything_else(client: TestClient):
    page = client.get("/audio").text
    assert 'id="pickdir"' in page
    assert "webkitdirectory" in page
    assert "webkitRelativePath" in page, "the folder name has to come from somewhere"


def test_dropping_a_folder_reads_the_tree_not_just_the_files(client: TestClient):
    """dataTransfer.files is empty for a dropped directory — the tree is only
    reachable through the entries API — so the drop zone accepted folders by
    doing nothing at all while the button worked.
    """
    page = client.get("/audio").text
    assert "webkitGetAsEntry" in page
    assert "createReader" in page
    assert "readEntries" in page


def test_the_drop_zone_and_the_button_share_one_import(client: TestClient):
    """Two code paths for the same job drift: one accepts a file type the
    other refuses, and only one of them says why."""
    page = client.get("/audio").text
    assert page.count("async function importFolder") == 1
    assert "importFolder(folder.name" in page  # the drop
    assert "importFolder(folder, picked)" in page  # the button


def test_a_stopped_broadcast_does_not_report_itself_as_feeding(client: TestClient):
    """The page had no idea of "stopped": it printed "feeding" whenever the
    feeder was not waiting for blocks, so a station that had been off for an
    hour showed a block playing, an elapsed time still counting up, and 30 fps.
    """
    page = client.get("/stream").text
    assert "f.running" in page
    assert "stopped" in page
    assert "last run" in page, "off-air figures have to say they are the last run's"

    body = client.get("/api/stream/status").json()
    assert body["feeder"]["running"] is False
    assert body["feeder"]["current_block_elapsed_s"] == 0.0


def test_stopping_does_not_leave_its_own_noise_as_an_error(client: TestClient):
    """ "ffmpeg closed the pipe" and "Failed to update header" are what a clean
    shutdown sounds like. Shown in red after every STOP, they teach the
    operator that the error line means nothing."""
    from app.services.stream.manager import _is_benign

    assert _is_benign("ffmpeg closed the pipe")
    assert _is_benign("[flv @ 0x55] Failed to update header with correct filesize.")
    assert not _is_benign("Connection refused")


def test_no_control_row_can_push_its_button_out_of_the_card():
    """A row of inputs and a button that cannot shrink overflows its card, and
    the button is clipped away with no sign that anything is missing.

    This is how the Google client credentials became impossible to save: two
    inputs and a save button need about 400px side by side, cards are 330, and
    an <input> will not shrink below its intrinsic width unless it is told to.
    So the row must either wrap or stack.
    """
    import re

    templates = Path(__file__).resolve().parent.parent / "app" / "web" / "templates"
    for page in sorted(templates.glob("*.html")):
        html = page.read_text(encoding="utf-8")
        # Each <div ... style="...display:flex..."> and what it holds up to the
        # next closing div at the same nesting is more than a regex can see, so
        # take the crude slice: from the opening tag to the first </div>.
        for match in re.finditer(r'<div[^>]*style="([^"]*display:\s*flex[^"]*)"[^>]*>', html):
            style = " ".join(match.group(1).split())
            body = html[match.end() : html.find("</div>", match.end())]
            inputs = body.count("<input")
            if inputs < 2 or "<button" not in body:
                continue
            wraps = "flex-wrap:wrap" in style.replace(" ", "")
            stacks = "flex-direction:column" in style.replace(" ", "")
            assert wraps or stacks, (
                f"{page.name}: a flex row holds {inputs} inputs and a button but "
                "neither wraps nor stacks — the button lands outside the card"
            )

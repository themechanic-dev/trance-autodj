"""Every filesystem location the application uses, derived from app.data_dir.

No module builds a data path by string concatenation; they ask here. That way
relocating the data directory (a Docker volume, a different disk) is one
setting and nothing else has to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import Config


@dataclass(frozen=True)
class Paths:
    root: Path

    music: Path
    playlists: Path
    images: Path
    clips: Path
    blocks: Path
    blocks_rejected: Path
    thumbnails: Path
    state: Path
    logs: Path
    models: Path
    tmp: Path

    db_file: Path
    nowplaying_file: Path
    stream_state_file: Path
    audio_state_file: Path
    generator_state_file: Path
    active_playlist: Path
    secrets_file: Path
    session_key_file: Path
    icecast_secrets_file: Path
    liquidsoap_script: Path
    icecast_config: Path
    video_fifo: Path

    @classmethod
    def from_config(cls, cfg: Config) -> Paths:
        root = Path(cfg.app.data_dir).expanduser().resolve()
        state = root / "state"
        blocks = root / "blocks"
        models = (
            Path(cfg.visual.ai.models_dir).expanduser().resolve()
            if cfg.visual.ai.models_dir
            else root / "models"
        )
        return cls(
            root=root,
            music=root / "music",
            playlists=root / "playlists",
            images=root / "images",
            clips=root / "clips",
            blocks=blocks,
            blocks_rejected=blocks / "rejected",
            thumbnails=root / "thumbnails",
            state=state,
            logs=root / "logs",
            models=models,
            tmp=root / "tmp",
            db_file=root / "db.sqlite",
            nowplaying_file=state / "nowplaying.json",
            stream_state_file=state / "stream.json",
            audio_state_file=state / "audio.json",
            generator_state_file=state / "generator.json",
            active_playlist=root / "playlists" / "active.m3u",
            secrets_file=state / "secrets.enc",
            session_key_file=state / "session.key",
            icecast_secrets_file=state / "icecast.secrets.json",
            liquidsoap_script=state / "autodj.liq",
            icecast_config=state / "icecast.xml",
            video_fifo=state / cfg.stream.fifo_name,
        )

    def directories(self) -> tuple[Path, ...]:
        """Directories that must exist and be writable before anything runs."""
        return (
            self.root,
            self.music,
            self.playlists,
            self.images,
            self.clips,
            self.blocks,
            self.blocks_rejected,
            self.thumbnails,
            self.state,
            self.logs,
            self.models,
            self.tmp,
        )

    def ensure(self) -> None:
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)


__all__ = ["Paths"]

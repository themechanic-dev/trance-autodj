"""Prompt presets for the trance aesthetic.

Loaded from config/prompts.json so they can be edited from the dashboard
without touching code. Every preset is deliberately abstract: no faces, no
text, no recognisable brands or characters — partly because diffusion models
are bad at all three, and partly because a 24/7 unattended stream should not
be generating anything it would need a human to review.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_NEGATIVE = (
    "text, watermark, logo, signature, face, human, person, hands, blurry, "
    "low quality, jpeg artifacts, oversaturated, borders, frame"
)

FALLBACK_PROMPT = (
    "deep space nebula, clouds of cyan and violet gas, distant starfield, "
    "volumetric depth, abstract, non-representational"
)


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    prompt: str
    weight: float = 1.0

    def full_prompt(self, style_suffix: str = "") -> str:
        return f"{self.prompt}, {style_suffix}" if style_suffix else self.prompt

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "prompt": self.prompt, "weight": self.weight}


class PromptSet:
    def __init__(
        self,
        presets: list[Preset],
        *,
        negative_prompt: str = DEFAULT_NEGATIVE,
        style_suffix: str = "",
    ) -> None:
        if not presets:
            presets = [Preset(id="fallback", name="Fallback", prompt=FALLBACK_PROMPT)]
        self._presets = presets
        self.negative_prompt = negative_prompt
        self.style_suffix = style_suffix

    def __len__(self) -> int:
        return len(self._presets)

    def __iter__(self):
        return iter(self._presets)

    @property
    def all(self) -> list[Preset]:
        return list(self._presets)

    def by_id(self, preset_id: str) -> Preset | None:
        return next((p for p in self._presets if p.id == preset_id), None)

    def choose(self, rng: random.Random) -> Preset:
        weights = [max(0.0, p.weight) for p in self._presets]
        if sum(weights) <= 0:
            return rng.choice(self._presets)
        return rng.choices(self._presets, weights=weights, k=1)[0]

    @classmethod
    def load(cls, path: Path) -> PromptSet:
        if not path.is_file():
            log.warning("no prompt file at %s; using the built-in fallback", path)
            return cls([])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("could not read %s (%s); using the built-in fallback", path, exc)
            return cls([])

        presets: list[Preset] = []
        for entry in data.get("presets", []):
            if not entry.get("enabled", True):
                continue
            prompt = str(entry.get("prompt", "")).strip()
            if not prompt:
                continue
            presets.append(
                Preset(
                    id=str(entry.get("id") or f"preset{len(presets)}"),
                    name=str(entry.get("name") or entry.get("id") or "Untitled"),
                    prompt=prompt,
                    weight=float(entry.get("weight", 1.0)),
                )
            )
        log.info("loaded %d prompt presets from %s", len(presets), path)
        return cls(
            presets,
            negative_prompt=str(data.get("negative_prompt") or DEFAULT_NEGATIVE),
            style_suffix=str(data.get("style_suffix") or ""),
        )


__all__ = ["DEFAULT_NEGATIVE", "FALLBACK_PROMPT", "Preset", "PromptSet"]

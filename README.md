# Trance AutoDJ & Visual Broadcaster

A self-hosted system that plays a trance playlist gaplessly with long
crossfades, renders abstract generative visuals, and broadcasts both to
YouTube Live 24/7 — on a machine with no GPU, without pinning the CPU.

> **Status: complete and measured.** Audio, visuals, broadcast and dashboard
> all work end to end. A live 1280×720 broadcast costs **0.26 of one core** on
> a 16-core machine, held flat across a 37-minute run — the number the whole
> architecture exists to produce. See [Measurements](#measurements).

---

## The idea

Real-time AI video on a CPU is not possible, and pretending otherwise is how
these projects die. So generation and broadcast are separated completely:

```
        SLOW, OFFLINE, LOW PRIORITY                FAST, LIVE, ALMOST FREE
   ┌────────────────────────────────┐      ┌──────────────────────────────┐
   │  visual generator (nice 19)    │      │  streamer                    │
   │  · AI stills  → animated clips │      │  · reads finished .ts blocks │
   │  · procedural animations       │ ───► │  · ffmpeg -c:v copy          │
   │  · clips → 10-minute blocks    │ pool │  · adds AAC audio only       │
   │  · crossfades baked in         │      │  · → RTMP → YouTube          │
   └────────────────────────────────┘      └──────────────────────────────┘
```

The generator may take hours per block; nobody is waiting for it. The streamer
never re-encodes video, so its cost is one AAC audio encode — a few percent of
one core. Every crossfade the viewer sees was baked into a file long before
the broadcast started.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  WEB DASHBOARD (FastAPI)                                       │
│  music upload · playlists · visual pool · settings · control   │
└───────────┬────────────────────────────────────┬───────────────┘
            │                                    │
┌───────────▼──────────────┐        ┌────────────▼───────────────┐
│  AUDIO ENGINE            │        │  VISUAL GENERATOR (offline)│
│  Liquidsoap              │        │  nice 19 / CPUQuota        │
│  playlist + crossfade    │        │  A) AI images (CPU or GPU) │
│         │                │        │  B) procedural animations  │
│         ▼                │        │         ↓                  │
│  Icecast (localhost)     │        │  clips → blocks (.ts)      │
└───────────┬──────────────┘        └────────────┬───────────────┘
            │                            ┌───────▼───────┐
            │                            │  BLOCK POOL   │
            │                            │  /data/blocks │
            │                            └───────┬───────┘
┌───────────▼────────────────────────────────────▼───────────────┐
│  STREAMER   feeder → FIFO → ffmpeg (video copy + AAC) → RTMP   │
└────────────────────────────────────────────────────────────────┘
```

## CPU and GPU

The same image runs on both. Nothing is assumed; everything is probed at
startup and the answer is shown on the dashboard and at
`/api/system/capabilities`.

| | detected | used for |
|---|---|---|
| **NVIDIA GPU** | `nvidia-smi`, then a real one-frame test encode | `h264_nvenc` encoding, CUDA image generation |
| **CPU only** | always available | `libx264 -preset veryfast`, procedural visuals |

`video.encoder: auto` and `visual.ai.backend: auto` resolve themselves. A
config written for a GPU machine still boots on a CPU-only VM: the unavailable
choice degrades with a logged reason instead of failing.

> ⚠️ A listed encoder is not a working encoder. ffmpeg advertises
> `h264_nvenc` whenever it was *built* with support, including on machines
> with no NVIDIA card. That is why the probe encodes an actual frame.

## Visuals

Six procedural generators, each a plugin with the same tiny interface
(`frame index -> RGB array`). No model, no download, no network.

| generator | what it is | loops by itself |
|---|---|---|
| `plasma` | interfering sine fields in slow drift | yes |
| `domainwarp` | marbled fields from noise warped by noise | yes |
| `waves` | moire interference between drifting plane waves | yes |
| `tunnel` | endless zoom down a textured corridor | yes |
| `flowfield` | particle trails through a curl-noise field | no |
| `reaction_diffusion` | Gray-Scott chemistry unfolding slowly | no |

Four of them are periodic by construction: time is the third dimension of a
*tileable* noise lattice, so sampling one full period returns the first frame
exactly and the clip loops seamlessly for free. The two stateful ones cannot —
their output is history — so their clips are rendered twice, the second pass
blending a kept tail onto the head. That costs CPU but keeps memory bounded at
about 20 MB instead of buffering the whole clip.

Measured at 640×360 (the default half-scale render), per frame:

| generator | ms/frame | vs real time |
|---|---|---|
| flowfield | 8 | 4.1× |
| reaction_diffusion | 22 | 1.5× |
| plasma | 24 | 1.4× |
| waves | 31 | 1.1× |
| tunnel | 38 | 0.9× |
| domainwarp | 88 | 0.4× |

Mixed evenly that is ~35 ms/frame, so a 10-minute block takes roughly **11
minutes of one core** to render, plus about a minute to encode. It runs at
`nice 19` under a CPU quota, so it takes as long as it takes.

Palettes and prompts live in `config/palettes.json` and `config/prompts.json`
and are re-read when the file changes — no restart.

## Tempo and beat alignment

Tracks are analysed when they are scanned — offline, never on the live path —
and the result is stored. Two settings then use it:

```yaml
audio:
  crossfade:
    beat_aligned: true        # crossfade = a whole number of bars
    beat_align_cue_in: false  # also trim each track to its first beat
```

With `beat_aligned`, the dashboard writes each track's own crossfade length
into the playlist and Liquidsoap uses it. A 14-second target becomes 8 bars at
140 BPM (13.7 s) and 7 bars at 128 BPM (13.1 s), so the fade spans musical
time rather than landing wherever fourteen seconds happens to end:

```
#EXTINF:90,Test 140 - 140 BPM
annotate:bpm="139.90",liq_cross_duration="13.724",bars="8":/data/music/140bpm.mp3
```

`beat_align_cue_in` is separate and off by default because it *removes* audio:
it skips the start of each track to reach the first beat, capped by
`max_cue_in_s`. Alignment of the grids, at the cost of up to two seconds of
intro.

**The detector.** By default this is about a hundred lines of numpy — spectral
flux weighted towards the kick, autocorrelation over a restricted tempo range,
then a phase search. Measured against synthetic 4/4 material at eight tempos:

```
tempo : mean error 0.18 BPM, worst 0.26     (0.04 beats of drift over 14 s)
phase : mean error 24 ms,    worst 43 ms    (a beat at 140 BPM is 429 ms)
```

`librosa` is used instead when it is installed, and `audio.analysis.prefer_librosa`
controls that. It is not a requirement: it pulls in scipy, scikit-learn and
numba, which is a great deal of machinery to add for one number about music
whose shape we already know. Install it if you want it:

```bash
pip install librosa
```

Anything outside `analysis.min_bpm`..`max_bpm` (118–152 by default) is
rejected rather than accepted, because a match outside trance tempo is almost
always a half- or double-tempo error rather than a discovery.

## Composing music

The station generates its own visuals; it can generate its own music too.
**Audio → compose** asks for a count, a length, a style and a playlist, and
a few minutes later there are tagged MP3s in the library that nobody else
has any rights to, because nothing in them was recorded — every sound is
computed from nothing.

```
seed → key, tempo, chord loop, arpeggio shape, melody rhythm and contour
     → drums, bass, arpeggio, pads, lead, risers, fills
     → sidechain, delay, reverb, stereo spread
     → ffmpeg → tagged MP3 in music/generated/
```

Four styles, and they differ where the genre differs — the bassline and
the journey, not just the tempo:

| style | BPM | what makes it |
|---|---|---|
| **uplifting** | 136–142 | builds, breaks, drops; offbeat bass; a melody you could hum |
| **progressive** | 126–131 | no drop — the filter opens across the whole track; bass on the beat, long and soft |
| **psy** | 142–148 | the rolling bassline (three sixteenths after every kick); an acid lead; hardly a pause |
| **tech** | 138–142 | driving and stripped back; the groove and the filter talk, the melody is a hint |

**mixed** lets the seed choose, so a batch of ten is not ten of the same.

**Same seed, same track, always** — sample for sample. A track worth keeping
can be rebuilt from its number; a batch pressed twice never repeats. The
tempo is written into the tags, because we composed it and know it exactly:
no analysis pass, and the beat-aligned crossfade gets a number it can trust.

Every bar of a section is not the same bar: a fill tightens at the end of
each eight, hi-hats double halfway through a phrase, a ride arrives on the
second, and the last note of a melodic phrase falls home so the phrase ends
rather than stops. The stereo width comes from panning the individual
detuned voices of each supersaw — not from delaying one channel, which
collapses the moment anything sums to mono. A phone, a club system and half
of YouTube's listeners all do.

It is numpy on one core, no model and no GPU — about ten times faster than
real time on a desktop, so a five-minute track takes half a minute; on a
small ARM NAS it is closer to real time, so the same track takes five. It
peaks at about 750 MB of memory for a five-minute track, in proportion to
the length. That is deliberate: the container cannot see a graphics card,
and a station meant to run on a NAS cannot depend on one.

**Ask for as many as you like.** A hundred tracks is a day on a desktop and
a week on a small NAS, and both are fine: the request returns at once and
the tracks go into a queue in the database, where one worker takes them one
at a time. Each is composed in a process of its own at the lowest CPU
priority, so a station that is on air while it composes keeps every frame —
the composer gets what is left over, which on a busy NAS may be very little,
and that is the right way round. The queue survives a restart: a track that
was half-built when the container went down goes back to the front, and the
rest are still waiting where they were. **stop the queue** drops whatever
has not started; the one being built finishes.

**What it is and is not.** It is a rule-based composer with a good ear for
the form, and it makes background-grade trance that is recognisably the
genre and belongs to you. It is not a hit machine: the rules vary
parameters, they do not have ideas, and after a few tracks a listener who is
paying attention will hear the shape. For a station that plays all day it
does the job; for a track anyone would seek out, write one.

## The play log

A copyright notice arrives with a timestamp and a thirty-second excerpt, and
points at your whole library. The station now keeps a record of what was on
air when: **Audio → play log**, type the time the notice names — in your own
clock, the way you read it off the screen — and it answers with the file.

The record keeps the name and the path in the row itself rather than looking
them up, so it still answers after you have deleted the track — which is the
first thing anyone does once they know which one it was.

**flag** marks a track without removing it: a notice lands while the station
is on air, and taking a file out from under Liquidsoap mid-broadcast is how
you get silence. Flag now, and **delete flagged** sweeps them up when it is
convenient. **unflag all** is for the false alarm.

## Requirements

| | Minimum | Comfortable |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB | 60 GB+ (the block pool defaults to a 50 GB cap) |
| OS | Linux with ffmpeg 6+, Liquidsoap 2.2+ | Ubuntu 26.04 (ffmpeg 8, Liquidsoap 2.4) |
| Network | 5 Mbit/s sustained upload for 720p30 at 3500 kbit/s | |

Realistic figures, measured against the intended workload:

- **Streaming** costs almost nothing: video is copied, not encoded. Expect a
  few percent of one core, plus the AAC encode.
- **Generating** one 10-minute procedural block costs roughly 10–25 minutes of
  one core, and it runs at `nice 19` under a CPU quota.
- **AI stills** on a CPU take 15–40 s each at 512×512 with SD-Turbo; on a
  modern NVIDIA GPU, well under a second.

A GPU is optional. With no model downloaded at all, the system still produces
a full visual pool from the procedural generators.

## Install

### Docker — a PC, a VPS, a QNAP, a Synology

```bash
git clone https://github.com/themechanic-dev/trance-autodj.git
cd trance-autodj/deploy
docker compose up -d
```

Then open <http://localhost:8080> (or your NAS's address on port 8080). The
first visit asks you to choose a password; nothing else is reachable until
you do.

That pulls a ready-made image for your machine's architecture — both
`amd64` and `arm64` are published under the same tag, so an Intel Synology,
an ARM QNAP and a laptop are the same three lines. The image carries ffmpeg,
Liquidsoap and Icecast, so the host needs nothing but Docker.

To build from source instead — after changing the code, or because you would
rather not pull from a registry — `docker compose up -d --build`. On an ARM
NAS that takes twenty to forty minutes and needs the internet; the pull
takes a minute.

### The first hour of pictures

A fresh install has no video blocks, and building one takes the generator
an hour or two of a core. So the release ships a **starter pack**: six
ten-minute blocks (1.7 GB, an hour of visuals, every procedural generator
represented) as `starter-blocks.tar` on the
[Releases](https://github.com/themechanic-dev/trance-autodj/releases) page.
Drop it into the data volume with one command from the directory you
downloaded it to:

```bash
docker run --rm -v deploy_autodj-data:/data -v "$PWD":/in alpine \
  tar -xf /in/starter-blocks.tar -C /data/blocks
```

Open the **Visuals** page and they are adopted. The generator adds to them
from there, or you build more on a faster machine and copy them in the same
way. The blocks are 1280×720 at 30 fps; the streamer copies them to YouTube
without re-encoding, which is why the broadcast costs almost nothing.

### On a NAS — QNAP Container Station, Synology Container Manager

No terminal needed. Open Container Station (or Container Manager), create a
new application, and paste
[`deploy/docker-compose.nas.yml`](deploy/docker-compose.nas.yml) as the
YAML. That file is the whole deployment: it pulls the published image for
the NAS's own chip — ARM or Intel — and has no build step, because there is
no source tree on a NAS to build from. Then open port 8080 and choose a
password.

Two things Container Station insists on, neither of which it says out loud:
the application name must be **lowercase** (`tranceautodj`, not
`TranceAutoDJ` — the form stays red until it is), and memory or CPU limits
go in its *Advanced Settings* panel, never in the YAML. Under *Advanced
Settings → Default Web URL Port*, service `autodj` and port `8080` give you a
one-click link to the dashboard from the Applications list.

Leave the visual generator off (it is not in that file, and it is off by
default in the full one). Build blocks on a desktop instead — a ten-minute
block is an hour or two on a small ARM chip, and sixty of them is most of a
week — or take the starter pack below. Copy the finished `.ts` files and
their `.json` sidecars into the volume under `blocks/`; they are adopted the
next time the **Visuals** page loads.

**Memory, honestly measured.** The broadcast itself runs in about 330 MB.
Composing a five-minute track peaks at roughly 750 MB on top of that — it
was 2.1 GB before the mixing buses were moved to single precision, which
would not have fit at all. On a 2 GB NAS that is the broadcast plus one
track being composed, with little to spare; if it is tight, compose
three-minute tracks, or compose on a desktop and copy the MP3s into
`music/`. Container Station will not accept memory or CPU limits inside the
compose file — it wants them in its own *Advanced Settings* — so the NAS
file carries none; set them there if you want a ceiling.

### First run, in order

1. **Add music — or have the station write its own.** Drag files onto
   **add music** on the **Audio** page, or press *choose files*; anything the
   decoder cannot use is refused before the upload starts rather than after
   it. Or skip the files entirely: **compose** builds trance from arithmetic
   and puts it straight in the library — see [Composing music](#composing-music). Copying files into the volume under
   `music/` and pressing *Rescan library* does the same job — it reports what
   it found, so you can see the ones you copied in arrive. Either way they are
   tagged with `mutagen` and an "All tracks" playlist is kept in step with the
   library automatically.

   **choose folder** imports a whole folder at once and makes a playlist of
   the same name, which is usually the name you were going to type anyway.
   Only music goes in: cover art, image subfolders and anything else are left
   behind, and a folder imported twice tops the list up rather than filling
   the library with second copies.

2. **Arrange the lists.** On the **Audio** page, *create* names as many
   playlists as you want and pick their tracks with **tracks**. The station
   plays either **one list, looping** — the one marked active — or **the
   rotation, in order**: the lists you put in the rotation, top to bottom,
   then round again. The ↑ ↓ buttons set that order and it takes effect
   immediately.

   The two are independent on purpose: you can build and arrange a rotation
   while a single list is on air, and swap over when it is ready. In a
   rotation the file order *is* the play order, so Liquidsoap reads straight
   through and any shuffling a list asks for happens when the file is written
   — re-saving the rotation deals a fresh hand.

   Every row in the library has a **remove** button, which deletes the file as
   well: the library is a view of the music folder, and a removed track that
   kept playing would be worse than confusing. A file that disappears from
   under the application is marked *missing* rather than dropped, so that a
   mount coming back does not cost you your playlists; **Remove missing**
   forgets those entries once you have decided they are gone for good.

   Changing the whole repertoire is a normal thing to need, so it is not
   eighty-nine clicks: **Empty library** takes every track and every file
   (it asks you to type `DELETE`, and refuses if the count changed since you
   looked), **delete + music** on a playlist takes the list and the music only
   it holds, and **Remove empty lists** sweeps up whatever is left holding
   nothing. Each one tells you where the next step is instead of stopping.
3. **Start the audio.** **Audio → Start audio** brings up Icecast and
   Liquidsoap. The page turns green when there is really sound on the mount,
   which is not the same as the processes being alive.
4. **Fill the visual pool.** Press **Generate blocks now** on the **Visuals**
   page for a few, or start the generator (below) and leave it. The stream
   needs at least one block; twelve is the default target.
5. **Watch it first.** **Stream → monitor → Start monitor** plays exactly what
   would be sent: the same blocks, the same audio, the same encoder settings,
   muxed for a browser instead of for YouTube. It uses the same pipe as the
   broadcast, so the two cannot run at once — which is the point of having it.
   It starts with sound: pressing the button is the user gesture that lets the
   browser play it.
6. **Paste the stream key** on the **Stream** page.
7. **GO LIVE.**

After that the station looks after itself. If the audio was on air, or the
broadcast was live, when the process last stopped, both come back on their own
the next time it starts — after a crash, a `docker compose up`, or a host
reboot. Only intent is restored: anything you stopped on purpose stays
stopped. Set `app.resume_on_start: false` to disable it.

To build by hand:

```bash
docker build -f deploy/Dockerfile -t trance-autodj:latest .
docker run -d --name trance-autodj -p 8080:8080 -v autodj-data:/data trance-autodj:latest
```

Check the environment without starting anything:

```bash
docker run --rm trance-autodj:latest preflight
```

Fill the visual pool. The generator is a separate process on purpose — it runs
at `nice 19` and is never on the live path. It is off by default:

```bash
docker compose --profile generator up -d
```

It is a service rather than `docker compose run` so that it stops and is
removed with everything else. A detached one-off run is not part of the
project: `docker compose down` leaves it running, and because it still holds
the data volume open, `down -v` keeps the volume as well — which looks exactly
like a reset that did not happen.

Or build a few blocks on demand from the **Visuals** page, which also lets you
watch them: blocks are MPEG-TS, which browsers cannot play, so the preview
endpoint remuxes to fragmented MP4 with `-c copy` — a container change, not a
re-encode.

**GPU in Docker.** Needs the native Docker engine plus
`nvidia-container-toolkit`, then `--gpus all`. Docker Desktop on *Linux* runs
containers inside its own VM and cannot see the GPU at all; on Windows it can,
through WSL2.

### Native (Debian/Ubuntu)

```bash
sudo apt install ffmpeg liquidsoap icecast2 python3-venv
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
python scripts/preflight.py
python -m app.main
```

For a permanent installation, the three systemd units in `deploy/systemd/`
encode the CPU policy in the only place that actually enforces it: the
generator is `Nice=19` with `CPUQuota=150%` and idle I/O, while the streamer
gets `Nice=-5` and `OOMScoreAdjust=-500` because the broadcast is the product.

## Configuration

Everything lives in `config/config.yaml` (start from
`config/config.example.yaml`). There are no hardcoded values in the logic.

Any setting can be overridden by an environment variable — `TAD_` prefix,
`__` between levels — which is what makes the container configurable without
a config file at all:

```bash
TAD_APP__PORT=9000
TAD_AUDIO__CROSSFADE__DURATION_S=18
TAD_VIDEO__ENCODER=libx264
TAD_STREAM__VIDEO_MODE=copy
```

Precedence: defaults → `config.yaml` → environment. Environment always wins,
so a value pinned by the operator cannot be overruled from the web UI.

### Settings worth knowing

| Setting | Default | Why |
|---|---|---|
| `audio.crossfade.duration_s` | `14` | Tuned for trance. Adjustable 4–30 s. |
| `audio.crossfade.beat_aligned` | `false` | Round each crossfade to whole bars of the outgoing track. |
| `audio.analysis.enabled` | `true` | Measure tempo when a file is scanned. |
| `audio.loudness.target_lufs` | `-14` | YouTube's own target; matching it avoids their re-processing. |
| `visual.block.duration_s` | `600` | Ten-minute blocks: long enough to be cheap to join, short enough to vary. |
| `visual.pool.max_disk_gb` | `50` | Least-recently-played blocks are deleted above this. |
| `visual.sources.ai_ratio` | `0.0` | Procedural only. Raise it after downloading a model. |
| `stream.video_mode` | `copy` | The whole point. `reencode` is the escape hatch. |
| `cpu.generator_quota_pct` | `150` | 1.5 cores for generation, enforced by systemd. |

### Naming the broadcast

RTMP carries no title. What goes up the wire is video and audio; the name, the
description and everything else a viewer sees belong to a *broadcast* inside
YouTube. Left alone, each session is called whatever that stream was called
last time.

To name it from here, **Stream → youtube title**: set a template, connect the
account once, and every broadcast is named as it starts.

```
Trance AutoDJ 24/7 — {datetime}     →   Trance AutoDJ 24/7 — 10/09/2026 06:49
```

`{date}`, `{time}` and `{datetime}` are filled in at the moment of going live,
in `app.timezone` — the container clock is UTC, and a broadcast that starts at
03:00 in Athens should say 03:00.

Connecting needs your own Google Cloud credentials, because the quota spent is
yours: create a project, enable the **YouTube Data API v3**, and make an OAuth
client of type **TV and Limited Input**. Paste the client ID and secret, press
*connect account*, and approve the code on a phone — there is no redirect URL
to configure and nothing has to be reachable from outside. The refresh token
is kept in the same encrypted store as the stream key and is never logged.

Cost is a rounding error: one broadcast lookup (1 unit) and one rename (50) per
session, against a default of 10,000 a day. Naming every *track* would be a
different matter — at seven minutes a track that is over 10,000 units a day on
its own, which is why the title is set once per broadcast and not per song.

Nothing here can stop a broadcast. An expired token, a quota refusal or a
network failure costs the title and is logged; the station keeps playing.

### Coming back by itself

The application restores what it was doing: if the audio was on air, or the
broadcast was live, both come back on the next start, and the broadcast is
named again with the *new* start time. That much is tested.

Whether the machine gets that far is the host's job, and it differs:

| | comes back after a reboot |
|---|---|
| Docker Engine on a VM or VPS | Yes — `systemctl enable docker`, and `restart: unless-stopped` does the rest |
| Native install | Yes — the systemd units are `WantedBy=multi-user.target` |
| Docker **Desktop** | **No, until you enable it**: `systemctl --user enable docker-desktop` |

Docker Desktop does not start itself at login, so a container with a perfect
restart policy still waits for someone to open the app. Worth knowing before
relying on it for a 24/7 station.

One more thing that surprises people, because it is deliberate: `docker stop`
and `docker kill` do **not** trigger the restart policy. Docker treats them as
an operator taking the container down on purpose. Crashes and host reboots do
trigger it.

### The YouTube stream key

1. Open <https://studio.youtube.com> → **Create** → **Go live**.
2. Choose **Streaming software** (not the webcam tab).
3. Copy the **Stream key**. It looks like `xxxx-xxxx-xxxx-xxxx-xxxx`.
4. Paste it into **Settings** in the dashboard, or set
   `TAD_STREAM__YOUTUBE__STREAM_KEY`.

The key is encrypted at rest with a key file that is mode `0600`, is masked in
the UI, and is stripped from every log line by a filter that runs on all
handlers — including the raw output of ffmpeg, which prints its command line.

Set `TAD_MASTER_KEY` if you rebuild the container and want the existing
encrypted store to stay readable:

```bash
docker run --rm trance-autodj:latest python -m app.core.security generate-key
```

## Troubleshooting

**The dashboard says `libx264` but the machine has an NVIDIA card.**
Read the reason next to it in `/api/system/capabilities`. Usually the
container cannot see the GPU (see *GPU in Docker* above), or ffmpeg was built
without NVENC.

**The stream stutters at the joins between blocks.**
This was the main technical risk in the design, and it has been measured
rather than assumed. Three independently built blocks were concatenated as
raw bytes and remuxed with `-c:v copy`:

```
3 blocks x 6s                     expected 18.0s / 540 frames
read naively (no -fflags +genpts) 540 frames, duration reported as 6.0s
remuxed with -fflags +genpts      540 frames, 18.000s, decoded with no errors
piped on to FLV (what YouTube gets) 540 frames, 18.067s, decoded with no errors
```

**No frames are lost.** ffmpeg logs `Packet corrupt` and `timestamp
discontinuity` once per join — that is inherent to concatenating MPEG-TS,
because the last PES packet of a file is left unterminated — and recovers from
both. Adding `-mpegts_flags +resend_headers` or `-flush_packets` made no
measurable difference, so the encode profile stays simple.

The one thing that *does* matter: the reader must use **`-fflags +genpts`**.
Without it the timestamps of the second block are not rebased and a naive
consumer sees only the first block's duration. The streamer command includes
it.

If you still see stutter on a specific setup, set
`stream.video_mode: reencode` — at the cost of a continuous x264 encode.

**`The data directory /data is not writable`.**
The container runs as uid 1000. Either `chown -R 1000:1000` the host directory
or run with `--user "$(id -u):$(id -g)"`.

**Named pipes fail the preflight check.**
The state directory must be on a filesystem that supports FIFOs. A Windows
bind mount or a FAT/NTFS volume will not work; use a Docker volume.

**Nothing is generated and the pool stays empty.**
Check `cpu.load_threshold_factor`: the generator waits while the 1-minute load
average is above `cpu_count × 0.7`. On a busy machine it will wait a long time,
by design.

## Roadmap

- [x] **Skeleton** — config, logging, database, capability detection, preflight, Docker, systemd
- [x] **Procedural visuals** — six generators, animator, block builder, pool management, validation
- [x] **Audio engine** — generated Liquidsoap config, playlists, crossfade, Icecast, now-playing
- [x] **Streamer** — feeder, ffmpeg, watchdog, and the measured `-c:v copy` CPU cost
- [x] **Dashboard** — uploads, playlists, visual grid, settings, live logs, authentication
- [x] **AI visuals** — CUDA / OpenVINO / stable-diffusion.cpp backends, model fetching, prompt presets
- [x] **Policy** — load throttling, quiet hours, pool quotas, systemd units, 304 tests

- [x] **Beat-aware crossfades** — offline tempo detection, crossfades in whole bars
- [x] **Audio-reactive overlay** — optional, measured, and honestly expensive
- [x] **1080p** — builds and validates; retires the existing pool, so it is a decision

The GPU path is written but untested. See [Measurements](#measurements).

## How it fits together

One container, one process, four supervised children:

```
python -m app.main
├── icecast2          started, watched, restarted   (audio transport)
├── liquidsoap        started, watched, restarted   (playlist + crossfade)
├── ffmpeg            started, watched, restarted   (the broadcast)
└── feeder thread     writes block bytes into a FIFO
```

The application owns all of them so the dashboard can start and stop the
broadcast. The alternative — a container each, plus the Docker socket handed
to the web process — is more moving parts and a far larger blast radius for a
single-operator system.

The **visual generator** is the exception: it runs as a separate process
(`docker compose --profile generator up -d`, or its own systemd unit) at
`nice 19` under a CPU quota. If it dies, the broadcast keeps playing whatever
is already in the pool. That separation is the whole architecture.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check app tests scripts
black --check app tests scripts
```

Around five hundred tests, no network and no GPU required. The ones that need ffmpeg are
marked `slow` and skip themselves when it is missing.

Everything runs inside the container too, which is how it was developed:

```bash
docker run --rm -v "$PWD":/src -w /src trance-autodj:latest \
  sh -c 'pip install -q pytest httpx && python -m pytest'
```

## Measurements

Everything below was measured on the machine this was built on (AMD Ryzen 7
2700X, 16 threads, no GPU visible to the container), not estimated.

### The broadcast costs almost nothing, and keeps costing it

Nineteen minutes of a live 1280×720 broadcast to a local RTMP listener, on a
fully loaded station — 89 tracks in seven playlists, 60 blocks in the pool —
with every process sampled from `/proc`. The mean, and the range:

```
                             mean          range
ffmpeg (the broadcast)       0.090 cores   0.086 - 0.096   ← what -c:v copy buys
liquidsoap (MP3 320k)        0.125 cores   0.088 - 0.176
icecast                      0.003 cores   0.003 - 0.003
python (dashboard + feeder)  0.010 cores   0.005 - 0.014
generator (pool full)        0.000 cores
────────────────────────────────────────────────────────
TOTAL                        0.228 cores of 16   (0.191 - 0.286)
```

Liquidsoap's range is the widest because its work is not flat: the top of it
is a crossfade, where two tracks are decoded and mixed at once.

The broadcast is under a tenth of a core because ffmpeg never touches the
video: it copies bytes. Over that run it delivered **38,414 frames at 30.0
fps, speed 1.00, zero dropped and zero duplicated**, with no watchdog restart.

Memory, and why the obvious number is the wrong one. `docker stats` and the
cgroup's `memory.peak` reported **6105 MB** for the generator — its whole
limit. Split into what it actually is:

```
                        process memory   file cache (reclaimable)
broadcast container        322 MB              118 MB
generator (idle)            96 MB             1897 MB
```

The peak was page cache from writing 14 GB of blocks, which the kernel keeps
because the memory is free and drops the moment anything wants it. The real
requirement is the left-hand column: **liquidsoap 264-309 MB, python 115-120
MB, ffmpeg 66-68 MB, icecast 17 MB** — about 550 MB for the broadcast, plus
roughly 300 MB for the generator while it is building.

That cache is doing work, not sitting idle: over the whole run neither the
feeder nor ffmpeg performed a single physical disk read, because the blocks
were still in it. On a machine with less RAM the same reads happen for real,
at the video bitrate — about 460 kB/s.

Network is the one cost that does not shrink: **4.13 Mbit/s** measured at the
interface for a 3707 kbit/s stream, which is 1.73 GB an hour, or about 1.25 TB
a month running continuously.

The flat range matters more than the mean. An earlier version of this
measurement ran for sixty seconds and reported 0.289 cores; that number was
true and useless, because the process it measured was two minutes old. The
same stack left running for an hour climbed to a full core and 2.3 GB. What
fixed it is one line of operator ordering — see below — and what caught it was
measuring for long enough to see a trend rather than a snapshot.

Liquidsoap rises to about 300 MB in the first ten minutes and then holds
there, moving by a few MB either way rather than climbing.

### The monitor is the broadcast, not an impression of it

The preview and the live stream are built from one function: same inputs, same
`-c:v copy`, same AAC bitrate and sample rate. Only the container differs —
fragmented MP4 to the browser instead of FLV to YouTube — and a test asserts
that every codec flag matches, because a preview encoded differently from the
thing it previews is worse than none: it invites you to trust it about
something it cannot know.

Measured from the browser: `readyState 4`, 1280×720, playing, no error, with
the picture and the live audio arriving together.

Two things about it are less obvious than they look.

**It cannot run while you are live**, because it holds the same FIFO. That is
the correct behaviour rather than a limitation — it is what you look at
*before* going live — and going live stops it rather than failing.

**A browser that goes away does not say so.** Neither a failing write nor
`is_disconnected()` reports it: the kernel keeps accepting writes into a
buffer that will never drain, so the response generator sits inside its
`yield` and never runs another check. The page therefore says it is still
watching, every five seconds, and a thread outside the request stops the
monitor when it goes quiet. Without that, closing a tab left ffmpeg running —
and holding the pipe that going live needs.

### The signal chain order is load-bearing

`normalize` placed upstream of `cross` leaks memory and CPU without bound.
Neither operator does this on its own — it is the pair. Two scripts differing
only in where the loudness normaliser sits, sampled from `/proc` for thirty
minutes:

```
                          RSS start -> end     CPU
normalize before cross    175 -> 872 MB        0.08 -> 0.38 cores, still climbing
normalize after cross     166 -> 191 MB        0.08 cores, flat
```

Left alone, the first one passes 2 GB and pins a full core — one whole
Liquidsoap audio thread, after which it has no headroom left to keep up with
real time. That is a broadcast that dies quietly some hours after you stop
watching it.

Isolating it took one container per operator and, crucially, one per *pair*:

```
cross alone                189 -> 128 MB    0.04 cores
normalize alone            144 ->  87 MB    0.03 cores
ReplayGain alone           257 -> 201 MB    0.03 cores
cross + ReplayGain         226 -> 276 MB    0.05 cores
cross + normalize          201 -> 1586 MB   0.99 cores
```

Per-track gain (`normalize_track_gain`) stays in front of the crossfade, where
it belongs — both sides of a transition have to match each other before they
overlap — and was measured not to leak there. Only the final broadcast
normaliser moved. Two tests pin the order; without them nothing would.

The move is also the better signal chain, and it costs nothing audibly: over
the same recording the output moved from -15.9 to -15.3 LUFS against a -14
target, with identical peaks.

### The stream survives block joins

Streamed to a local RTMP listener for 150 seconds, crossing about ten block
boundaries:

```
150.1 s received · 4502 of 4503 frames · H.264 1280×720 + AAC
every frame decoded without error · no silence anywhere in the audio
speed 1.01× · 0 dropped frames · watchdog stable
```

### The crossfade really overlaps

Four test tracks at different frequencies, captured from the Icecast mount and
analysed by spectrum. During a transition both tones are present, and the
handover is smooth and monotonic over the configured six seconds:

```
 8.5s   t2 1.00   t3 0.31      ← the next track fades in
 9.5s   t2 1.00   t3 0.81
10.5s   t2 0.88   t3 1.00      ← the outgoing one fades out
12.5s   t2 0.42   t3 1.00
13.5s   t2 0.03   t3 1.00
```

### Generation is slow, and that is the design

At 640×360 (the default half-scale render), per frame:

| generator | ms/frame | vs real time |
|---|---|---|
| flowfield | 8 | 4.1× |
| reaction_diffusion | 22 | 1.5× |
| plasma | 24 | 1.4× |
| waves | 31 | 1.1× |
| tunnel | 38 | 0.9× |
| domainwarp | 88 | 0.4× |

Mixed evenly that is ~35 ms/frame, so a 10-minute block takes roughly **11
minutes of one core**, plus about a minute to encode. It runs at `nice 19`
under a CPU quota and waits when the machine is busy, so it takes as long as
it takes — nobody is waiting for it.

### The audio-reactive overlay is expensive

It is off by default, and this is why. The same stream, measured the same way:

```
-c:v copy, no overlay        0.078 cores    speed 1.01x
overlay + re-encode          2.991 cores    speed 0.98x
```

Thirty-eight times the cost, and *still* slightly under real time on a
16-thread machine. It also forces `stream.video_mode: reencode`, which the
configuration refuses to let you skip. Verified to draw correctly — on a black
input the strip lights up and reacts to the audio while everything above it
stays exactly black — but treat it as a feature you turn on knowing the price.

### 1080p

A 1920×1080 block builds and passes validation (High profile, level 4.1,
yuv420p). Changing the resolution changes the profile fingerprint, so every
existing block is retired by the pool as unplayable — correctly, since they
cannot be copied into the same stream. Expect to rebuild the pool, and budget
about 2.3× the render time per block.

### What is *not* measured

**The GPU path.** `h264_nvenc` and the CUDA image backend are written, and the
detection around them is tested, but neither has run on real hardware: Docker
Desktop on Linux cannot see the host GPU, and the reference VM has none. Both
degrade to the CPU path with a logged reason, which *is* tested — but treat
"works on a GPU" as unverified until you have run it.

## Music rights

**You are responsible for broadcasting only music you own or are licensed to
broadcast.** YouTube's Content ID will detect commercial trance releases and
may mute, block, or place a claim on your stream, and repeated claims can
affect the channel itself. This tool does not circumvent anything, and it has
no mechanism to do so. Use your own productions, licensed libraries, or tracks
released under terms that permit broadcast — or press **compose** and let
the station make music that nobody can claim, because nobody recorded it.

## Licence

MIT — see [LICENSE](LICENSE).

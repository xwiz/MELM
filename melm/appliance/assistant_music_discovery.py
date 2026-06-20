"""Music discovery and download for MELM Assistant OS."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MusicDiscoveryResult:
    status: str  # "found", "not_found", "downloaded", "error"
    title: str = ""
    path: str = ""
    error: str = ""


class MusicDiscoverer:
    """Searches local media inventory and optionally downloads music."""

    def __init__(self, ytdlp_command: str = "", media_dir: str = ""):
        self.ytdlp_command = ytdlp_command
        self.media_dir = media_dir

    def search_inventory(self, query: str, store: Any | None) -> list[dict[str, Any]]:
        """Search media inventory for items matching query."""
        if store is None:
            return []
        inventory = store.load_inventory("media")
        query_lower = query.lower()
        results = []
        for item_id, item in inventory.items():
            title = item.get("title", "").lower()
            tags = [str(t).lower() for t in item.get("tags", ())]
            if query_lower in title or any(query_lower in t for t in tags):
                results.append(item)
        return results

    def offer_search(self, query: str) -> str:
        """Return a user-facing prompt to ask about downloading."""
        return f"I don't have any '{query}' music yet. Would you like me to look for some?"

    def download(self, query: str) -> MusicDiscoveryResult:
        """Download music matching query using yt-dlp."""
        if not self.ytdlp_command:
            return MusicDiscoveryResult(status="error", error="yt-dlp not configured")
        if not self.media_dir:
            return MusicDiscoveryResult(status="error", error="media_dir not configured")
        media_path = Path(self.media_dir)
        media_path.mkdir(parents=True, exist_ok=True)
        search_query = f"ytsearch:{query} audio"
        output_template = str(media_path / "%(title)s.%(ext)s")
        try:
            cmd = shlex.split(self.ytdlp_command) + [
                search_query,
                "-x", "--audio-format", "mp3",
                "-o", output_template,
                "--print", "filename",
                "--no-playlist",
                "--max-downloads", "1",
            ]
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if result.returncode == 0:
                filename = result.stdout.strip().split("\n")[-1].strip()
                return MusicDiscoveryResult(
                    status="downloaded",
                    title=Path(filename).stem,
                    path=filename,
                )
            return MusicDiscoveryResult(
                status="error",
                error=f"yt-dlp exited with code {result.returncode}: {result.stderr[:200]}",
            )
        except FileNotFoundError:
            return MusicDiscoveryResult(status="error", error="yt-dlp not found on PATH")
        except Exception as exc:
            return MusicDiscoveryResult(status="error", error=str(exc))

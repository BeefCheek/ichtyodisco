"""Push the rendered overlay to YouTube / Twitch / any RTMP(S) server (or a file) with ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from aquarium.config import StreamConfig

log = logging.getLogger(__name__)


def build_ffmpeg_cmd(cfg: StreamConfig) -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cfg.width}x{cfg.height}",
        "-r", str(cfg.fps), "-i", "pipe:0",
    ]
    if cfg.silent_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    gop = str(cfg.fps * 2)  # keyframe every 2 s (YouTube/Twitch recommendation)
    cmd += ["-c:v", cfg.encoder, "-pix_fmt", "yuv420p", "-b:v", cfg.bitrate,
            "-maxrate", cfg.bitrate, "-bufsize", _double(cfg.bitrate), "-g", gop,
            "-keyint_min", gop]
    if cfg.encoder == "libx264":
        cmd += ["-preset", cfg.preset, "-tune", "zerolatency", "-sc_threshold", "0"]
    if cfg.silent_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    url = cfg.url
    if url.startswith(("rtmp://", "rtmps://")) or url.endswith(".flv"):
        cmd += ["-f", "flv"]
    cmd.append(url)
    return cmd


def _double(bitrate: str) -> str:
    num = "".join(ch for ch in bitrate if ch.isdigit() or ch == ".")
    unit = bitrate[len(num):]
    return f"{float(num) * 2:g}{unit}"


class FFmpegStreamer:
    """Feeds frames to ffmpeg at a *constant* frame rate.

    RTMP servers expect a steady cadence; the analysis loop does not provide one (inference
    time varies). A pacing thread re-sends the latest frame every 1/fps so the stream stays
    smooth even when the pipeline briefly slows down, and restarts ffmpeg if it dies
    (network hiccup).
    """

    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.frames_sent = 0
        self.restarts = 0

    def start(self) -> None:
        if not self.cfg.url:
            raise ValueError("stream.url is empty (rtmp://... or output file)")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtmp-streamer")
        self._thread.start()

    def push(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._latest = frame

    def _spawn(self) -> subprocess.Popen[bytes]:
        cmd = build_ffmpeg_cmd(self.cfg)
        log.info("starting ffmpeg: %s", " ".join(c if "live" not in c else "<url>" for c in cmd))
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _loop(self) -> None:
        period = 1.0 / self.cfg.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                frame = self._latest
            if frame is None:
                time.sleep(0.05)
                next_t = time.monotonic()
                continue
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:
                    self.restarts += 1
                    log.warning("ffmpeg exited (code %s), restarting in 2 s", self._proc.returncode)
                    time.sleep(2.0)
                self._proc = self._spawn()
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(frame.tobytes())
                self.frames_sent += 1
            except (BrokenPipeError, OSError):
                self._proc = None if self._proc.poll() is not None else self._proc
                continue
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                next_t = time.monotonic()
        self._close()

    def _close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self._proc.kill()
            self._proc = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None

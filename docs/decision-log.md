## Recent Decisions

### Video capture (Dec 2025)

- **What we delivered**
  - Built a `WebcamCapture` wrapper around `cv2.VideoCapture`.
  - Added a dedicated thread so capture never blocks the downstream pipeline.
  - Introduced a thread-safe circular buffer (deque + lock) to decouple capture cadence from consumption cadence.
  - Added automatic reconnect attempts whenever `read()` fails or the webcam goes offline.
  - Exposed configurable capture vs inference resolutions (4K native, 640×360 default for ML) plus a resizing helper.
  - Added target FPS configuration and a rolling `fps_actual` metric computed from capture timestamps.

- **Options discussed**
  - *Single-thread + direct `cap.read()` in the main loop*: rejected because it stalled inference and made FPS control harder.
  - *Unbounded queue*: rejected to avoid memory leaks in backlog scenarios; a bounded deque keeps only the frames that matter.
  - *Caller-managed reconnect logic*: rejected in favor of handling hardware state inside the capture thread.

- **Why this approach**
  - Threading keeps a steady feed for ML even when downstream workload spikes.
  - The bounded buffer protects memory and caps latency.
  - Internal auto-reconnect increases robustness without leaking hardware concerns elsewhere.
  - Separate capture/inference resolutions preserve rich data for recording while keeping inference fast.
  - Rolling FPS tracking gives observability into camera health vs theoretical frame rate.

### Explanatory notes

- **deque (collections.deque)**
  Double-ended queue optimized for O(1) inserts/removals on both ends. Setting `maxlen` turns it into a circular buffer that automatically discards older entries, ideal for keeping just the latest frames without unbounded memory growth. Random access is still possible (O(n)), and a separate lock is required for safe multi-threaded use.

- **Inference resolution choice**
  Defaulting to 640-wide frames balances latency and detail for convolutional models (similar to YOLO defaults). Larger resolutions (e.g., 960 or 1280 wide) remain configurable when more spatial detail is required, acknowledging the roughly quadratic cost of extra pixels.

- **FPS monitoring**
  `fps_actual` is computed from the timestamps of the last N captured frames, providing a rolling measurement that highlights drops due to disconnects or pipeline backpressure without depending on downstream processing time.

### Depth, behaviour analytics and livestream (Nov 2026)

- **What we delivered**
  - Full pipeline `capture → YOLO/ByteTrack → async monocular depth → analytics → overlay → dashboard / RTMP / OSC / SQLite`, behind one CLI (`ichtyodisco`).
  - Monocular depth through Hugging Face `transformers` (Apple Depth Pro, Depth Anything V2), normalised to the tank volume (`z ∈ [0 front glass, 1 back wall]`).
  - An analytics engine that learns the tank: circadian baselines, adaptive swim-state thresholds, resident re-identification, social metrics, occupancy maps, rule-based events and a "mood".
  - Zero-dependency web dashboard (stdlib HTTP, MJPEG, Server-Sent Events), an OBS overlay widget, and an ffmpeg RTMP pusher.
  - A synthetic 3D aquarium (`--demo`) that feeds ground-truth tracks and depth, used for demos and the test suite.

- **Options discussed**
  - *Depth on every frame*: rejected. Depth Pro is ~0.3 s/frame on a datacentre GPU; fish change depth slowly, so depth runs on its own thread at a capped rate on the newest frame (older pending frames are dropped, never queued).
  - *Absolute metric depth*: rejected for now. Refraction through the glass and scale ambiguity make metres meaningless; robust near/far percentile planes, smoothed over time, give a stable relative tank depth.
  - *Fixed thresholds for "agitated", "resting", "sprint"*: rejected. Every tank (species, size, lighting) is different. Thresholds are z-scores against per-hour baselines and quantiles of the tank's own speed distribution, all with time-based exponential forgetting.
  - *Count-based running stats*: rejected in favour of time-based decay, so the learnt model does not depend on FPS or on how long the process ran.
  - *FastAPI/websockets for the dashboard*: rejected to keep dependencies minimal; stdlib `ThreadingHTTPServer` with MJPEG + SSE is enough for a handful of viewers. Public viewers watch the RTMP stream, not the dashboard.
  - *Streaming only via OBS*: kept as an option, but the built-in ffmpeg pusher lets a headless box stream 24/7.
  - *Colour-histogram re-id alone*: too noisy with small fish and occlusions; combined with spatio-temporal gating (a track that appears where a resident vanished < 3 s ago is the same fish) and a strict ratio test otherwise.

- **Why this approach**
  - The hot path (tracking + drawing) stays at camera rate; everything slow is async (depth, SQLite writes, ffmpeg pacing, HTTP).
  - Everything learnt is JSON-serialisable and persisted every minute, so restarts don't reset what the system knows about its residents.
  - The simulator makes the whole stack testable in CI without camera, GPU or model downloads.

- **Licensing note**
  - Apple Depth Pro weights are research-only (Apple ASCL); Depth Anything V2 Small is Apache-2.0, Base/Large are CC-BY-NC-4.0. The default is V2 Small so a public (possibly monetised) stream stays compliant.

### Music: MIDI conductor instead of the old Max patch (Nov 2026)

- **What we delivered**
  - `aquarium.music`: a generative conductor that maps the analytics (mood, activity vs the learnt norm, cohesion, polarisation, per-fish height / depth / speed / swim state, events) to MIDI: pad, bass, drums, one voice per fish, event phrases.
  - Live MIDI output (virtual port or any existing port) and `.mid` recording; `--audio-in` to put the synths' audio in the RTMP stream.

- **Options discussed**
  - *Keep the colour-tracking Max patch*: dropped, too basic; Max stays reachable through OSC.
  - *Code our own synth engine in Python*: rejected for now ("a different project"); MIDI lets real synths (hardware or plugins) do the sound design, which sounds far better for a livestream.
  - *Map positions straight to pitch*: rejected; everything is quantised to the chord / mode so the result is always musical, and the mode follows the tank's mood.

- **Why this approach**
  - Musical decisions run on their own clock thread (16th-note grid, note-offs scheduled on a heap) so video inference jitter never affects timing.
  - Mode changes wait for a mood to be stable for a bar, so the harmony doesn't flap.
  - Every note is released on exit (all-notes-off panic), so synths never hang on a stuck note.

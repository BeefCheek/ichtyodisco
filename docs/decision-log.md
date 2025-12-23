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

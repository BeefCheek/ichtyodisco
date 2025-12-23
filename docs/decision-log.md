## Recent Decisions

### Video capture (Dec 2025)

- **What we delivered**
  - Built a `WebcamCapture` wrapper around `cv2.VideoCapture`.
  - Added a dedicated thread so capture never blocks the downstream pipeline.
  - Introduced a thread-safe circular buffer (deque + lock) to decouple capture cadence from consumption cadence.
  - Added automatic reconnect attempts whenever `read()` fails or the webcam goes offline.

- **Options discussed**
  - *Single-thread + direct `cap.read()` in the main loop*: rejected because it stalled inference and made FPS control harder.
  - *Unbounded queue*: rejected to avoid memory leaks in backlog scenarios; a bounded deque keeps only the frames that matter.
  - *Caller-managed reconnect logic*: rejected in favor of handling hardware state inside the capture thread.

- **Why this approach**
  - Threading keeps a steady feed for ML even when downstream workload spikes.
  - The bounded buffer protects memory and caps latency.
  - Internal auto-reconnect increases robustness without leaking hardware concerns elsewhere.

### Explanatory notes

- **deque (collections.deque)**
  Double-ended queue optimized for O(1) inserts/removals on both ends. Setting `maxlen` turns it into a circular buffer that automatically discards older entries, ideal for keeping just the latest frames without unbounded memory growth. Random access is still possible (O(n)), and a separate lock is required for safe multi-threaded use.

# Face reference photos

Drop one image per person here, filename = name. The vision sidecar
(`observe.py`) loads each `.jpg` / `.jpeg` / `.png` at startup, extracts
the largest face with OpenCV's Haar cascade, and saves a grayscale
histogram. Each frame from the camera then gets compared against these
histograms; matches above 0.5 correlation are labelled by filename, the
rest are `"Unknown"`.

Conventions:
- Filename → person name. `rajneesh.jpg` → `"name": "Rajneesh"`.
- One clearly-visible frontal face per image. Multi-face images keep the
  largest face only.
- The image files themselves are gitignored (personal photos), but this
  README and the folder are tracked.

To test, start `make vision-up`, look at the camera, and watch the
`visual_context.json` `name` field flip from `"Unknown"` to your name.

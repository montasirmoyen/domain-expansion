# Domain Expansion Web Experience

A browser-only React, MediaPipe, and Three.js adaptation of the original Python computer-vision project. Webcam frames and hand landmarks stay on the user's device.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL, choose **Enter with camera**, and allow webcam access. The **Preview domains** button cycles through every world without requiring a webcam or hand sign.

## Production checks

```bash
npm run lint
npm run build
npm run preview
```

Camera access requires HTTPS in production. `localhost` and `127.0.0.1` work during development.

## Architecture

- `src/domain/gestureRecognition.ts` ports the Python landmark geometry, rule priority, and eight-frame vote smoothing.
- `src/vision/useHandTracking.ts` owns webcam lifecycle and MediaPipe Hand Landmarker inference.
- `src/visuals/DomainCanvas.tsx` contains the neutral world and six procedural domain scenes with whole-world transitions.
- `public/models` and `public/mediapipe` contain the locally served model and WASM runtime.

The scenes use original procedural geometry while following the canonical environments: Unlimited Void's information-filled cosmos; Malevolent Shrine's grotesque shrine; Mahito's clasped-hand lattice; Hakari's pachinko game; Yuta's katana field; and Yuji's remembered hometown and station.

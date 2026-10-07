import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
// Loaded from Google's model storage so the page also works when hosted (e.g. GitHub Pages)
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const STABLE_FRAMES = 4;   // same label this many frames in a row before showing a sticker
const HIDE_FRAMES = 6;     // keep the sticker this many frames after it is lost

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const labelEl = document.getElementById("label");
const barsEl = document.getElementById("bars");
const statusEl = document.getElementById("status");
const thresholdEl = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");
const showSkeleton = document.getElementById("showSkeleton");

thresholdEl.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdEl.value).toFixed(2);
});

// ---------- stickers ----------
let nikeImageUrl = null; // set if web/nike.png exists

function stickerContent(label) {
  switch (label.toLowerCase()) {
    case "nike":
      if (nikeImageUrl) return `<img src="${nikeImageUrl}" alt="Nike">`;
      return document.getElementById("nike-logo").innerHTML;
    case "ok":
      return "👌";
    case "heart":
      return "❤️";
    case "ㅗ":
      return "🖕";
    default:
      return null;
  }
}

// Heart gesture: hearts fly out of the hand toward the viewer (grow + fade)
const HEART_INTERVAL_MS = 120;

function shootHeart(x, y) {
  const el = document.createElement("div");
  el.className = "flying-heart";
  el.textContent = "❤️";
  el.style.left = `${x * 100}%`;
  el.style.top = `${y * 100}%`;
  // random spread so the hearts fan out instead of stacking
  el.style.setProperty("--dx", `${(Math.random() - 0.5) * 360}px`);
  el.style.setProperty("--dy", `${(Math.random() - 0.5) * 260}px`);
  el.style.setProperty("--rot", `${(Math.random() - 0.5) * 60}deg`);
  el.addEventListener("animationend", () => el.remove());
  overlay.appendChild(el);
}

// One sticker slot per hand index
const slots = [0, 1].map(() => ({ el: null, shown: null, candidate: null, streak: 0, missing: 0, lastShot: 0 }));

function updateSlot(slot, label, x, y) {
  if (label && label === slot.candidate) {
    slot.streak++;
  } else {
    slot.candidate = label;
    slot.streak = label ? 1 : 0;
  }

  if (label && slot.streak >= STABLE_FRAMES && stickerContent(label)) {
    slot.missing = 0;
    if (slot.shown !== label) {
      slot.el?.remove();
      slot.el = document.createElement("div");
      slot.el.className = "sticker";
      slot.el.innerHTML = stickerContent(label);
      overlay.appendChild(slot.el);
      slot.shown = label;
    }
  } else if (slot.el && ++slot.missing > HIDE_FRAMES) {
    slot.el.remove();
    slot.el = null;
    slot.shown = null;
  }

  if (slot.el && x != null) {
    slot.el.style.left = `${x * 100}%`;
    slot.el.style.top = `${y * 100}%`;

    const now = performance.now();
    if (slot.shown?.toLowerCase() === "heart" && slot.missing === 0 && now - slot.lastShot > HEART_INTERVAL_MS) {
      slot.lastShot = now;
      shootHeart(x, y);
    }
  }
}

// ---------- classifier (sklearn MLP exported by export_model_web.py) ----------
let model = null;

function predict(features) {
  let x = features;
  model.layers.forEach((layer, i) => {
    const out = layer.b.slice();
    for (let r = 0; r < x.length; r++) {
      const xr = x[r];
      if (xr === 0) continue;
      const row = layer.W[r];
      for (let c = 0; c < out.length; c++) out[c] += xr * row[c];
    }
    const last = i === model.layers.length - 1;
    x = last ? out : out.map((v) => Math.max(0, v));
  });
  if (model.out_activation === "logistic") {
    const p = 1 / (1 + Math.exp(-x[0]));
    return [1 - p, p];
  }
  const max = Math.max(...x);
  const exp = x.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

// Must match gesture_features.landmarks_to_features in Python
function toFeatures(world, handedness) {
  const pts = world.map((p) => [p.x - world[0].x, p.y - world[0].y, p.z - world[0].z]);
  if (handedness === "Left") pts.forEach((p) => { p[0] = -p[0]; });
  const scale = Math.max(...pts.map((p) => Math.hypot(p[0], p[1], p[2])));
  return pts.flat().map((v) => (scale > 0 ? v / scale : v));
}

function buildBars() {
  barsEl.innerHTML = model.classes.map((c) => `
    <div class="bar" data-label="${c}">
      <span>${c}</span>
      <div class="track"><div class="fill"></div></div>
      <span class="pct">0%</span>
    </div>`).join("");
}

function updateBars(probs) {
  barsEl.querySelectorAll(".bar").forEach((bar, i) => {
    const p = probs ? probs[i] : 0;
    bar.querySelector(".fill").style.width = `${p * 100}%`;
    bar.querySelector(".pct").textContent = `${Math.round(p * 100)}%`;
  });
}

// ---------- drawing ----------
function drawHand(landmarks) {
  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ef4444";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------- main loop ----------
let landmarker;
let lastTime = -1;
let fpsFrames = 0, fpsStart = performance.now(), fps = 0;

function loop() {
  if (video.currentTime !== lastTime && video.readyState >= 2) {
    lastTime = video.currentTime;

    // Mirror the frame before detection, exactly like the Python apps (cv2.flip),
    // so handedness and features match the training data.
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const result = landmarker.detectForVideo(canvas, performance.now());
    const hands = result.landmarks ?? [];
    const world = result.worldLandmarks ?? [];
    const handedness = result.handedness ?? result.handednesses ?? [];
    const threshold = Number(thresholdEl.value);

    let firstProbs = null;
    let firstLabel = hands.length ? "Unknown" : "손이 보이지 않음";

    slots.forEach((slot, i) => {
      if (i >= hands.length) {
        updateSlot(slot, null);
        return;
      }
      const probs = predict(toFeatures(world[i], handedness[i][0].categoryName));
      const best = probs.indexOf(Math.max(...probs));
      const label = probs[best] >= threshold ? model.classes[best] : null;

      if (showSkeleton.checked) drawHand(hands[i]);
      const cx = hands[i].reduce((s, p) => s + p.x, 0) / hands[i].length;
      const cy = hands[i].reduce((s, p) => s + p.y, 0) / hands[i].length;
      updateSlot(slot, label, cx, cy);

      if (i === 0) {
        firstProbs = probs;
        firstLabel = label ?? "Unknown";
      }
    });

    labelEl.textContent = firstLabel;
    updateBars(firstProbs);

    fpsFrames++;
    const now = performance.now();
    if (now - fpsStart > 1000) {
      fps = Math.round((fpsFrames * 1000) / (now - fpsStart));
      fpsFrames = 0;
      fpsStart = now;
      statusEl.textContent = `FPS ${fps} · 손 ${hands.length}개 · 모델: ${model.classes.join(", ")}`;
    }
  }
  requestAnimationFrame(loop);
}

async function main() {
  const loading = document.getElementById("loading");
  try {
    model = await (await fetch("model.json")).json();
    buildBars();

    const nike = await fetch("nike.png", { method: "HEAD" }).catch(() => null);
    if (nike?.ok) nikeImageUrl = "nike.png";

    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
    });
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, options("GPU"));
    } catch {
      landmarker = await HandLandmarker.createFromOptions(vision, options("CPU"));
    }

    loading.textContent = "카메라 권한을 허용해 주세요 (주소창 옆 팝업)";
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    loading.classList.add("hidden");
    requestAnimationFrame(loop);
  } catch (err) {
    const hint = err.name === "NotReadableError"
      ? " (다른 프로그램이 카메라를 사용 중입니다. 파이썬 웹캠 창을 닫고 새로고침하세요.)"
      : err.name === "NotAllowedError" ? " (카메라 권한이 거부되었습니다. 주소창 왼쪽 아이콘에서 허용하세요.)" : "";
    loading.textContent = `오류: ${err.message ?? err}${hint}`;
    console.error(err);
  }
}

main();

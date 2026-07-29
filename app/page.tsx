"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  ObjectDetector,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

type FacingMode = "user" | "environment";
type AppStatus =
  | "idle"
  | "camera"
  | "loading"
  | "tracking"
  | "object"
  | "lost"
  | "error";

type Landmark = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
};

type ObjectDetection = {
  boundingBox?: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
  categories: Array<{
    categoryName?: string;
    displayName?: string;
    score?: number;
  }>;
};

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const OBJECT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite";

const LIVING_OBJECT_CATEGORIES = [
  "person",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "potted plant",
];
const LIVING_OBJECT_CATEGORY_SET = new Set(LIVING_OBJECT_CATEGORIES);
const OBJECT_DETECTION_INTERVAL_MS = 300;

const THAI_OBJECT_NAMES: Record<string, string> = {
  person: "คน",
  bicycle: "จักรยาน",
  car: "รถยนต์",
  motorcycle: "รถจักรยานยนต์",
  airplane: "เครื่องบิน",
  bus: "รถโดยสาร",
  train: "รถไฟ",
  truck: "รถบรรทุก",
  boat: "เรือ",
  "traffic light": "ไฟจราจร",
  "fire hydrant": "หัวจ่ายน้ำดับเพลิง",
  "stop sign": "ป้ายหยุด",
  "parking meter": "มิเตอร์จอดรถ",
  bench: "ม้านั่ง",
  bird: "นก",
  cat: "แมว",
  dog: "สุนัข",
  horse: "ม้า",
  sheep: "แกะ",
  cow: "วัว",
  elephant: "ช้าง",
  bear: "หมี",
  zebra: "ม้าลาย",
  giraffe: "ยีราฟ",
  backpack: "กระเป๋าเป้",
  umbrella: "ร่ม",
  handbag: "กระเป๋าถือ",
  tie: "เนกไท",
  suitcase: "กระเป๋าเดินทาง",
  frisbee: "จานร่อน",
  skis: "สกี",
  snowboard: "สโนว์บอร์ด",
  "sports ball": "ลูกบอล",
  kite: "ว่าว",
  "baseball bat": "ไม้เบสบอล",
  "baseball glove": "ถุงมือเบสบอล",
  skateboard: "สเกตบอร์ด",
  surfboard: "กระดานโต้คลื่น",
  "tennis racket": "ไม้เทนนิส",
  bottle: "ขวด",
  "wine glass": "แก้วไวน์",
  cup: "แก้ว",
  fork: "ส้อม",
  knife: "มีด",
  spoon: "ช้อน",
  bowl: "ชาม",
  banana: "กล้วย",
  apple: "แอปเปิล",
  sandwich: "แซนด์วิช",
  orange: "ส้ม",
  broccoli: "บรอกโคลี",
  carrot: "แครอต",
  "hot dog": "ฮอตดอก",
  pizza: "พิซซ่า",
  donut: "โดนัท",
  cake: "เค้ก",
  chair: "เก้าอี้",
  couch: "โซฟา",
  "potted plant": "ต้นไม้กระถาง",
  bed: "เตียง",
  "dining table": "โต๊ะอาหาร",
  toilet: "โถสุขภัณฑ์",
  tv: "โทรทัศน์",
  laptop: "แล็ปท็อป",
  mouse: "เมาส์",
  remote: "รีโมต",
  keyboard: "แป้นพิมพ์",
  "cell phone": "โทรศัพท์มือถือ",
  microwave: "ไมโครเวฟ",
  oven: "เตาอบ",
  toaster: "เครื่องปิ้งขนมปัง",
  sink: "อ่างล้าง",
  refrigerator: "ตู้เย็น",
  book: "หนังสือ",
  clock: "นาฬิกา",
  vase: "แจกัน",
  scissors: "กรรไกร",
  "teddy bear": "ตุ๊กตาหมี",
  "hair drier": "ไดร์เป่าผม",
  toothbrush: "แปรงสีฟัน",
};

const CONNECTIONS: Array<[number, number]> = [
  [0, 7],
  [0, 8],
  [7, 11],
  [8, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [30, 32],
];

const STATUS_COPY: Record<AppStatus, string> = {
  idle: "พร้อมเริ่มกล้อง",
  camera: "เปิดกล้องแล้ว",
  loading: "กำลังเตรียม AI ตรวจจับสิ่งมีชีวิต",
  tracking: "ตรวจพบท่าทาง",
  object: "ตรวจพบสิ่งมีชีวิต",
  lost: "กำลังมองหาคน สัตว์ และต้นไม้",
  error: "เปิดกล้องไม่สำเร็จ",
};

function isAppleMobileDevice() {
  return (
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  );
}

function Icon({
  name,
  size = 24,
}: {
  name:
    | "camera"
    | "flip"
    | "body"
    | "shield"
    | "home"
    | "close"
    | "chevronDown"
    | "chevronUp"
    | "box";
  size?: number;
}) {
  const paths = {
    camera: (
      <>
        <path d="M8.5 7 10 4h4l1.5 3H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h3.5Z" />
        <circle cx="12" cy="13" r="4" />
      </>
    ),
    flip: (
      <>
        <path d="M7 7h10l-2.5-2.5M17 17H7l2.5 2.5" />
        <path d="M19 9a7 7 0 0 1 .3 6M5 15A7 7 0 0 1 5.3 9" />
      </>
    ),
    body: (
      <>
        <circle cx="12" cy="4.5" r="2" />
        <path d="m12 7 0 6m0-3-4-1.5M12 10l4-1.5M12 13l-3 6M12 13l3 6" />
        <circle cx="8" cy="8.5" r=".8" />
        <circle cx="16" cy="8.5" r=".8" />
        <circle cx="9" cy="19" r=".8" />
        <circle cx="15" cy="19" r=".8" />
      </>
    ),
    shield: (
      <path d="M12 3 20 6v5c0 5.1-3.4 8.5-8 10-4.6-1.5-8-4.9-8-10V6l8-3Zm-3 9 2 2 4-5" />
    ),
    home: (
      <>
        <path d="m3 11 9-8 9 8" />
        <path d="M5 10v10h14V10M9 20v-6h6v6" />
      </>
    ),
    close: <path d="m6 6 12 12M18 6 6 18" />,
    chevronDown: <path d="m6 9 6 6 6-6" />,
    chevronUp: <path d="m6 15 6-6 6 6" />,
    box: (
      <>
        <path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4" />
        <rect height="8" rx="1.5" width="8" x="8" y="8" />
      </>
    ),
  };

  return (
    <svg
      aria-hidden="true"
      className="ui-icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      >
        {paths[name]}
      </g>
    </svg>
  );
}

function angle(a: Landmark, b: Landmark, c: Landmark) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const length = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!length) return 180;
  return (Math.acos(Math.max(-1, Math.min(1, dot / length))) * 180) / Math.PI;
}

function classifyPose(points: Landmark[]) {
  if (points.length < 33) return "กำลังอ่านท่าทาง";
  const visible = (index: number) => (points[index]?.visibility ?? 1) > 0.55;
  const required = [11, 12, 15, 16, 23, 24, 25, 26, 27, 28];
  if (!required.every(visible)) return "เห็นร่างกายไม่ครบ";

  const shoulderY = (points[11].y + points[12].y) / 2;
  const handsAbove = points[15].y < shoulderY && points[16].y < shoulderY;
  if (handsAbove) return "ยกแขนขึ้น";

  const shoulderSpan = Math.abs(points[11].x - points[12].x);
  const wristSpan = Math.abs(points[15].x - points[16].x);
  const wristsLevel =
    Math.abs(points[15].y - shoulderY) < 0.13 &&
    Math.abs(points[16].y - shoulderY) < 0.13;
  if (wristsLevel && wristSpan > shoulderSpan * 2.1) return "กางแขน";

  const leftKnee = angle(points[23], points[25], points[27]);
  const rightKnee = angle(points[24], points[26], points[28]);
  if (leftKnee < 135 && rightKnee < 135) return "ย่อเข่า";
  if (leftKnee < 125 || rightKnee < 125) return "ก้าวหรือยกขา";

  return "ยืน";
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const objectCanvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const posePromiseRef = useRef<Promise<PoseLandmarker> | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const objectPromiseRef = useRef<Promise<ObjectDetector> | null>(null);
  const visionPromiseRef = useRef<ReturnType<
    typeof FilesetResolver.forVisionTasks
  > | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);
  const activeRef = useRef(false);
  const facingRef = useRef<FacingMode>("user");
  const mirrorRef = useRef(true);
  const skeletonRef = useRef(true);
  const objectsEnabledRef = useRef(true);
  const previousPointsRef = useRef<Landmark[] | null>(null);
  const lastDetectRef = useRef(0);
  const lastObjectDetectRef = useRef(0);
  const lastObjectFoundRef = useRef(0);
  const fpsWindowRef = useRef({ started: 0, frames: 0 });

  const [active, setActive] = useState(false);
  const [mirror, setMirror] = useState(true);
  const [skeleton, setSkeleton] = useState(true);
  const [objectsEnabled, setObjectsEnabled] = useState(true);
  const [facing, setFacing] = useState<FacingMode>("user");
  const [status, setStatus] = useState<AppStatus>("idle");
  const [poseName, setPoseName] = useState("รอเริ่มตรวจจับ");
  const [fps, setFps] = useState(0);
  const [error, setError] = useState("");
  const [installOpen, setInstallOpen] = useState(false);
  const [standalone, setStandalone] = useState(false);
  const [controlsHidden, setControlsHidden] = useState(false);

  useEffect(() => {
    const nav = navigator as Navigator & { standalone?: boolean };
    const standaloneTimer = window.setTimeout(() => {
      setStandalone(
        window.matchMedia("(display-mode: standalone)").matches ||
          nav.standalone === true,
      );
    }, 0);
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => undefined);
    }
    return () => window.clearTimeout(standaloneTimer);
  }, []);

  const ensureVision = useCallback(() => {
    if (!visionPromiseRef.current) {
      visionPromiseRef.current = FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm",
      ).catch((visionError) => {
        visionPromiseRef.current = null;
        throw visionError;
      });
    }
    return visionPromiseRef.current;
  }, []);

  const ensurePose = useCallback(async () => {
    if (poseRef.current) return poseRef.current;
    if (posePromiseRef.current) return posePromiseRef.current;

    posePromiseRef.current = (async () => {
      const vision = await ensureVision();
      const create = (delegate: "GPU" | "CPU") =>
        PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: POSE_MODEL_URL,
            delegate,
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputSegmentationMasks: false,
        });

      try {
        poseRef.current = await create("GPU");
      } catch {
        poseRef.current = await create("CPU");
      }
      return poseRef.current;
    })();

    try {
      return await posePromiseRef.current;
    } catch (modelError) {
      posePromiseRef.current = null;
      throw modelError;
    }
  }, [ensureVision]);

  const ensureObjectDetector = useCallback(async () => {
    if (objectDetectorRef.current) return objectDetectorRef.current;
    if (objectPromiseRef.current) return objectPromiseRef.current;

    objectPromiseRef.current = (async () => {
      const vision = await ensureVision();
      const create = (delegate: "GPU" | "CPU") =>
        ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: OBJECT_MODEL_URL,
            delegate,
          },
          runningMode: "VIDEO",
          displayNamesLocale: "en",
          maxResults: 12,
          scoreThreshold: 0.3,
          categoryAllowlist: LIVING_OBJECT_CATEGORIES,
        });

      if (isAppleMobileDevice()) {
        objectDetectorRef.current = await create("CPU");
        return objectDetectorRef.current;
      }

      try {
        objectDetectorRef.current = await create("GPU");
      } catch {
        objectDetectorRef.current = await create("CPU");
      }
      return objectDetectorRef.current;
    })();

    try {
      return await objectPromiseRef.current;
    } catch (modelError) {
      objectPromiseRef.current = null;
      throw modelError;
    }
  }, [ensureVision]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const clearObjectCanvas = useCallback(() => {
    const canvas = objectCanvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawPose = useCallback(
    (rawPoints: Landmark[]) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video || !video.videoWidth || !video.videoHeight) return;

      const cssWidth = canvas.clientWidth;
      const cssHeight = canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const targetWidth = Math.round(cssWidth * dpr);
      const targetHeight = Math.round(cssHeight * dpr);
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, cssWidth, cssHeight);

      const prior = previousPointsRef.current;
      const points = rawPoints.map((point, index) => {
        const old = prior?.[index];
        if (!old) return point;
        const currentWeight = 0.48;
        return {
          ...point,
          x: old.x * (1 - currentWeight) + point.x * currentWeight,
          y: old.y * (1 - currentWeight) + point.y * currentWeight,
          z:
            (old.z ?? 0) * (1 - currentWeight) +
            (point.z ?? 0) * currentWeight,
        };
      });
      previousPointsRef.current = points;
      setPoseName(classifyPose(points));

      if (!skeletonRef.current) return;

      const scale = Math.max(
        cssWidth / video.videoWidth,
        cssHeight / video.videoHeight,
      );
      const renderedWidth = video.videoWidth * scale;
      const renderedHeight = video.videoHeight * scale;
      const offsetX = (cssWidth - renderedWidth) / 2;
      const offsetY = (cssHeight - renderedHeight) / 2;
      const pointToCanvas = (point: Landmark) => ({
        x:
          (mirrorRef.current ? 1 - point.x : point.x) * renderedWidth +
          offsetX,
        y: point.y * renderedHeight + offsetY,
      });

      context.save();
      context.lineCap = "round";
      context.lineJoin = "round";
      context.shadowColor = "rgba(57, 231, 255, .72)";
      context.shadowBlur = 10;
      context.strokeStyle = "#39e7ff";
      context.lineWidth = Math.max(3, Math.min(cssWidth, cssHeight) * 0.006);

      CONNECTIONS.forEach(([from, to]) => {
        const first = points[from];
        const second = points[to];
        if (
          !first ||
          !second ||
          (first.visibility ?? 1) < 0.45 ||
          (second.visibility ?? 1) < 0.45
        )
          return;
        const start = pointToCanvas(first);
        const end = pointToCanvas(second);
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      });

      context.shadowColor = "rgba(185, 255, 74, .85)";
      context.shadowBlur = 12;
      points.forEach((point, index) => {
        if ((point.visibility ?? 1) < 0.55 || (index > 0 && index < 7)) return;
        const current = pointToCanvas(point);
        const radius = Math.max(3.5, Math.min(cssWidth, cssHeight) * 0.007);
        context.beginPath();
        context.arc(current.x, current.y, radius, 0, Math.PI * 2);
        context.fillStyle = "#071006";
        context.fill();
        context.lineWidth = Math.max(2, radius * 0.48);
        context.strokeStyle = "#b9ff4a";
        context.stroke();
      });
      context.restore();
    },
    [],
  );

  const drawObjects = useCallback((detections: ObjectDetection[]) => {
    const canvas = objectCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth || !video.videoHeight) return;

    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetWidth = Math.round(cssWidth * dpr);
    const targetHeight = Math.round(cssHeight * dpr);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }

    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    if (!objectsEnabledRef.current) return;

    const scale = Math.max(
      cssWidth / video.videoWidth,
      cssHeight / video.videoHeight,
    );
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const offsetX = (cssWidth - renderedWidth) / 2;
    const offsetY = (cssHeight - renderedHeight) / 2;

    context.save();
    context.lineJoin = "round";
    context.textBaseline = "middle";

    detections.forEach((detection) => {
      const box = detection.boundingBox;
      const category = detection.categories?.[0];
      if (!box || !category) return;

      const sourceX = mirrorRef.current
        ? video.videoWidth - box.originX - box.width
        : box.originX;
      const x = sourceX * scale + offsetX;
      const y = box.originY * scale + offsetY;
      const width = box.width * scale;
      const height = box.height * scale;
      const rawName =
        category.categoryName || category.displayName || "object";
      const translatedName = THAI_OBJECT_NAMES[rawName.toLowerCase()] || rawName;
      const confidence = Math.round((category.score ?? 0) * 100);
      const label = `${translatedName}  ${confidence}%`;
      const fontSize = Math.max(12, Math.min(cssWidth, cssHeight) * 0.022);

      context.shadowColor = "rgba(185, 255, 74, .72)";
      context.shadowBlur = 9;
      context.strokeStyle = "#b9ff4a";
      context.lineWidth = Math.max(2.5, Math.min(cssWidth, cssHeight) * 0.005);
      context.strokeRect(x, y, width, height);

      context.font = `650 ${fontSize}px -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif`;
      const horizontalPadding = fontSize * 0.58;
      const labelHeight = fontSize * 1.8;
      const labelWidth = context.measureText(label).width + horizontalPadding * 2;
      const labelX = Math.max(
        0,
        Math.min(x, Math.max(0, cssWidth - labelWidth)),
      );
      const labelY = y >= labelHeight ? y - labelHeight : y;

      context.shadowBlur = 0;
      context.fillStyle = "rgba(7, 16, 6, .92)";
      context.fillRect(labelX, labelY, labelWidth, labelHeight);
      context.fillStyle = "#d9ff90";
      context.fillText(
        label,
        labelX + horizontalPadding,
        labelY + labelHeight / 2,
      );
    });

    context.restore();
  }, []);

  const runDetection = useCallback(() => {
    const tick = () => {
      if (!activeRef.current) return;
      const video = videoRef.current;
      const pose = poseRef.current;
      const objectDetector = objectDetectorRef.current;
      const now = performance.now();
      let processedObjectFrame = false;

      if (
        video &&
        objectDetector &&
        objectsEnabledRef.current &&
        video.readyState >= 2 &&
        now - lastObjectDetectRef.current >= OBJECT_DETECTION_INTERVAL_MS
      ) {
        lastObjectDetectRef.current = now;
        processedObjectFrame = true;
        try {
          const result = objectDetector.detectForVideo(video, now);
          const livingDetections = (
            result.detections as ObjectDetection[]
          ).filter((detection) => {
            const name =
              detection.categories?.[0]?.categoryName?.toLowerCase() || "";
            return LIVING_OBJECT_CATEGORY_SET.has(name);
          });
          drawObjects(livingDetections);
          if (livingDetections.length > 0) {
            lastObjectFoundRef.current = now;
            setStatus("object");
          }
        } catch {
          // A transient frame decode error is safe to ignore.
        }
      }

      if (
        video &&
        pose &&
        !processedObjectFrame &&
        video.readyState >= 2 &&
        now - lastDetectRef.current >= 50
      ) {
        lastDetectRef.current = now;
        try {
          const result = pose.detectForVideo(video, now);
          const points = result.landmarks?.[0] as Landmark[] | undefined;
          if (points?.length) {
            drawPose(points);
            setStatus("tracking");
          } else {
            clearCanvas();
            previousPointsRef.current = null;
            setPoseName("ยังไม่พบคน");
            setStatus(
              now - lastObjectFoundRef.current < 600 ? "object" : "lost",
            );
          }

          const window = fpsWindowRef.current;
          if (!window.started) window.started = now;
          window.frames += 1;
          if (now - window.started >= 1000) {
            setFps(Math.round((window.frames * 1000) / (now - window.started)));
            fpsWindowRef.current = { started: now, frames: 0 };
          }
        } catch {
          // A transient frame decode error is safe to ignore.
        }
      }
      frameRef.current = requestAnimationFrame(tick);
    };

    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(tick);
  }, [clearCanvas, drawObjects, drawPose]);

  const openStream = useCallback(
    async (mode: FacingMode) => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
    },
    [],
  );

  const stopCamera = useCallback(() => {
    activeRef.current = false;
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    previousPointsRef.current = null;
    fpsWindowRef.current = { started: 0, frames: 0 };
    lastObjectDetectRef.current = 0;
    lastObjectFoundRef.current = 0;
    clearCanvas();
    clearObjectCanvas();
    setFps(0);
    setPoseName("รอเริ่มตรวจจับ");
    setStatus("idle");
    setActive(false);
  }, [clearCanvas, clearObjectCanvas]);

  const startCamera = useCallback(async () => {
    if (activeRef.current) {
      stopCamera();
      return;
    }

    setError("");
    setStatus("camera");
    try {
      await openStream(facingRef.current);
      activeRef.current = true;
      setActive(true);
      setStatus("loading");
      setPoseName("กำลังโหลด AI ตรวจจับสิ่งมีชีวิต");

      let objectReady = false;
      try {
        await ensureObjectDetector();
        objectReady = true;
        if (!activeRef.current) return;
        setPoseName("AI สิ่งมีชีวิตพร้อมแล้ว");
        runDetection();
      } catch {
        setError("AI สิ่งมีชีวิตโหลดไม่สำเร็จ กำลังเปิดโครงร่างแทน");
      }

      if (!activeRef.current) return;
      setPoseName("กำลังโหลด AI โครงร่าง");

      try {
        await ensurePose();
      } catch (poseError) {
        if (!objectReady) throw poseError;
        setError((current) => current || "AI โครงร่างโหลดไม่สำเร็จ");
      }

      if (!activeRef.current) return;
      setPoseName(
        poseRef.current ? "กำลังมองหาคน" : "AI สิ่งมีชีวิตพร้อมใช้งาน",
      );
      setStatus("lost");
      runDetection();
    } catch (cameraError) {
      stopCamera();
      const name =
        cameraError instanceof DOMException ? cameraError.name : "UnknownError";
      if (name === "NotAllowedError") {
        setError("Safari ยังไม่ได้รับอนุญาตให้ใช้กล้อง กรุณาอนุญาตกล้องแล้วลองใหม่");
      } else if (name === "NotFoundError") {
        setError("ไม่พบกล้องที่ใช้งานได้บนอุปกรณ์นี้");
      } else {
        setError("เกิดข้อผิดพลาดขณะเปิดกล้องหรือโหลดระบบตรวจจับ กรุณาลองใหม่");
      }
      setStatus("error");
    }
  }, [
    ensureObjectDetector,
    ensurePose,
    openStream,
    runDetection,
    stopCamera,
  ]);

  const switchCamera = useCallback(async () => {
    const next: FacingMode =
      facingRef.current === "user" ? "environment" : "user";
    facingRef.current = next;
    setFacing(next);
    mirrorRef.current = next === "user";
    setMirror(next === "user");
    if (!activeRef.current) return;

    setStatus("camera");
    try {
      await openStream(next);
      previousPointsRef.current = null;
      lastObjectDetectRef.current = 0;
      runDetection();
    } catch {
      setError("สลับกล้องไม่สำเร็จ อุปกรณ์อาจมีกล้องให้เลือกเพียงตัวเดียว");
    }
  }, [openStream, runDetection]);

  useEffect(
    () => () => {
      activeRef.current = false;
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      poseRef.current?.close();
      objectDetectorRef.current?.close();
    },
    [],
  );

  return (
    <main className="app-shell">
      <header className="app-header">
        <button
          aria-label="คำแนะนำการติดตั้ง"
          className="brand-button"
          onClick={() => setInstallOpen(true)}
          type="button"
        >
          <span className="brand-mark" aria-hidden="true">
            <Icon name="body" size={28} />
          </span>
          <span className="brand-copy">
            <strong>Pose + Objects</strong>
            <small>ตรวจคน สัตว์ และต้นไม้จากกล้อง</small>
          </span>
        </button>

        <div className="privacy-badge" title="ภาพกล้องไม่ถูกอัปโหลด">
          <Icon name="shield" size={19} />
          <span>ประมวลผลบนเครื่อง</span>
          <i aria-hidden="true" />
          <span>เป็นส่วนตัว</span>
        </div>
      </header>

      <section
        className="camera-shell"
        aria-label="พื้นที่ตรวจจับคน สัตว์ และต้นไม้"
      >
        <div className={`camera-stage ${active ? "is-active" : ""}`}>
          <video
            aria-label="ภาพจากกล้อง"
            className={mirror ? "is-mirrored" : ""}
            muted
            playsInline
            ref={videoRef}
          />
          <canvas
            aria-hidden="true"
            className="pose-canvas"
            ref={canvasRef}
          />
          <canvas
            aria-hidden="true"
            className="object-canvas"
            ref={objectCanvasRef}
          />

          {!active && (
            <div className="empty-state">
              <span className="empty-body" aria-hidden="true">
                <Icon name="body" size={74} />
              </span>
              <h1>มองเห็นสิ่งมีชีวิต<br />แบบเรียลไทม์</h1>
              <p>ตีกรอบเฉพาะคน สัตว์ และต้นไม้ พร้อมโครงกระดูก</p>
            </div>
          )}

          <div className={`status-pill status-${status}`}>
            <i aria-hidden="true" />
            <span>{STATUS_COPY[status]}</span>
          </div>

          {active && (
            <div className="pose-card" aria-live="polite">
              <span>ท่าที่เห็น</span>
              <strong>{poseName}</strong>
              {fps > 0 && <small>{fps} FPS</small>}
            </div>
          )}

          {error && (
            <div className="error-toast" role="alert">
              {error}
            </div>
          )}

          <div
            className={`control-dock ${controlsHidden ? "is-collapsed" : ""}`}
          >
            <button
              aria-expanded={!controlsHidden}
              aria-label={
                controlsHidden ? "แสดงแถบเครื่องมือ" : "ซ่อนแถบเครื่องมือ"
              }
              className="dock-visibility-toggle"
              onClick={() => setControlsHidden((value) => !value)}
              type="button"
            >
              <Icon
                name={controlsHidden ? "chevronUp" : "chevronDown"}
                size={20}
              />
              <span>{controlsHidden ? "แสดงเครื่องมือ" : "ซ่อน"}</span>
            </button>

            <button
              className={`start-button ${active ? "is-stop" : ""}`}
              onClick={startCamera}
              type="button"
            >
              <Icon name="camera" size={30} />
              <span>{active ? "หยุดกล้อง" : "เริ่มกล้อง"}</span>
            </button>

            <div className="control-row">
              <button
                aria-pressed={mirror}
                className={mirror ? "is-on" : ""}
                onClick={() =>
                  setMirror((value) => {
                    const next = !value;
                    mirrorRef.current = next;
                    return next;
                  })
                }
                type="button"
              >
                <Icon name="flip" />
                <span>กระจก</span>
              </button>
              <button onClick={switchCamera} type="button">
                <Icon name="camera" />
                <span>{facing === "user" ? "กล้องหน้า" : "กล้องหลัง"}</span>
              </button>
              <button
                aria-pressed={skeleton}
                className={skeleton ? "is-on" : ""}
                onClick={() => {
                  setSkeleton((value) => {
                    const next = !value;
                    skeletonRef.current = next;
                    if (!next) clearCanvas();
                    return next;
                  });
                }}
                type="button"
              >
                <Icon name="body" />
                <span>โครงร่าง</span>
              </button>
              <button
                aria-pressed={objectsEnabled}
                className={objectsEnabled ? "is-on" : ""}
                onClick={() => {
                  setObjectsEnabled((value) => {
                    const next = !value;
                    objectsEnabledRef.current = next;
                    if (!next) clearObjectCanvas();
                    return next;
                  });
                }}
                type="button"
              >
                <Icon name="box" />
                <span>สิ่งมีชีวิต</span>
              </button>
            </div>
          </div>

          {!standalone && (
            <button
              className="install-hint"
              onClick={() => setInstallOpen(true)}
              type="button"
            >
              <Icon name="home" size={18} />
              <span>เพิ่มไปยังหน้าจอโฮม</span>
            </button>
          )}
        </div>
      </section>

      {installOpen && (
        <div
          aria-labelledby="install-title"
          aria-modal="true"
          className="modal-backdrop"
          role="dialog"
        >
          <section className="install-card">
            <button
              aria-label="ปิด"
              className="modal-close"
              onClick={() => setInstallOpen(false)}
              type="button"
            >
              <Icon name="close" />
            </button>
            <span className="install-icon">
              <Icon name="home" size={34} />
            </span>
            <h2 id="install-title">เพิ่มแอปบนหน้าจอโฮม</h2>
            <ol>
              <li>
                เปิดหน้านี้ด้วย <strong>Safari</strong>
              </li>
              <li>
                แตะปุ่ม <strong>แชร์</strong> ด้านล่าง
              </li>
              <li>
                เลือก <strong>เพิ่มไปยังหน้าจอโฮม</strong>
              </li>
            </ol>
            <button
              className="modal-done"
              onClick={() => setInstallOpen(false)}
              type="button"
            >
              เข้าใจแล้ว
            </button>
          </section>
        </div>
      )}
    </main>
  );
}

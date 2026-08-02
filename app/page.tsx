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

type SegmentationMask = {
  width: number;
  height: number;
  getAsFloat32Array: () => Float32Array;
};

type PoseFrame = {
  points: Landmark[];
  mask?: SegmentationMask;
};

type PersonPalette = {
  fill: [number, number, number];
  edge: [number, number, number];
  line: string;
  joint: string;
  shadow: string;
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
const PERSON_CONFIRMATION_SCORE = 0.45;
const PERSON_CONFIRMATION_WINDOW_MS = 800;
const PERSON_MASK_UPDATE_INTERVAL_MS = 100;
const PERSON_MASK_THRESHOLD = 0.5;
const MAX_PEOPLE_OPTIONS = [1, 2, 3, 4, 6, 8] as const;
const MAX_PEOPLE_STORAGE_KEY = "hucam-max-people";
const PERSON_PALETTES: PersonPalette[] = [
  {
    fill: [57, 231, 255],
    edge: [185, 255, 74],
    line: "#39e7ff",
    joint: "#b9ff4a",
    shadow: "rgba(57, 231, 255, .72)",
  },
  {
    fill: [255, 99, 196],
    edge: [255, 176, 226],
    line: "#ff63c4",
    joint: "#ffd1ed",
    shadow: "rgba(255, 99, 196, .72)",
  },
  {
    fill: [255, 180, 55],
    edge: [255, 232, 115],
    line: "#ffb437",
    joint: "#fff0a6",
    shadow: "rgba(255, 180, 55, .72)",
  },
  {
    fill: [167, 139, 250],
    edge: [224, 211, 255],
    line: "#a78bfa",
    joint: "#e8deff",
    shadow: "rgba(167, 139, 250, .72)",
  },
  {
    fill: [74, 222, 128],
    edge: [193, 255, 214],
    line: "#4ade80",
    joint: "#c8ffda",
    shadow: "rgba(74, 222, 128, .72)",
  },
  {
    fill: [96, 165, 250],
    edge: [205, 229, 255],
    line: "#60a5fa",
    joint: "#d4e8ff",
    shadow: "rgba(96, 165, 250, .72)",
  },
  {
    fill: [255, 138, 101],
    edge: [255, 218, 205],
    line: "#ff8a65",
    joint: "#ffe0d6",
    shadow: "rgba(255, 138, 101, .72)",
  },
  {
    fill: [232, 121, 249],
    edge: [251, 210, 255],
    line: "#e879f9",
    joint: "#fbd2ff",
    shadow: "rgba(232, 121, 249, .72)",
  },
];

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

function inferFacingMode(label: string, fallback: FacingMode): FacingMode {
  const normalized = label.toLowerCase();
  if (/front|user|facetime/.test(normalized)) return "user";
  if (/back|rear|environment|ultra|telephoto/.test(normalized)) {
    return "environment";
  }
  return fallback;
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
    | "box"
    | "silhouette"
    | "people"
    | "fullscreen";
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
    silhouette: (
      <>
        <circle cx="12" cy="5" r="2.2" />
        <path d="M8.4 20c.2-3.3.7-5.6 1.7-7.1L9 9.5c1.8-1.1 4.2-1.1 6 0l-1.1 3.4c1 1.5 1.5 3.8 1.7 7.1" />
        <path d="M9.2 10.5 6.5 14M14.8 10.5l2.7 3.5" />
      </>
    ),
    people: (
      <>
        <circle cx="9" cy="7" r="2.5" />
        <circle cx="16.5" cy="8.5" r="2" />
        <path d="M3.8 19c.5-4 2.1-6.2 5.2-6.2s4.7 2.2 5.2 6.2M14.2 13.5c3.5-.8 5.4 1.2 6 4.5" />
      </>
    ),
    fullscreen: (
      <path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5" />
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

function getPoseCenter(points: Landmark[]) {
  const preferred = [11, 12, 23, 24]
    .map((index) => points[index])
    .filter(
      (point): point is Landmark =>
        Boolean(point) && (point.visibility ?? 1) > 0.35,
    );
  const visible =
    preferred.length >= 2
      ? preferred
      : points.filter((point) => (point.visibility ?? 1) > 0.45);

  if (!visible.length) return { x: 0.5, y: 0.5 };
  return visible.reduce(
    (center, point) => ({
      x: center.x + point.x / visible.length,
      y: center.y + point.y / visible.length,
    }),
    { x: 0, y: 0 },
  );
}

function orderPoseFrames(frames: PoseFrame[], previous: Landmark[][]) {
  if (!previous.length) {
    return frames.map((frame) => ({ ...frame, prior: undefined }));
  }

  const unmatched = new Set(frames.map((_, index) => index));
  const ordered: Array<PoseFrame & { prior?: Landmark[] }> = [];

  previous.forEach((prior) => {
    const priorCenter = getPoseCenter(prior);
    let bestIndex = -1;
    let bestDistance = Number.POSITIVE_INFINITY;

    unmatched.forEach((index) => {
      const center = getPoseCenter(frames[index].points);
      const distance = Math.hypot(
        center.x - priorCenter.x,
        center.y - priorCenter.y,
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });

    if (bestIndex >= 0 && bestDistance < 0.45) {
      ordered.push({ ...frames[bestIndex], prior });
      unmatched.delete(bestIndex);
    }
  });

  unmatched.forEach((index) => {
    ordered.push({ ...frames[index], prior: undefined });
  });
  return ordered;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraStageRef = useRef<HTMLDivElement>(null);
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
  const selectedDeviceIdRef = useRef("");
  const mirrorRef = useRef(true);
  const skeletonRef = useRef(true);
  const personMaskEnabledRef = useRef(true);
  const objectsEnabledRef = useRef(true);
  const maxPeopleRef = useRef(1);
  const poseConfiguringRef = useRef(false);
  const previousPosesRef = useRef<Landmark[][]>([]);
  const lastDetectRef = useRef(0);
  const lastObjectDetectRef = useRef(0);
  const lastObjectFoundRef = useRef(0);
  const lastPersonFoundRef = useRef(0);
  const personMaskCanvasesRef = useRef<
    Array<HTMLCanvasElement | null>
  >([]);
  const lastPersonMaskUpdateRef = useRef(0);
  const fpsWindowRef = useRef({ started: 0, frames: 0 });

  const [active, setActive] = useState(false);
  const [mirror, setMirror] = useState(true);
  const [skeleton, setSkeleton] = useState(true);
  const [personMaskEnabled, setPersonMaskEnabled] = useState(true);
  const [objectsEnabled, setObjectsEnabled] = useState(true);
  const [maxPeople, setMaxPeople] = useState(1);
  const [peopleUpdating, setPeopleUpdating] = useState(false);
  const [facing, setFacing] = useState<FacingMode>("user");
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [status, setStatus] = useState<AppStatus>("idle");
  const [poseName, setPoseName] = useState("รอเริ่มตรวจจับ");
  const [fps, setFps] = useState(0);
  const [error, setError] = useState("");
  const [installOpen, setInstallOpen] = useState(false);
  const [standalone, setStandalone] = useState(false);
  const [cleanView, setCleanView] = useState(false);

  useEffect(() => {
    const nav = navigator as Navigator & { standalone?: boolean };
    const savedMaxPeople = Number(
      window.localStorage.getItem(MAX_PEOPLE_STORAGE_KEY),
    );
    if (
      MAX_PEOPLE_OPTIONS.includes(
        savedMaxPeople as (typeof MAX_PEOPLE_OPTIONS)[number],
      )
    ) {
      maxPeopleRef.current = savedMaxPeople;
    }
    const standaloneTimer = window.setTimeout(() => {
      setMaxPeople(maxPeopleRef.current);
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
          numPoses: maxPeopleRef.current,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputSegmentationMasks: true,
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
    personMaskCanvasesRef.current = [];
    lastPersonMaskUpdateRef.current = 0;
  }, []);

  const clearObjectCanvas = useCallback(() => {
    const canvas = objectCanvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawPoses = useCallback(
    (
      rawPoses: Landmark[][],
      personMasks: SegmentationMask[],
      now: number,
    ) => {
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

      const frames = rawPoses
        .slice(0, maxPeopleRef.current)
        .map((points, index) => ({
          points,
          mask: personMasks[index],
        }));
      const orderedFrames = orderPoseFrames(
        frames,
        previousPosesRef.current,
      );
      const poses = orderedFrames.map((frame) => {
        const points = frame.points.map((point, index) => {
          const old = frame.prior?.[index];
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
        return { points, mask: frame.mask };
      });
      previousPosesRef.current = poses.map((pose) => pose.points);

      const primaryPose = poses[0]?.points;
      if (!primaryPose) return;
      const primaryPoseName = classifyPose(primaryPose);
      setPoseName(
        poses.length > 1
          ? `พบ ${poses.length} คน • ${primaryPoseName}`
          : primaryPoseName,
      );

      if (!skeletonRef.current && !personMaskEnabledRef.current) return;

      const scale = Math.max(
        cssWidth / video.videoWidth,
        cssHeight / video.videoHeight,
      );
      const renderedWidth = video.videoWidth * scale;
      const renderedHeight = video.videoHeight * scale;
      const offsetX = (cssWidth - renderedWidth) / 2;
      const offsetY = (cssHeight - renderedHeight) / 2;
      const cachedMasks = personMaskCanvasesRef.current;
      const shouldUpdateMasks =
        personMaskEnabledRef.current &&
        (now - lastPersonMaskUpdateRef.current >=
          PERSON_MASK_UPDATE_INTERVAL_MS ||
          cachedMasks.length !== poses.length);

      if (shouldUpdateMasks) {
        const nextMaskCanvases = poses.map((pose, poseIndex) => {
          const personMask = pose.mask;
          if (!personMask) return null;
          const palette = PERSON_PALETTES[poseIndex % PERSON_PALETTES.length];

          const width = personMask.width;
          const height = personMask.height;
          const confidence = personMask.getAsFloat32Array();
          const cachedMaskCanvas = cachedMasks[poseIndex];
          let maskCanvas = cachedMaskCanvas;

          if (
            !maskCanvas ||
            maskCanvas.width !== width ||
            maskCanvas.height !== height
          ) {
            const nextMaskCanvas = document.createElement("canvas");
            nextMaskCanvas.width = width;
            nextMaskCanvas.height = height;
            maskCanvas = nextMaskCanvas;
          }

          const maskContext = maskCanvas.getContext("2d");
          if (!maskContext || confidence.length < width * height) return null;

          const maskImage = maskContext.createImageData(width, height);
          const pixels = maskImage.data;
          const edgeRadius = 2;

          for (let index = 0; index < width * height; index += 1) {
            const value = confidence[index];
            if (value < PERSON_MASK_THRESHOLD) continue;

            const x = index % width;
            const y = Math.floor(index / width);
            const onEdge =
              x < edgeRadius ||
              y < edgeRadius ||
              x >= width - edgeRadius ||
              y >= height - edgeRadius ||
              confidence[index - edgeRadius] < PERSON_MASK_THRESHOLD ||
              confidence[index + edgeRadius] < PERSON_MASK_THRESHOLD ||
              confidence[index - edgeRadius * width] <
                PERSON_MASK_THRESHOLD ||
              confidence[index + edgeRadius * width] <
                PERSON_MASK_THRESHOLD;
            const pixel = index * 4;

            if (onEdge) {
              pixels[pixel] = palette.edge[0];
              pixels[pixel + 1] = palette.edge[1];
              pixels[pixel + 2] = palette.edge[2];
              pixels[pixel + 3] = 235;
            } else {
              pixels[pixel] = palette.fill[0];
              pixels[pixel + 1] = palette.fill[1];
              pixels[pixel + 2] = palette.fill[2];
              pixels[pixel + 3] = Math.round(42 + Math.min(1, value) * 38);
            }
          }

          maskContext.putImageData(maskImage, 0, 0);
          return maskCanvas;
        });

        personMaskCanvasesRef.current = nextMaskCanvases;
        lastPersonMaskUpdateRef.current = now;
      }

      if (personMaskEnabledRef.current) {
        personMaskCanvasesRef.current.forEach((maskCanvas) => {
          if (!maskCanvas) return;
          context.save();
          context.imageSmoothingEnabled = true;
          if (mirrorRef.current) {
            context.translate(cssWidth, 0);
            context.scale(-1, 1);
          }
          context.drawImage(
            maskCanvas,
            offsetX,
            offsetY,
            renderedWidth,
            renderedHeight,
          );
          context.restore();
        });
      }

      if (!skeletonRef.current) return;

      const pointToCanvas = (point: Landmark) => ({
        x:
          (mirrorRef.current ? 1 - point.x : point.x) * renderedWidth +
          offsetX,
        y: point.y * renderedHeight + offsetY,
      });

      poses.forEach(({ points }, poseIndex) => {
        const palette = PERSON_PALETTES[poseIndex % PERSON_PALETTES.length];
        context.save();
        context.lineCap = "round";
        context.lineJoin = "round";
        context.shadowColor = palette.shadow;
        context.shadowBlur = 10;
        context.strokeStyle = palette.line;
        context.lineWidth = Math.max(
          3,
          Math.min(cssWidth, cssHeight) * 0.006,
        );

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

        context.shadowColor = palette.shadow;
        context.shadowBlur = 12;
        points.forEach((point, index) => {
          if ((point.visibility ?? 1) < 0.55 || (index > 0 && index < 7)) {
            return;
          }
          const current = pointToCanvas(point);
          const radius = Math.max(
            3.5,
            Math.min(cssWidth, cssHeight) * 0.007,
          );
          context.beginPath();
          context.arc(current.x, current.y, radius, 0, Math.PI * 2);
          context.fillStyle = "#071006";
          context.fill();
          context.lineWidth = Math.max(2, radius * 0.48);
          context.strokeStyle = palette.joint;
          context.stroke();
        });
        context.restore();
      });
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
      const rawName =
        category.categoryName || category.displayName || "object";
      if (rawName.toLowerCase() === "person") return;

      const sourceX = mirrorRef.current
        ? video.videoWidth - box.originX - box.width
        : box.originX;
      const x = sourceX * scale + offsetX;
      const y = box.originY * scale + offsetY;
      const width = box.width * scale;
      const height = box.height * scale;
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
        (objectsEnabledRef.current ||
          skeletonRef.current ||
          personMaskEnabledRef.current) &&
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
          const personConfirmed = livingDetections.some((detection) => {
            const category = detection.categories?.[0];
            return (
              category?.categoryName?.toLowerCase() === "person" &&
              (category.score ?? 0) >= PERSON_CONFIRMATION_SCORE
            );
          });
          if (personConfirmed) lastPersonFoundRef.current = now;

          if (livingDetections.length > 0) {
            lastObjectFoundRef.current = now;
            if (objectsEnabledRef.current) setStatus("object");
          }
        } catch {
          // A transient frame decode error is safe to ignore.
        }
      }

      if (
        video &&
        pose &&
        !poseConfiguringRef.current &&
        (skeletonRef.current || personMaskEnabledRef.current) &&
        !processedObjectFrame &&
        video.readyState >= 2 &&
        now - lastDetectRef.current >= 50
      ) {
        lastDetectRef.current = now;
        try {
          const result = pose.detectForVideo(video, now);
          try {
            const poses = (result.landmarks || []) as Landmark[][];
            const personMasks = (result.segmentationMasks ||
              []) as SegmentationMask[];
            const personRecentlyConfirmed =
              !objectDetector ||
              now - lastPersonFoundRef.current <=
                PERSON_CONFIRMATION_WINDOW_MS;
            if (poses.length && personRecentlyConfirmed) {
              drawPoses(poses, personMasks, now);
              setStatus("tracking");
            } else {
              clearCanvas();
              previousPosesRef.current = [];
              setPoseName(
                poses.length ? "ยังไม่ยืนยันว่าเป็นคน" : "ยังไม่พบคน",
              );
              setStatus(
                objectsEnabledRef.current &&
                  now - lastObjectFoundRef.current < 600
                  ? "object"
                  : "lost",
              );
            }
          } finally {
            (result as { close?: () => void }).close?.();
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
  }, [clearCanvas, drawObjects, drawPoses]);

  const openStream = useCallback(
    async (mode: FacingMode, deviceId?: string) => {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          ...(deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: { ideal: mode } }),
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
      });

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();

      const settings = stream.getVideoTracks()[0]?.getSettings();
      const actualFacing =
        settings?.facingMode === "user" ||
        settings?.facingMode === "environment"
          ? settings.facingMode
          : mode;
      facingRef.current = actualFacing;
      setFacing(actualFacing);
      mirrorRef.current = actualFacing === "user";
      setMirror(actualFacing === "user");

      const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(
        (device) => device.kind === "videoinput",
      );
      setVideoDevices(inputs);

      const activeDeviceId = settings?.deviceId || deviceId || "";
      selectedDeviceIdRef.current = activeDeviceId;
      setSelectedDeviceId(activeDeviceId);
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
    previousPosesRef.current = [];
    fpsWindowRef.current = { started: 0, frames: 0 };
    lastObjectDetectRef.current = 0;
    lastObjectFoundRef.current = 0;
    lastPersonFoundRef.current = 0;
    clearCanvas();
    clearObjectCanvas();
    setFps(0);
    setPoseName("รอเริ่มตรวจจับ");
    setStatus("idle");
    setActive(false);
    setCleanView(false);
  }, [clearCanvas, clearObjectCanvas]);

  const startCamera = useCallback(async () => {
    if (activeRef.current) {
      stopCamera();
      return;
    }

    setError("");
    setStatus("camera");
    try {
      await openStream(
        facingRef.current,
        selectedDeviceIdRef.current || undefined,
      );
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

  const cycleCamera = useCallback(async () => {
    if (!activeRef.current || videoDevices.length < 2) {
      const next: FacingMode =
        facingRef.current === "user" ? "environment" : "user";
      facingRef.current = next;
      setFacing(next);
      mirrorRef.current = next === "user";
      setMirror(next === "user");
      if (!activeRef.current) {
        selectedDeviceIdRef.current = "";
        setSelectedDeviceId("");
        return;
      }

      setStatus("camera");
      try {
        await openStream(next);
        previousPosesRef.current = [];
        lastObjectDetectRef.current = 0;
        lastObjectFoundRef.current = 0;
        lastPersonFoundRef.current = 0;
        clearCanvas();
        runDetection();
      } catch {
        setError("สลับกล้องไม่สำเร็จ อุปกรณ์อาจมีกล้องให้เลือกเพียงตัวเดียว");
      }
      return;
    }

    const currentIndex = Math.max(
      0,
      videoDevices.findIndex(
        (device) => device.deviceId === selectedDeviceIdRef.current,
      ),
    );
    const nextDevice = videoDevices[(currentIndex + 1) % videoDevices.length];
    const nextFacing = inferFacingMode(nextDevice.label, facingRef.current);

    setStatus("camera");
    setError("");
    try {
      await openStream(nextFacing, nextDevice.deviceId);
      previousPosesRef.current = [];
      lastObjectDetectRef.current = 0;
      lastObjectFoundRef.current = 0;
      lastPersonFoundRef.current = 0;
      clearCanvas();
      runDetection();
    } catch {
      setError("เปิดเลนส์นี้ไม่สำเร็จ กรุณาลองเลนส์ถัดไป");
    }
  }, [clearCanvas, openStream, runDetection, videoDevices]);

  const changeMaxPeople = useCallback(
    async (requestedValue: number) => {
      if (
        poseConfiguringRef.current ||
        !MAX_PEOPLE_OPTIONS.includes(
          requestedValue as (typeof MAX_PEOPLE_OPTIONS)[number],
        ) ||
        requestedValue === maxPeopleRef.current
      ) {
        return;
      }

      const previousValue = maxPeopleRef.current;
      maxPeopleRef.current = requestedValue;
      setMaxPeople(requestedValue);
      window.localStorage.setItem(
        MAX_PEOPLE_STORAGE_KEY,
        String(requestedValue),
      );
      previousPosesRef.current = [];
      clearCanvas();

      const pose = poseRef.current;
      if (!pose) return;

      poseConfiguringRef.current = true;
      setPeopleUpdating(true);
      setError("");
      if (activeRef.current) {
        setStatus("loading");
        setPoseName(`กำลังปรับเป็นสูงสุด ${requestedValue} คน`);
      }

      try {
        await pose.setOptions({ numPoses: requestedValue });
        if (activeRef.current) {
          setStatus("lost");
          setPoseName(`กำลังมองหาคน สูงสุด ${requestedValue} คน`);
        }
      } catch {
        maxPeopleRef.current = previousValue;
        setMaxPeople(previousValue);
        window.localStorage.setItem(
          MAX_PEOPLE_STORAGE_KEY,
          String(previousValue),
        );
        setError("เปลี่ยนจำนวนคนไม่สำเร็จ กรุณาลองใหม่");
      } finally {
        poseConfiguringRef.current = false;
        setPeopleUpdating(false);
      }
    },
    [clearCanvas],
  );

  const enterCleanView = useCallback(async () => {
    setCleanView(true);
    try {
      if (cameraStageRef.current?.requestFullscreen) {
        await cameraStageRef.current.requestFullscreen();
      }
    } catch {
      // iPhone Safari uses the full-viewport CSS fallback.
    }
  }, []);

  const exitCleanView = useCallback(async () => {
    setCleanView(false);
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
    } catch {
      // The CSS view has already been restored.
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) setCleanView(false);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

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

  const selectedCameraIndex = videoDevices.findIndex(
    (device) => device.deviceId === selectedDeviceId,
  );
  const cameraButtonLabel =
    videoDevices.length > 1 && selectedCameraIndex >= 0
      ? `เลนส์ ${selectedCameraIndex + 1}/${videoDevices.length}`
      : facing === "user"
        ? "กล้องหน้า"
        : "กล้องหลัง";

  return (
    <main className={`app-shell ${cleanView ? "is-clean-view" : ""}`}>
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
        <div
          aria-label={cleanView ? "แตะเพื่อออกจากเต็มจอ" : undefined}
          className={`camera-stage ${active ? "is-active" : ""}`}
          onClick={() => {
            if (cleanView) void exitCleanView();
          }}
          onKeyDown={(event) => {
            if (
              cleanView &&
              (event.key === "Enter" || event.key === " ")
            ) {
              void exitCleanView();
            }
          }}
          ref={cameraStageRef}
          role={cleanView ? "button" : undefined}
          tabIndex={cleanView ? 0 : undefined}
        >
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
              <p>คนเป็นเงาโปร่งแสงพร้อมโครงกระดูก สัตว์เป็นกรอบสี่เหลี่ยม</p>
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

          <div className="control-dock">
            <button
              aria-label="เข้าโหมดเต็มจอ"
              className="dock-visibility-toggle"
              disabled={!active}
              onClick={(event) => {
                event.stopPropagation();
                void enterCleanView();
              }}
              type="button"
            >
              <Icon name="fullscreen" size={19} />
              <span>เต็มจอ</span>
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
              <button onClick={cycleCamera} type="button">
                <Icon name="camera" />
                <span>{cameraButtonLabel}</span>
              </button>
              <label
                className={`people-count-control ${peopleUpdating || status === "loading" ? "is-updating" : ""}`}
              >
                <Icon name="people" />
                <select
                  aria-label="เลือกจำนวนคนสูงสุดที่ต้องการตรวจจับ"
                  disabled={peopleUpdating || status === "loading"}
                  onChange={(event) => {
                    void changeMaxPeople(Number(event.target.value));
                  }}
                  value={maxPeople}
                >
                  {MAX_PEOPLE_OPTIONS.map((count) => (
                    <option key={count} value={count}>
                      {count} คน
                    </option>
                  ))}
                </select>
              </label>
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
                <span>Skeleton</span>
              </button>
              <button
                aria-label="เปิดหรือปิดเส้นขอบและสีโปร่งแสงของคน"
                aria-pressed={personMaskEnabled}
                className={personMaskEnabled ? "is-on" : ""}
                onClick={() => {
                  setPersonMaskEnabled((value) => {
                    const next = !value;
                    personMaskEnabledRef.current = next;
                    clearCanvas();
                    return next;
                  });
                }}
                type="button"
              >
                <Icon name="silhouette" />
                <span>เงาคน</span>
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

import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Pose Skeleton Camera",
    short_name: "Pose Skeleton",
    description:
      "ตรวจจับท่าทางมนุษย์และแสดงเส้นโครงกระดูกแบบเรียลไทม์",
    start_url: "/",
    display: "standalone",
    background_color: "#070a0c",
    theme_color: "#070a0c",
    orientation: "any",
    icons: [
      {
        src: "/icon.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "any",
      },
      {
        src: "/icon-maskable.svg",
        sizes: "any",
        type: "image/svg+xml",
        purpose: "maskable",
      },
    ],
  };
}

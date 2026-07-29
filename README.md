# HuCam — Pose Skeleton Camera

เว็บแอปตรวจจับท่าทางมนุษย์จากกล้องและวาดเส้นโครงกระดูกแบบเรียลไทม์
ออกแบบสำหรับ Safari บน iPhone/iPad และรองรับการติดตั้งผ่าน Add to Home Screen

## ความสามารถ

- รองรับแนวตั้งและแนวนอน
- สลับกล้องหน้าและกล้องหลัง
- เปิด/ปิดภาพกระจกและเส้นโครงกระดูก
- แสดงชื่อท่าพื้นฐาน เช่น ยืน กางแขน ยกแขน และย่อเข่า
- ประมวลผลภาพบนอุปกรณ์ด้วย MediaPipe Pose Landmarker
- PWA พร้อม manifest, service worker และไอคอน

## เริ่มใช้งาน

```bash
npm install
npm run dev
```

เปิด `http://localhost:3000` และอนุญาตให้เว็บไซต์เข้าถึงกล้อง

## Production

```bash
npm run build
npm start
```

import os
import cv2
import numpy as np
import requests
import tempfile
from ultralytics import YOLO
from firebase_admin import storage, firestore

import YOLOv11.YOLO as YOLO, YOLOv11.geocoding as geocoding, MediaPipe.lstm_Analysis as lstm_p1


# firestore 이미지 다운로드
def download_image(url):
    """이미지 URL에서 이미지를 다운로드해 numpy array로 반환"""
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        print(f"🚫 이미지 다운로드 실패: {url}")
        return None


# GPS 정도 바탕 지번주소 추출
def find_adress(doc_id):
    # 신고 정보 중 GPS 가져와 지번주소 추출
    db_fs = firestore.client()
    doc_ref = db_fs.collection("Report").document(doc_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_data = doc.to_dict()
        gps_info = doc_data.get("gpsInfo")
    if gps_info:
        lat_str, lon_str = gps_info.strip().split()
        lat = float(lat_str)
        lon = float(lon_str)
        parcel_addr = geocoding.reverse_geocode(lat, lon, os.getenv("VWorld_API"))
        return lat, lon, parcel_addr


# firebase 데이터 저장 메소드
def save_conclusion(
    doc_id,
    date,
    user_id,
    violation,
    result,
    region,
    gpsInfo,
    imageUrl,
    reportImgUrl,
    aiConclusion=None,
    detectedBrand=None,
    confidence=None,
):

    db_fs = firestore.client()
    full_doc_id = f"conclusion_{doc_id}"

    # 저장할 데이터
    conclusion_data = {
        "date": date,
        "userId": user_id,
        "aiConclusion": aiConclusion or [],
        "violation": violation,
        "result": result,
        "region": region,
        "gpsInfo": gpsInfo,
        "imageUrl": imageUrl,
        "reportImgUrl": reportImgUrl or imageUrl,
    }

    # 브랜드
    if detectedBrand:
        conclusion_data["detectedBrand"] = detectedBrand
    # conf
    if confidence is not None:
        conclusion_data["confidence"] = confidence

    db_fs.collection("Conclusion").document(full_doc_id).set(conclusion_data)


def center(box):
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

def process_image(image_url, date, user_id, violation, doc_id):
    print(f"🔥 이미지 처리 시작: {image_url}")
    image = download_image(image_url)
    if image is None:
        print("🚫 이미지 로드 실패, 건너뜀")
        return

    traffic_violation_detection = []

    # 킥보드/사람 bbox 리스트 감지
    kickboard_boxes = YOLO.kickboard_boxes(image)
    person_boxes = YOLO.person_boxes(image)

    # 감지 피드백
    if len(kickboard_boxes) == 0:
        traffic_violation_detection.append("킥보드 감지 실패")
        print("🚫 킥보드 감지 안됨")
    else :
        print("✅ 킥보드 감지")
    
    if len(person_boxes) == 0:
        traffic_violation_detection.append("사람 감지 실패")
        print("🚫 사람 감지 안됨")
    else :
        print("✅ 사람 감지")

    if len(kickboard_boxes) != 0 and len(person_boxes) != 0:
        # 사람별로 가장 가까운 킥보드 한 곳에만 배정
        person_centers = [center(p) for p in person_boxes]
        kb_centers = [center(kb) for kb in kickboard_boxes]
        person_assignment = {}
        for i, p_c in enumerate(person_centers):
            min_dist = None
            assigned_kb = None
            for j, kb_c in enumerate(kb_centers):
                dist = np.hypot(p_c[0] - kb_c[0], p_c[1] - kb_c[1])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    assigned_kb = j
            person_assignment[i] = (assigned_kb, min_dist)

        # 킥보드별로, 자기로 배정된 사람 중 가까운 2명만 선택
        kb_persons = {i: [] for i in range(len(kickboard_boxes))}
        for person_idx, (kb_idx, dist) in person_assignment.items():
            kb_persons[kb_idx].append((dist, person_boxes[person_idx]))  # 거리와 함께 저장

        pad = 100
        for k_idx, k_box in enumerate(kickboard_boxes):
            close_persons = sorted(kb_persons[k_idx], key=lambda x: x[0])[:2]
            if not close_persons:
                continue
            group_boxes = [p_box for (_, p_box) in close_persons]

            # crop 영역 계산 + 패딩 적용
            xs = [k_box[0], k_box[2]] + [b[0] for b in group_boxes] + [b[2] for b in group_boxes]
            ys = [k_box[1], k_box[3]] + [b[1] for b in group_boxes] + [b[3] for b in group_boxes]
            x1 = int(max(min(xs) - pad, 0))
            x2 = int(min(max(xs) + pad, image.shape[1]))
            y1 = int(max(min(ys) - pad, 0))
            y2 = int(min(max(ys) + pad, image.shape[0]))
            cropped = image[y1:y2, x1:x2].copy()

            # ====== crop별 추가 분석 ======
            # 필드 분석은 cropped 이미지를 인자로 사용
            brand = YOLO.brand_analysis(cropped)
            helmet_detected, helmet_results, top_helmet_confidence = YOLO.helmet_analysis(cropped)
            
            aiConclusion = []

            if helmet_detected:
                YOLO.draw_boxes(helmet_results, cropped, (0, 0, 255), "Helmet")
                print("✅ 헬멧 감지")
                aiConclusion.append("위반 사항 없음")
            else:
                aiConclusion.append("헬멧 미착용")
                print("🚫 헬멧 미착용")

            bucket = storage.bucket()
            conclusion_blob = bucket.blob(f"Conclusion/{doc_id}_{k_idx}.jpg")

            _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
            cv2.imwrite(temp_annotated, cropped)
            conclusion_blob.upload_from_filename(temp_annotated)
            conclusion_url = conclusion_blob.public_url

            # 신고 정보 중 GPS 가져와 지번주소 추출
            lat, lon, parcel_addr = find_adress(doc_id)

            # Firestore 저장
            lat, lon, parcel_addr = find_adress(doc_id)
            save_conclusion(
                doc_id=f"{doc_id}_{k_idx}",
                date=date,
                user_id=user_id,
                violation=violation,
                result="미확인",
                aiConclusion=aiConclusion,
                detectedBrand=brand,
                confidence=top_helmet_confidence,
                gpsInfo=f"{lat} {lon}",
                region=parcel_addr,
                imageUrl=conclusion_url,
                reportImgUrl=image_url,
            )

            print(f"✅ 킥보드 {k_idx} 분석 및 저장 완료: {conclusion_url}")


    else:
        print("🛑 킥보드 혹은 사람을 감지하지 못했습니다. 자동 반려처리 진행됩니다.\n")

        # 신고 정보 중 GPS 가져와 지번주소 추출
        lat, lon, parcel_addr = find_adress(doc_id)

        save_conclusion(
            doc_id=doc_id,
            date=date,
            user_id=user_id,
            violation=violation,
            result="반려",
            aiConclusion=traffic_violation_detection,
            gpsInfo=f"{lat} {lon}",
            region=parcel_addr,
            imageUrl=image_url,
            reportImgUrl=image_url,
        )

        print(f"❌ 반려된 사진 url : {image_url}\n")


# Firestore 실시간 리스너 설정
def on_snapshot(col_snapshot, changes, read_time):
    # 초기 스냅샷은 무시 (최초 1회 실행 시 건너뜀)
    # if not hasattr(on_snapshot, "initialized"):
    #     on_snapshot.initialized = True
    #     return

    for change in changes:
        if change.type.name == "ADDED":
            doc_id = change.document.id
            doc_data = change.document.to_dict()
            if "imageUrl" in doc_data:
                print(f"🔥 새로운 신고 감지 : {doc_id}")
                violation = doc_data.get("violation", "")
                # 배열이면 문자열로 합침
                if isinstance(violation, list):
                    violation = ", ".join(violation)
                process_image(
                    doc_data["imageUrl"],
                    doc_data.get("date", ""),
                    doc_data.get("userId", ""),
                    violation,
                    doc_id,
                )


if __name__ == "__main__":
    import time
    import YOLOv11.firebase_config
    from firebase_admin import firestore

    db_fs = firestore.client()
    report_col = db_fs.collection("Report")
    listener = report_col.on_snapshot(on_snapshot)

    print("🔥 Firestore 실시간 감지 시작 (종료: Ctrl+C) 🔥")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        listener.unsubscribe()
        print("\n🛑 Firestore 실시간 감지를 종료합니다.")

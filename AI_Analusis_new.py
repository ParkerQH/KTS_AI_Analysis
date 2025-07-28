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
    imageUrl,
    reportImgUrl,
    idx,
    aiConclusion=None,
    detectedBrand=None,
    confidence=None,
):

    db_fs = firestore.client()
    full_doc_id = f"conclusion_{doc_id}_{idx}"

    # 신고 정보 중 GPS 가져와 지번주소 추출
    lat, lon, parcel_addr = find_adress(doc_id)

    # 저장할 데이터
    conclusion_data = {
        "date": date,
        "userId": user_id,
        "aiConclusion": aiConclusion or [],
        "violation": violation,
        "result": result,
        "region": parcel_addr,
        "gpsInfo": f"{lat} {lon}",
        "imageUrl": imageUrl,
        "reportImgUrl": reportImgUrl,
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

    # ====== 감지 피드백 ======
    # 킥보드 감지
    if len(kickboard_boxes) == 0:
        traffic_violation_detection.append("킥보드 감지 실패")
        print("🚫 킥보드 감지 안됨")
    else :
        print("✅ 킥보드 감지")
    # 사람 감지
    if len(person_boxes) == 0:
        traffic_violation_detection.append("사람 감지 실패")
        print("🚫 사람 감지 안됨")
    else :
        print("✅ 사람 감지")

    # ====== AI 분석 ======
    # 사진에서 킥보드와 사람이 모두 감지된 경우
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

        pad = 200   # 객체 스샷 시 패딩값
        idx = 0     # 사진 속 킥보드 별 인덱스 번호

        for k_idx, k_box in enumerate(kickboard_boxes):
            close_persons = sorted(kb_persons[k_idx], key=lambda x: x[0])[:3]
            if not close_persons:   # 킥보드에 할당된 사람이 한 명도 없으면 이후 분석 블록이 실행 X
                continue
            group_boxes = [p_box for (_, p_box) in close_persons]

            # crop 영역 계산 + 패딩 적용
            xs = [k_box[0], k_box[2]] + [b[0] for b in group_boxes] + [b[2] for b in group_boxes]
            ys = [k_box[1], k_box[3]] + [b[1] for b in group_boxes] + [b[3] for b in group_boxes]
            x1 = int(max(min(xs) - pad, 0))
            x2 = int(min(max(xs) + pad, image.shape[1]))
            y1 = int(max(min(ys) - pad, 0))
            y2 = int(min(max(ys) + pad, image.shape[0]))
            cropped = image[y1:y2, x1:x2].copy()    # 분할된 이미지 데이터

            # AI 분석 내용
            aiConclusion = []

            lstm_results = []
            for p_box in group_boxes:
                # 각 사람 crop 추출
                px1, py1, px2, py2 = map(int, p_box)
                person_crop = cropped[
                    max(py1 - y1, 0): max(py2 - y1, 0),   # crop 내 상대 좌표 변환
                    max(px1 - x1, 0): max(px2 - x1, 0)
                ]
                # 포즈 LSTM 분석 결과 (True=탑승자, False=보행자, None/예외=분석불가)
                try:
                    pose_result = lstm_p1.lstm_Analysis_per1(person_crop)
                    if pose_result is None:
                        lstm_results.append("분석불가")
                    elif pose_result:
                        lstm_results.append("탑승자")
                    else:
                        lstm_results.append("보행자")
                except Exception as e:
                    lstm_results.append("분석불가")

            # --------------------------------------
            # lstm_results에 저장된 결과 카운트
            n_rider = lstm_results.count("탑승자")
            n_pedestrian = lstm_results.count("보행자")
            n_unknown = lstm_results.count("분석불가")

            # 판단 로직
            if n_rider >= 2:
                aiConclusion.append("2인탑승 의심")
                print("🚫 2인탑승으로 의심됩니다.")
            elif n_rider == 1:
                print("✅ 1인탑승으로 판단")
            elif n_rider == 0 and n_pedestrian >= 1:
                aiConclusion.append("보행자로 판단")
                print("✅ 보행자로 판단")
                print("🛑 보행자로 판단됩니다. 자동 반려처리 진행됩니다.\n")   

                # 자동 반려 처리
                bucket = storage.bucket()
                conclusion_blob = bucket.blob(f"Conclusion/{doc_id}_{idx}.jpg")

                _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
                cv2.imwrite(temp_annotated, cropped)
                conclusion_blob.upload_from_filename(temp_annotated)
                conclusion_url = conclusion_blob.public_url

                save_conclusion(
                    doc_id=doc_id,
                    date=date,
                    user_id=user_id,
                    violation=violation,
                    result="반려",
                    aiConclusion=aiConclusion,
                    imageUrl=conclusion_url,
                    reportImgUrl=image_url,
                    idx=idx
                )
                print(f"❌ 반려된 사진 url : {conclusion_url}\n")
                idx += 1
                continue
            else :
                print("❌ 방해 요소가 많아 분석이 불가능 합니다.")

            # ====== crop별 추가 분석 ======
            # 브랜드 분석
            brand = YOLO.brand_analysis(cropped)
            if brand is None:   # 브랜드 감지 실패 시 자동 반려처리
                bucket = storage.bucket()
                conclusion_blob = bucket.blob(f"Conclusion/{doc_id}_{idx}.jpg")

                _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
                cv2.imwrite(temp_annotated, cropped)
                conclusion_blob.upload_from_filename(temp_annotated)
                conclusion_url = conclusion_blob.public_url
                
                save_conclusion(
                    doc_id=doc_id,
                    date=date,
                    user_id=user_id,
                    violation=violation,
                    result="반려",
                    aiConclusion="브랜드 감지 실패",
                    imageUrl=conclusion_url,
                    reportImgUrl=image_url,
                    idx=idx
                )

                print(f"❌ 반려된 사진 url : {conclusion_url}\n")
                idx += 1
                continue

            # 헬멧 착용 여부 분석
            helmet_detected, helmet_results, top_helmet_confidence = YOLO.helmet_analysis(cropped)
            if helmet_detected:
                YOLO.draw_boxes(helmet_results, cropped, (0, 0, 255), "Helmet")
                print("✅ 헬멧 감지\n")
                aiConclusion.append("위반 사항 없음")
            else:
                aiConclusion.append("헬멧 미착용")
                print("🚫 헬멧 미착용\n")

            bucket = storage.bucket()
            conclusion_blob = bucket.blob(f"Conclusion/{doc_id}_{idx}.jpg")

            _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
            cv2.imwrite(temp_annotated, cropped)
            conclusion_blob.upload_from_filename(temp_annotated)
            conclusion_url = conclusion_blob.public_url

            save_conclusion(
                doc_id=doc_id,
                date=date,
                user_id=user_id,
                violation=violation,
                result="미확인",
                aiConclusion=aiConclusion,
                detectedBrand=brand,
                confidence=top_helmet_confidence,
                imageUrl=conclusion_url,
                reportImgUrl=image_url,
                idx=idx
            )

            print(f"✅ 킥보드 {idx}번 분석 및 저장 완료: {conclusion_url}\n")
            idx +=1


    else:
        print("🛑 킥보드 혹은 사람을 감지하지 못했습니다. 자동 반려처리 진행됩니다.\n")

        save_conclusion(
            doc_id=doc_id,
            date=date,
            user_id=user_id,
            violation=violation,
            result="반려",
            aiConclusion=traffic_violation_detection,
            imageUrl=image_url,
            reportImgUrl=image_url,
            idx = 0
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

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


def process_image(image_url, date, user_id, violation, doc_id):
    print(f"🔥 이미지 처리 시작: {image_url}")
    image = download_image(image_url)
    if image is None:
        print("🚫 이미지 로드 실패, 건너뜀")
        return

    traffic_violation_detection = []

    # 1. 킥보드, 사람 감지
    kickboard = YOLO.kickboard_analysis(image)
    person = YOLO.person_analysis(image)

    # 1-2. 킥보드 감지 피드백
    if kickboard:
        print("✅ 킥보드 감지")
    else:
        traffic_violation_detection.append("킥보드 감지 실패")
        print("🚫 킥보드 감지 안됨")

    # 1-3. 사람 감지 피드백
    if person:
        print("✅ 사람 감지")
    else:
        traffic_violation_detection.append("사람 감지 실패")
        print("🚫 사람 감지 안됨")

    # 1-4. 탑승자와 보행자 구분
    if kickboard and person:
        lstm_pose = lstm_p1.lstm_Analysis_per1(image)
        if lstm_pose:
            print("✅ 탑승자 감지")
        else:
            traffic_violation_detection.append("보행자로 판단됨")
            print("🚫 보행자로 판단")

    # 킥보드 감지, 사람 감지, 탑승자로 판단된 경우
    if kickboard and person and lstm_pose:
        # 3-1. 전동킥보드 브랜드 분석
        top_brand_class = YOLO.brand_analysis(image)

        # 3-2. 헬멧 착용 여부 분석
        helmet_detected, helmet_results, top_helmet_confidence = YOLO.helmet_analysis(
            image
        )
        if helmet_detected:
            YOLO.draw_boxes(helmet_results, image, (0, 0, 255), "Helmet")
            print("✅ 헬멧 감지")
            traffic_violation_detection.append("위반 사항 없음")
        else:
            traffic_violation_detection.append("헬멧 미착용")
            print("🚫 헬멧 미착용")

        # 분석 이미지 저장 (Firebase Storage)
        bucket = storage.bucket()
        conclusion_blob = bucket.blob(f"Conclusion/{doc_id}.jpg")

        # 임시 파일 생성 (분석 이미지)
        _, temp_annotated = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(temp_annotated, image)
        conclusion_blob.upload_from_filename(temp_annotated)
        conclusion_url = conclusion_blob.public_url

        # 신고 정보 중 GPS 가져와 지번주소 추출
        lat, lon, parcel_addr = find_adress(doc_id)

        # Firestore에 저장될 내용
        save_conclusion(
            doc_id=doc_id,
            date=date,
            user_id=user_id,
            violation=violation,
            result="미확인",
            aiConclusion=traffic_violation_detection,
            detectedBrand=top_brand_class,
            confidence=top_helmet_confidence,
            gpsInfo=f"{lat} {lon}",
            region=parcel_addr,
            imageUrl=conclusion_url,
            reportImgUrl=image_url,
        )

        print(f"✅ 분석된 사진 url : {conclusion_url}\n")

    elif kickboard and person:
        print("🛑 사진 속 사람은 보행자로 판단됩니다. 자동 반려처리 진행됩니다.\n")
    
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

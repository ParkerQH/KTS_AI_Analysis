import os
import cv2
import numpy as np
import YOLOv11.YOLO as YOLO

def center(box):
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

image_path = "output/2025-07-27 112851.jpg"
image = cv2.imread(image_path)
kickboard_boxes = YOLO.kickboard_boxes(image)
person_boxes = YOLO.person_boxes(image)

# 사람별로 가장 가까운 킥보드 매칭(딱 1곳)
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

# 킥보드별로, "자신에게 가장 가까운" 사람만 모으고, 상위 3명만 선정
kb_persons = {i: [] for i in range(len(kickboard_boxes))}
for person_idx, (kb_idx, dist) in person_assignment.items():
    kb_persons[kb_idx].append((dist, person_boxes[person_idx]))  # 거리와 같이

pair_idx = 0
for k_idx, k_box in enumerate(kickboard_boxes):
    # 가까운 순으로 2명까지 자르기
    close_persons = sorted(kb_persons[k_idx], key=lambda x: x[0])[:3]
    if not close_persons:
        continue
    group_boxes = [p_box for (_, p_box) in close_persons]

    # 전체 범위 조절
    xs = [k_box[0], k_box[2]] + [b[0] for b in group_boxes] + [b[2] for b in group_boxes]
    ys = [k_box[1], k_box[3]] + [b[1] for b in group_boxes] + [b[3] for b in group_boxes]
    x1, x2 = int(max(min(xs) - 10, 0)), int(min(max(xs) + 10, image.shape[1]))
    y1, y2 = int(max(min(ys) - 10, 0)), int(min(max(ys) + 10, image.shape[0]))
    cropped = image[y1:y2, x1:x2].copy()

    # --- 박싱(사각형) 표시 추가 ---
    # 킥보드 박스(파랑)
    kx1, ky1 = int(k_box[0] - x1), int(k_box[1] - y1)
    kx2, ky2 = int(k_box[2] - x1), int(k_box[3] - y1)
    cv2.rectangle(cropped, (kx1, ky1), (kx2, ky2), (255, 0, 0), 2)

    # 사람 박스(초록)
    for p_box in group_boxes:
        px1, py1 = int(p_box[0] - x1), int(p_box[1] - y1)
        px2, py2 = int(p_box[2] - x1), int(p_box[3] - y1)
        cv2.rectangle(cropped, (px1, py1), (px2, py2), (0, 255, 0), 2)

    os.makedirs("output", exist_ok=True)
    save_path = f"output/_{pair_idx}.jpg"
    cv2.imwrite(save_path, cropped)
    print(f"✅ 킥보드별 가까운 사람 2명 포함 crop+박싱 저장: {save_path}")
    pair_idx += 1
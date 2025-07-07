import json
import os
import numpy as np
import pickle
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODE = 'alone' # 單人
#MODE = 'multi' # 多人

if MODE == 'alone':
    INPUT_DIR = os.path.join(BASE_DIR, "alone")
    OUTPUT_DIR = os.path.join(BASE_DIR, "alone")
elif MODE == 'multi':
    INPUT_DIR = os.path.join(BASE_DIR, "multi")
    OUTPUT_DIR = os.path.join(BASE_DIR, "multi")
else:
    raise ValueError("我已經把單人和多人區分開來，所以json位置一定要是 alon(單人) 或 multi(多人)裡面")

# 開始把所有json 都拿出來翻譯(等等會再去重)
json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

if not json_files:
    print("沒有找到任何 JSON 檔案。")
    exit()

print(f"在 {INPUT_DIR} 找到 {len(json_files)} 個 JSON 檔案。")

for i in json_files:
    json_path = os.path.join(INPUT_DIR, i)
    pkl_file = os.path.splitext(i)[0] + ".pkl" # 取同樣檔名但是副檔名不一樣。 test00001.json -> test00001.pkl 
    pkl_output_path = os.path.join(OUTPUT_DIR, pkl_file)

    if os.path.exists(pkl_output_path):
        print(f"{pkl_file} 已存在，跳過。")
        continue

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"讀取 {i} 發生錯誤：{e}")
        continue

    # defaultdict 讓我們在第一次遇到新的 track_id 時，自動為其建立一個空列表
    tracked_persons = defaultdict(list) # 這裡之所以準備 track_id是為了未來多人影像，現在單人當然沒有那種東西
    for frame in sorted(data, key=lambda x: x["frame_id"]):
        instances = frame.get("instances", [])
        if not instances:
            print(f"在第 {i} 的 {frame} 幀時沒有instances")
            continue
        for j in instances:
            kpts = j.get("keypoints")
            if not kpts:
                print(f"在第 {i} 的 {frame} 幀時的 {j} 個關節點，沒有 keypoints")
                continue
            
            track_id = j.get("track_id", 0)
            tracked_persons[track_id].append(kpts)
            
    if not tracked_persons:
        print(f"{i} 中沒有任何有效資料，跳過。")
        continue

    print(f"{i}：找到 {len(tracked_persons)} 個 track_id。") # 順便檢查人數是否正確


    output_data = []
    for track_id, kpts_data in tracked_persons.items():
        keypoints_3d = np.array(kpts_data) # 取出數據
        num_frames = keypoints_3d.shape[0] # 就是 frame_id
        keypoints_2d = np.zeros((num_frames, 17, 2), dtype=np.float32)
        visible = np.ones((num_frames, 17), dtype=np.float32)

        data_sample = {
            "keypoints": keypoints_2d,
            "keypoints_visible": visible,
            "keypoints_3d": keypoints_3d,
            "keypoints_3d_visible": visible,
            "track_id": track_id
        }

        output_data.append(data_sample)

    with open(pkl_output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"{pkl_file} 做好了。")

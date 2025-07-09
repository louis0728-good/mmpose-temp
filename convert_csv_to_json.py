import csv
import json
import os
import glob

def convert_csv_to_json(output_path, input_path, original_json_path):

    if not os.path.exists(original_json_path):
        print(f"[警告] 找不到原始的 json 檔案 {original_json_path}，先略過")
        return
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"正在從原始檔案 {os.path.basename(original_json_path)} 讀取原始「自信分數」。...")
    scores_map = {}

    for frame in  original_data:
        # 先跳過沒有 instances 的幀，理論上不太可能，但還是要注意
        if 'instances' not in frame or not frame['instances']:
            print(f"[警告] 在第 {frame} 沒有找到 'instances' ")
            continue

        try:
            # 先解包 instances，處理 [[{...}]] 的錯誤格式
            first_item = frame['instances'][0] # instances 的第零個物件包括 keypoints 還有 keypoints_scores

            if isinstance(first_item, list) and first_item:
                i_data = first_item[0] #是錯誤的格式，被多包了一層，這邊解包後變成 [{}, {}, ] 正常格式
            else:
                i_data = first_item

            # 確認 i_data 是包括所有東西的列表後，再用比較安全的方式獲取自信分數
            if 'keypoint_scores' not in i_data:
                continue
            scores = i_data['keypoint_scores']

            # 若是 keypoint_scores 還有自己的錯誤格式。
            while scores and isinstance(scores, list) and isinstance(scores[0], list) and len(scores) == 1: # [[]] 像這樣就錯了
                scores = scores[0]

            scores_map[frame['frame_id']] = scores

        except (TypeError, IndexError) as e:
            print(f"[警告] 在讀取原始 json 的第 {frame.get('frame_id', '未知')} 幀時，結構很怪，無法獲得自信分數: {e}")
            continue
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 我們前面有標題，所以要跳過
        next(reader)
        next(reader)
        file_name = os.path.basename(input_path)

        for row in reader:
            try:
                frame_id = int(row[0]) # 第三列開始的第一個都是幀數，前提是你們 csv 格式跟我一樣，所以盡量用我放到 FB 的那個
                kps = []  # 要放 keypoints
                        
                # xyz 成一組，每次讀一組(xyz)
                for i in range(1, len(row), 3):
                    if i+2 < len(row):
                        x = float(row[i])
                        y = float(row[i+1])
                        z = float(row[i+2])
                        kps.append([x, y, z])
                keypoint_scores = scores_map.get(frame_id)

                # 檢查 keypoint 數量是否和 scores 數量一樣
                if not keypoint_scores or len(keypoint_scores) != len(kps):
                    print(f"[警告] 在 {file_name} 的第 {frame_id} 時，自信分數和關節點數量不符，或找不到此幀的自信分數")
                    keypoint_scores = [1.0] * len(kps)

                frame_data = {
                    "frame_id": frame_id,
                    "instances": 
                    [
                        {
                            "keypoints": kps,
                            "keypoint_scores": keypoint_scores
                        }
                    ]
                }
                data.append(frame_data)

            except(ValueError, IndexError) as e:
                print("==================================================\n")
                print(f"   檔案名稱: {file_name}")
                # 檢查 row 是否為空，避免額外的錯誤
                print(f"   問題幀數 (行號): {row[0] if row else '未知'}")
                print(f"   錯誤原因: {e}")
                print(f"   這行可能裡面有些沒辦法翻譯的東西，比如說空格或其他可以去檢查") 
                # 這邊我補做了這個因為可以順便檢查剛剛的csv是否有我們不小心露改的。
                print(f"   問題行內容: {row}\n")

        # 確保輸出目錄存在，我是有自己建立啦，但是怕你們沒有。
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as jf:
            json.dump(data, jf, indent=2, ensure_ascii=False)
        print(f"成功將 '{os.path.basename(input_path)}' 轉換並儲存在 '{output_path}'")

def main():
    # 我這的檔案所在的目錄(你們要根據自己的位置自己調)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 路徑設定 (可切換) ---
    
    # 單人
    csv_input_dir = os.path.join(base_dir, 'result_csv', 'one_person')
    json_input_dir = os.path.join(base_dir, 'one_person')
    json_output_dir = os.path.join(base_dir, 'after_refined', 'alone')

    # 多人
    # csv_input_dir = os.path.join(base_dir, 'result_csv', 'multi_people')
    # json_input_dir = os.path.join(base_dir, 'multi_people')
    # json_output_dir = os.path.join(base_dir, 'after_refined', 'multi')
    
    print(f"開始讀取csv： '{csv_input_dir}'")
    # 尋找所有在 csv_input_dir 中的 .csv 檔案
    csv_files = glob.glob(os.path.join(csv_input_dir, '*.csv'))

    if not csv_files:
        print(f"在 '{csv_input_dir}' 中找不到任何 CSV 檔案。")
        return

    for csv_path in csv_files:
        file_name_without_ext = os.path.splitext(os.path.basename(csv_path))[0]
        
        original_json_path = os.path.join(json_input_dir, file_name_without_ext + '.json')
        output_json_path = os.path.join(json_output_dir, file_name_without_ext + '.json')
        
        if os.path.exists(output_json_path):
            print(f"{output_json_path} 已存在，跳過此檔案。")
            continue

        # 第1個參數 output_path  = output_json_path
        # 第2個參數 input_path   = csv_path
        # 第3個參數 original_json_path = original_json_path
        convert_csv_to_json(output_json_path, csv_path, original_json_path)

main()
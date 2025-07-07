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

    # 我們直接透過 frame_id 去找對應的keypoints_scores，因為我們 CSV 沒有記錄所以只能從原檔案抄
    scores_map = {frame['frame_id']: frame['instances'][0]['keypoint_scores'] 
                  for frame in original_data if 'instances' in frame and frame['instances']}
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 我們前面有標題，所以要跳過
        next(reader)
        next(reader)

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
                    print(f"[警告] 在第 {frame_id} 時，自信分數和關節點數量不符，或找不到此幀的自信分數")
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
                file_name = os.path.basename(input_path)
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
    json_output_dir = os.path.join(base_dir, 'after_refined_json', 'alone')

    # 多人
    # csv_input_dir = os.path.join(base_dir, 'result_csv', 'multi_people')
    # json_input_dir = os.path.join(base_dir, 'multi_people')
    # json_output_dir = os.path.join(base_dir, 'after_refined_json', 'multi')
    
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
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from tqdm import tqdm


# 這邊先設定檔案的路徑，你們可以自己針對自己的位置去調整
BASE_DIR = r"C:\Users\LOUIS\mmpose\outputs\human3d\after_refined"

# 輸入，要讀取 json
INPUT_DIR = os.path.join(BASE_DIR, "alone") # 單人
# INPUT_DIR = os.path.join(BASE_DIR, "multi") # 多人

# 輸出，要輸出影片
OUTPUT_DIR = os.path.join(BASE_DIR, "result_video", "one_person") # 單人
# OUTPUT_DIR = os.path.join(BASE_DIR, "result_video", "multi_people") # 多人

SKELETON = [ # human3.6m 的規定連線順序。(還是其實是mmpose 我不是很確定專業說法)
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16]
]

# 我盡量讓顏色和 mmpose 輸出的骨架顏色差不多
SKELETON_COLOR = {
    (0,1): 'lightsalmon', (1,2): 'lightsalmon', (2,3): 'lightsalmon', # 右腿
    (0,4): 'mediumaquamarine', (4,5): 'mediumaquamarine', (5,6): 'mediumaquamarine', # 左腿
    (0,7): 'skyblue', (7,8): 'skyblue', (8,9): 'skyblue', (9,10): 'skyblue',  # 身體
    (8,11): 'mediumaquamarine', (11,12): 'mediumaquamarine', (12,13): 'mediumaquamarine', # 左臂
    (8,14): 'lightsalmon', (14,15): 'lightsalmon', (15,16): 'lightsalmon' # 右臂
}

# 把所有幀的座標一次掃描，找出 XYZ 的最小最大值(極限)，我要當作我的背景的界線，這樣才不會超出範圍
# 你們可以自己去看json的格式
def find_plot_limits(data):
    all_kpts = []
    for frame_data in data:
        for i in frame_data.get('instances', []):
            all_kpts.extend(i['keypoints']) # 把每一幀、每一個人的 keypoints 加進 all_kpts
    if not all_kpts:
        return [-1, 1], [-1, 1], [-1, 1]
    all_kpts = np.array(all_kpts) # 等等可以直接用 max(), min()
    min_vals = np.min(all_kpts, axis=0)
    max_vals = np.max(all_kpts, axis=0)
    # 這邊是我發現輸出骨架有時候會太過於大，導致整個貼到背景，所以給個 padding
    padding = 0.3
    xlim = [min_vals[0] - padding, max_vals[0] + padding]
    ylim = [min_vals[1] - padding, max_vals[1] + padding]
    zlim = [min_vals[2] - padding, max_vals[2] + padding]
    return xlim, ylim, zlim

def process_json(json_path, output_video_path):
    print(f"\n正在處理: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 我要根據所有關節點，找最大最小座標，可以當背景板
    xlim, ylim, zlim = find_plot_limits(data)

    saved_frame_temp = [] 
    # 影像是由很多張照片組成，所以我們必須要有一個先暫存的位置把所有照片丟到裡面，然後等等再合成成一個 mp4
    for frame_data in tqdm(data, desc="繪製影格"): 
        # tqdm 顯示進度條，這個你們要自己下載(也可不用，不用的話要把 tqdm() 刪掉)，下載指令應該是 pip install tqdm
        frame_id = frame_data['frame_id']
        size = plt.figure(figsize=(8, 8)) # 尺寸設定成8x8
        picture = size.add_subplot(111, projection='3d') # 做 3D 空間背景

        # 解下來就是拆解 json 的東西了，跟我們之前轉檔的做法有點像
        for instance in frame_data.get('instances', []):
            kpts = np.array(instance['keypoints'], dtype=np.float32)  

            # 左右轉，這個很重要，因為我那時候就是左右相反，所以要再反過來。
            kpts[:,0] = -kpts[:,0]

            # 骨盆平移到中心
            kpts -= kpts[0]

            # 正規化長度 (骨盆到胸腔當作標準長度)
            # 理由是因為我們 json 本身就會有些跳動，可能忽大忽小，所以我們要先把距離取正規化
            scale = np.linalg.norm(kpts[8] - kpts[0])  # 8: 胸腔
            if scale > 0.05:
                kpts /= scale
            else:
                #print(f"[欸欸] {frame_id}: scale太小，跳過這幀")
                continue   

            track_id = instance.get('track_id', 0) # 以因應未來多人
            picture.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c="black", marker='o') # 點
             # 原
            for (p1_idx, p2_idx) in SKELETON:
                p1 = kpts[p1_idx]
                p2 = kpts[p2_idx]
                color = SKELETON_COLOR.get((p1_idx, p2_idx), 'gray')
                picture.plot( 
                    [p1[0], p2[0]], # X 軸的兩端
                    [p1[1], p2[1]], # Y 軸的兩端
                    [p1[2], p2[2]], # Z 軸的兩端
                    c=color,
                    linewidth=2
                )

        # 畫背景線
        picture.set_xlabel('X')
        picture.set_ylabel('Y')
        picture.set_zlabel('Z') # 我也不知道為甚麼沒遍色但反正輸出有看到就好。

        # 標題
        picture.set_title(f'Frame ID: {frame_id}')

        # 固定背景
        picture.set_xlim(xlim) 
        picture.set_ylim(ylim)
        picture.set_zlim(zlim)

        # elev: 俯視 ; azim: 左右角度
        picture.view_init(elev=20, azim=-80) # 改
        #x.view_init(elev=10, azim=-85) # 再改
        picture.grid(False) # 不畫背景格線，怕太雜
        picture.set_box_aspect([1,1,1]) # 3d背景長寬高
        #ax.set_box_aspect([0.5, 1, 1.5]) # 改

        frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.png") 
        plt.savefig(frame_path) # 把剛剛畫的3D圖存成圖片
        plt.close(size) #  關閉畫布，減少性能消耗
        saved_frame_temp.append(frame_path) # 把這張圖路徑記下來

    if not saved_frame_temp:
        print("[警告] 沒有偵測到任何幀，無法生成影片。")
        return

    print("正在合成影片...")
    first_frame = cv2.imread(saved_frame_temp[0])
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 格式
    video = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h)) # 30 幀

    # 讀圖片 -> 合成進影片 -> 刪照片
    for frame_path in tqdm(saved_frame_temp, desc="合成影片"):
        img = cv2.imread(frame_path)
        video.write(img)
        os.remove(frame_path)
    video.release()
    print(f"影片完成: {output_video_path}")

def main():
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    if not json_files:
        print("沒有找到任何 JSON 檔案。")
        return

    for json_name in json_files:
        json_path = os.path.join(INPUT_DIR, json_name)
        output_video_path = os.path.join(OUTPUT_DIR, json_name.replace('.json', '.mp4'))

        if os.path.exists(output_video_path):
            print(f"已存在，跳過: {output_video_path}")
            continue

        process_json(json_path, output_video_path)

main()

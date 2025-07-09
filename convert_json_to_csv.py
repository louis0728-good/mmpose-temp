import os
import json
import csv

# 我們用 human3.6m
our_keypoints_names = [
    '骨盆',
    '右髖', '右膝蓋', '右腳踝',       
    '左髖', '左膝蓋', '左腳踝', '脊椎',  
    '胸腔', '頭部', '鼻子', '左肩',
    '左手肘', '左手腕', '右肩', '右手肘',
    '右手腕'
]
# 關鍵點對應連結，https://blog.csdn.net/u010087338/article/details/131964886/ 
# 另外「胸腔」和「頸部」好像都可以互通，「鼻子」和「頭部」也是，但我還是把它改的更嚴謹。

def generate_the_csv_header(keypoints_names): # 這邊就是我單獨開一個，設定 csv檔的 header(就是標頭)的單位名稱區
    headers_row1 = [' ']
    headers_row2 = ['幀數'] # 只有在第二列開始才有單位

    for i in keypoints_names:
        headers_row1.extend([i, '', ''])  # 這裡很討厭，他成形後會長得像 | 頭部 |    |    |
                                            #                            |  x  |  y |  z |

                                            # 但我想要的是 |    頭部    |
                                            #             | x | y | z | ，反正目前照上面那樣很醜的作法，之後我再改，先求有。
        headers_row2.extend(['x', 'y', 'z'])
    return headers_row1, headers_row2

def convert(input_dir, output_dir):
    
    json_files = [ i for i in os.listdir(input_dir) if i.endswith('.json')]
    json_files.sort()
    if not json_files:
        print(f"沒找到 {input_dir} 的檔案路徑")
        return 

    header_row1 , header_row2 = generate_the_csv_header(our_keypoints_names)

    ep_kp_nums = len(our_keypoints_names) * 3
    #其實我原本想說用 17*3 就好，但是後面怕說未來可能會去減少偵測某些關鍵點，
    # 所以還是用彈性的寫法，這邊其實就只是為了讓格式好看點，所以硬設定 51 個大小(因為還有 xyz)

    for file_name in json_files:
        json_path = os.path.join(input_dir, file_name)

        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                this_file = json.load(f)

            if not isinstance(this_file, list):
                print(f"快去檢查為甚麼 {this_file} 不是 list 型態")
                continue

            curr_frames_data = [] # 本幀的數據都儲存起來。

            for i in this_file: # 開始一個一個跑了
                frame_id = i['frame_id'] # 這個你們可以自己去看 json 格式，會有"frame_id", "instances"(裡面有關鍵點的 xyz) 
                instances = i['instances']
            
                if instances:

                    """ 改版 2025-07-08，以因應 單人影像卻誤認為多人"""
                    # 先取出 instance 列表的第一個元素
                    first_item = instances[0]
                    if isinstance(first_item, list) and first_item: # 如果第一個東西是列表(錯誤，誤判為多人)
                        i_data = first_item[0] # 我目前只有做一個人的，之後我在想要怎麼處理多人的 mmpose 反正先這樣。

                    else: # 如果不是 (代表是正常格式)，就直接使用
                        i_data = first_item

                    keypoints = i_data.get('keypoints', [])
                    if keypoints and isinstance(keypoints[0], list) and len(keypoints[0]) > 1:
                        if len(keypoints) == 1:
                            keypoints = keypoints[0]
                            
                    rec_xyz = []
                    for kp in keypoints:
                        rec_xyz.extend(kp)
                    csv_row = [frame_id] + rec_xyz
                else:
                    # 代表這一幀甚麼東西都沒有
                    print(f" !! {this_file} 在 {frame_id} 幀甚麼東西都沒有")
                    rec_xyz = [''] * ep_kp_nums # 因為甚麼都沒有，所以就空值
                    csv_row = [frame_id] + rec_xyz

                curr_frames_data.append(csv_row) # 就把 x y z 還有 ID 丟進總列表裡面

            csv_file_name = os.path.splitext(file_name)[0] + ".csv" # 建立 csv 檔名，我是設定跟 json檔名一樣啦
            output_path = os.path.join(output_dir, csv_file_name)
            if(os.path.exists(output_path)):
                print(f"已有{file_name}，所以不覆蓋")
                continue # 這裡我有附上這個功能，免得我們之後如果對 csv 作筆記，筆記會在新一輪被刷新。

            
            with open(output_path, 'w', newline="", encoding='utf-8-sig') as o: 
                # 關於 newline="" ，下面連結你們如果有人有興趣可以往下滑到 解釋csv 那裏。 反正簡單講不會亂插入奇怪的符號像是 \r\n。
                # https://dev.to/codemee/python-csv-mo-zu-de-dang-an-kai-dang-shi-wei-shi-mo-yao-zhi-ding-newline-can-shu-wei--46ne
                writer = csv.writer(o)
                writer.writerow(header_row1)
                writer.writerow(header_row2)
                writer.writerows(curr_frames_data)
                print(f"成功輸出 {output_path}")

        except json.JSONDecodeError:
            print(f"{file_name} 不知道為甚麼無法開啟，去檢查")

        except Exception as e:
            print(f"{file_name} 有未知錯誤 {e}。")
    
def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # 一律建議使用相對路徑，這樣未來搬家不需要重調

    input_dir = os.path.join(file_dir, 'one_person') # 單人
    #input_dir = os.path.join(file_dir, 'multi_people') # 多人

    #output_dir_temp = os.path.dirname(os.path.abspath(file_dir)) # 取上一級的路徑
    output_dir = os.path.join(file_dir, 'result_csv', 'one_person') # 一樣單人
    #output_dir = os.path.join(file_dir, 'result_csv', 'multi_people') # 多人

    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 沒有就建立一個就好

    if not os.path.exists(input_dir):
        print(f"{input_dir} 找不到我們的 json 檔")
        return
    else:
        convert(input_dir, output_dir)

main()
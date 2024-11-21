import os, json, argparse

def convert(input_folder, output_folder):

    # 檢查輸出資料夾是否存在，若不存在則建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷資料夾以及所有子資料夾
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 檢查是否為 txt 檔案
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)

                s = ''
                
                # 讀取檔案內容
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 讀取每一行的內容
                        content = json.dumps(json.loads(line), ensure_ascii=False)
                        s += content + '\n'

                # 生成新檔案的名稱
                file_name, file_extension = os.path.splitext(file)
                new_file_path = os.path.join(output_folder, f"{file_name}.jsonl")

                # 將轉換後的內容寫入新檔案
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(s)

                print(f"已轉換並儲存：{new_file_path}")

# 取得 cmd 引數
parser = argparse.ArgumentParser(description='建立資料庫')
parser.add_argument('--input_folder', default='./data', help='資料來源路徑', type=str)
parser.add_argument('--output_folder', default='./tmp', help='轉換後的資料儲存目的路徑', type=str)
args = parser.parse_args()

# 指定需要轉換的資料夾路徑
input_folder = args.input_folder
output_folder = args.output_folder
convert(input_folder, output_folder)
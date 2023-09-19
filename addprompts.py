import os
from PIL import Image, PngImagePlugin

folder_path = os.getcwd()
folder_path = "D://ceshi//AutoV2V"

# 基础文件夹 里面的 输入文件夹
frame_path = "D:/AItools/Temporalnet2/originimages" #定义原始图像文件夹

# 基础文件夹 里面的 输出文件夹 
out_path = "D:/AItools/Temporalnet2/reamakeimages"


# tem2的py文件的全路径
tem2_file_path= "D:/AItools/Temporalnet2/temporalvideo.py"

# folder_path = "D:\\ceshi\\AutoV2V"
# frame_path = "D:\\ceshi\\AutoV2V\\2Yuantu\\ok"
# out_path = "D:\\ceshi\\AutoV2V\\3Datu\\ready"
# tem2_file_path = "D:\\ceshi\\tool\\temporalvideo.py"



# 定义文本文件状态
no_txt = 0

# 定义全局提示词
Gobal_prompt = " masterpiece, best quality, school uniform, cat ear,white shoes, white thighhighs,  "

# 定义全局反向提示词
Gobal_neg_prompt = "(worst quality, low quality:1.4),"

# 轮询输入目录
frame_files = [f for f in os.listdir(frame_path) if f.endswith('.png')]

txt_files = [f for f in os.listdir(frame_path) if f.endswith('.txt')]

frame_files1 = [f for f in os.listdir(out_path) if f.endswith('.png')]

if len(frame_files) == 0:
    print("裁切后图片目录中没有任何图片，请检查"+frame_path+"目录后重试。")
    quit()
if len(txt_files) == 0:
    print("未找到任何提示词文件，请使用wd14-tagger插件（或其他类似功能）生成提示词，放入"+frame_path+"目录后重试。")
    no_txt=1


# 输出文件夹不存在就创建
if not os.path.exists(out_path):
    os.makedirs(out_path)

for frame in frame_files:
    if frame in frame_files1:
        continue  # 跳过当前循环，继续下一次循环
    
    frame_file = os.path.join(frame_path,frame)
    
    txt_file = os.path.join(frame_path,f'{os.path.splitext(frame)[0]}.txt')
    
    tag = "masterpiece"
    if no_txt != 1:
        with open(txt_file, 'r') as t:
            tag = t.read()
    all_prompt = Gobal_prompt + tag
    

    # 载入单张图片基本参数
    im = Image.open(frame_file)

    frame_w,frame_h = im.size

    # os.system(f"python {tem2_file_path} --prompt {all_prompt} --negative-prompt {Gobal_neg_prompt} --init-image {frame} --input-dir {frame_path} --output-dir {out_path} --width {frame_w} --height {frame_h}")
    os.system(f'python {tem2_file_path} --prompt "{all_prompt}" --negative-prompt "{Gobal_neg_prompt}" --input-dir {frame_path} --output-dir {out_path} ')

    
    # print(f'python {tem2_file_path} --prompt "{all_prompt}" --negative-prompt "{Gobal_neg_prompt}" --init-image {frame_file} --input-dir {str(frame_path)} --output-dir {out_path} --width {frame_w} --height {frame_h}')

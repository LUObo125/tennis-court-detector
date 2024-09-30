import os
import shutil
import cv2
import torch
from main import TennisCourtClassifier, transforms
from PIL import Image

# 加载模型
model = TennisCourtClassifier()
model.load_state_dict(torch.load('tennis_court_classifier.pth'))
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义视频文件夹路径
video_folder = './input'  # 替换为你的视频文件夹路径

# 获取文件夹中所有视频文件
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 循环处理每个视频文件
for i, video_file in enumerate(video_files):
    print("processing video ", i)
    video_path = os.path.join(video_folder, video_file)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化变量
    frame_count = -1
    start_frame = 0
    end_frame = 0
    is_playing = False
    output_video_count = 0

    # 循环处理每一帧
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 预处理帧
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform(Image.fromarray(image)).unsqueeze(0)

        # 使用模型预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        # 判断是否识别到球场
        if predicted.item() == 0:  # 类别 0 表示球场
            if not is_playing:
                start_frame = frame_count
                is_playing = True
        else:
            if is_playing:
                end_frame = frame_count
                is_playing = False

                # 保存视频片段
                if end_frame - start_frame > 4 * fps:  # 最小片段长度为 4 秒
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(f'output_video_{output_video_count}.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
                    for i in range(start_frame, end_frame-1):
                        ret, frame = cap.read()
                        if ret:
                            out.write(frame)
                    out.release()
                    output_video_count += 1

    # 释放资源
    cap.release()

    # 获取当前目录下的所有视频文件
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]

    # 创建 output 文件夹
    if not os.path.exists('output'):
        os.makedirs('output')

    # 获取 output 文件夹中已存在的视频文件数量
    existing_files = [f for f in os.listdir('output') if f.endswith('.mp4')]
    next_file_number = len(existing_files)

    # 将视频文件移动到 output 文件夹并重命名
    for i, file in enumerate(video_files):
        source_path = os.path.join('.', file)
        destination_path = os.path.join('output', f'{next_file_number + i}.mp4')
        shutil.move(source_path, destination_path)

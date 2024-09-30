import os
import random

def rename_images(image_folder):
  """
  将图像从0开始按顺序重新命名。

  Args:
    image_folder: 图像所在的文件夹路径。
  """
  image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
  random.shuffle(image_files)
  for i, filename in enumerate(image_files):
    old_path = os.path.join(image_folder, filename)
    new_filename = f"{i}.jpg"  # 您可以更改文件扩展名
    new_path = os.path.join(image_folder, new_filename)
    os.rename(old_path, new_path)


if __name__ == "__main__":
    # 使用示例
    image_folder = 'out_folder/no_court'  # 替换为您的图像文件夹路径
    rename_images(image_folder)
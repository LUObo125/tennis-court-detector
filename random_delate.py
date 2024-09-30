""" import os
import random
import shutil

def rename_and_remove_images(image_folder):
  """
"""   随机删除一半的图像，并将剩下的图像从0开始按顺序重新命名。

Args:
image_folder: 图像所在的文件夹路径。 """
"""
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)


for i, image in enumerate(image_files):
new_name = f"{i}.jpg"  # 假设图像格式为jpg，可以根据需要修改
os.rename(os.path.join(image_folder, image), os.path.join('out_folder/have_court', new_name))

if __name__ == "__main__":
    image_folder = 'out_folder/has_court'  # 替换为您的图像文件夹路径
    rename_and_remove_images(image_folder) """

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
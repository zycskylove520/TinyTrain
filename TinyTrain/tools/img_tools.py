import os

from PIL import Image


def convert_images_format(input_dir, output_dir, input_format, output_format):
    """
    批量将指定目录中的图片从一种格式转换为另一种格式。
    :param input_dir: 包含要转换的图片的目录
    :param output_dir: 转换后的图片保存的目录
    :param input_format: 原图片格式（例如 'jpg', 'png' 等）
    :param output_format: 转换后的图片格式（例如 'png', 'jpg' 等）
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"输入目录 {input_dir} 不存在！")
        return

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_list = os.listdir(input_dir)
    num = len(img_list)

    # 遍历输入目录中的所有文件
    for i, filename in enumerate(img_list):
        # 检查文件扩展名是否匹配输入格式
        if filename.lower().endswith(f".{input_format.lower()}"):
            # 构造完整的输入文件路径
            input_path = os.path.join(input_dir, filename)
            # 构造输出文件路径
            output_filename = os.path.splitext(filename)[0] + f".{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)

            try:
                # 打开图片
                with Image.open(input_path) as img:
                    # 保存为新的格式
                    img.save(output_path)
                    print(f"一共{num}张图片，当前第{i}张图片 {filename} 已成功转换并保存为 {output_filename}")
            except Exception as e:
                print(f"处理图片 {filename} 时发生错误: {e}")

# 示例用法
if __name__ == "__main__":
    input_directory = r"E:\DownLoad\huajin\door\img2"  # 包含要转换的图片的目录
    output_directory = r"E:\DownLoad\huajin\door\img2s"  # 转换后的图片保存的目录
    input_format = "png"  # 原图片格式
    output_format = "jpg"  # 转换后的图片格式

    convert_images_format(input_directory, output_directory, input_format, output_format)
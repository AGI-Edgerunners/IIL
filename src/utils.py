import base64
import json
import math
import os

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
import jieba


def write_json(datas, file):
    with open(file, mode="w+", encoding="utf-8") as f:
        json.dump(datas, f, indent=4, ensure_ascii=False)
        f.close()


def load_json(file):
    with open(file, mode="r", encoding="utf-8") as f:
        datas = json.load(f)
        f.close()
    return datas


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def text_image_concat_v1(text: str, image_file: str, saving_file: str):
    image = mpimg.imread(image_file)
    plt.title(text)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(saving_file)
    return


def text_image_concat(text: str, image_file: str, saving_file: str, demo_file=None, height: int = 0, width: int = 0,
                      text_pos: str = "b",
                      image_pos: str = "c", bold=False, text_size=20):
    """
    :param text: concatenating text
    :param image_file: concatenating image (path)
    :param saving_file: path to save concatenated image
    :param height: set the height concatenated image (optional)
    :param width: set the width concatenated image (optional)
    :param text_pos: position of the text (t for top on the image，b for bottom of the image)
    :param image_pos: position of image (c for aligning center and l for aligning left)
    :param bold: bold text or not (True or False)
    :param text_size: font size of text (default 20)
    :return:
    """
    image = Image.open(image_file)
    if demo_file:
        width = get_image_size(demo_file)[1]
    if bold:
        font_file = "src/Roboto-Bold.ttf"
    else:
        font_file = "src/AaBianYaKai/AaBianYaKai/AaBianYaKai-2.ttf"
    total_width, line_height = calculate_text_width(text, text_size, font_file)
    upper_letter_width, _ = calculate_text_width("ABCD", text_size, font_file)
    lower_letter_width, _ = calculate_text_width("abcd", text_size, font_file)
    chinese_letter_width, _ = calculate_text_width("你好，啊", text_size, font_file)
    u_l_pix_per_l = upper_letter_width/4
    l_l_pex_per_l = lower_letter_width/4
    c_l_pix_per_l = chinese_letter_width/4
    line_pix = max(width, image.width) - 10
    if line_pix < 250:
        line_pix = int(1.8 * line_pix)
    # char_per_pix = len(text) / total_width
    # char_per_line = int(char_per_pix * (image.width-20))
    lines = []
    line_words = ''

    seg_list = jieba.cut(text, cut_all=False)
    for word in seg_list:
        try_line = line_words + ' ' + word if line_words else word
        # 大写字母数量 count of uppercase
        u_c = sum(1 for char in try_line if char.isupper())
        # 中文字数量 count of chinese characters
        c_c = sum(1 for char in try_line if '\u4e00' <= char <= '\u9fff')
        # 小写字母数量 count of lowercase
        l_c = len(try_line) - u_c - c_c
        if '\n' in word:
            temp_lines = word.split('\n')
            for temp_idx, temp_word in enumerate(temp_lines):
                try_line = line_words + '' + temp_word if line_words else temp_word
                # 大写字母数量
                u_c = sum(1 for char in try_line if char.isupper())
                # 中文字数量
                c_c = sum(1 for char in try_line if '\u4e00' <= char <= '\u9fff')
                # 小写字母数量
                l_c = len(try_line) - u_c - c_c
                if u_c*u_l_pix_per_l+c_c*c_l_pix_per_l+l_c*l_l_pex_per_l <= line_pix:
                    line_words = line_words + '' + temp_word if line_words else temp_word
                    if temp_idx == len(temp_lines) - 1:
                        pass
                    else:
                        lines.append(line_words)
                        line_words = ''
                else:
                    lines.append(line_words)
                    line_words = temp_word
                    if temp_idx == len(temp_lines) - 1:
                        pass
                    else:
                        lines.append(line_words)
                        line_words = ''
        elif u_c*u_l_pix_per_l+c_c*c_l_pix_per_l+l_c*l_l_pex_per_l <= line_pix:
            line_words = line_words + '' + word if line_words else word
        else:
            lines.append(line_words)
            line_words = word
    lines.append(line_words)

    height = max(height, image.height) + len(lines) * line_height
    width = max(width, image.width)
    if width < 250:
        width = int(1.8 * width)
    to_image = Image.new(
        "RGB",
        (width, height),
        color=(255, 255, 255)
    )
    image_loc = [0, 0]
    lines_loc = []
    if image_pos == "c":
        image_loc[0] = int((width - image.width) / 2)
    else:
        raise NotImplementedError()
    for line_idx, line in enumerate(lines):
        if text_pos == 'b':
            lines_loc.append([20, image.height + line_idx * line_height])
            image_loc[1] = 0
        elif text_pos == 't':
            image_loc[1] = (line_idx + 1) * line_height
            lines_loc.append([20, line_idx * line_height])
        else:
            raise NotImplementedError()

    to_image.paste(image, (image_loc[0], image_loc[1]))
    to_image.save(saving_file)
    font = ImageFont.truetype(font_file, size=text_size)
    draw = ImageDraw.Draw(to_image)
    for loc, line in zip(lines_loc, lines):
        draw.text((loc[0], loc[1]), line, fill=(0, 0, 0), font=font)

    to_image.save(saving_file)
    return


def calculate_text_width(text, font_size, font_path=None):
    # Load font
    font = ImageFont.truetype(font_path, size=font_size) if font_path else ImageFont.load_default()

    # Get the bounding box of the text
    bbox = font.getbbox(text)

    # Calculate the width (right coordinate of the bounding box)
    text_width, text_height = bbox[2], bbox[3]

    return text_width, text_height


def images_concat(image_file1: str, image_file2: str, saving_file: str, pos="l"):
    """
    图片上下拼接
    :param image_file1: path of first image
    :param image_file2: path of second image
    :param saving_file: path to save concatenated image
    :param pos: images position (c for aligning center and l for aligning left)
    :return:
    """
    rom_image_1 = Image.open(image_file1)
    rom_image_2 = Image.open(image_file2)
    to_width = max(rom_image_1.width, rom_image_2.width)
    to_height = rom_image_1.height + rom_image_2.height
    to_image = Image.new(
        'RGB',
        (to_width, to_height),
        (255, 255, 255)
    )
    if pos == "l":
        to_image.paste(rom_image_1, (0, 0))
        to_image.paste(rom_image_2, (0, rom_image_1.height))
        # to_image.save(saving_file)
    elif pos == "c":
        width_start_1 = int((to_width - rom_image_1.width) / 2)
        width_start_2 = int((to_width - rom_image_2.width) / 2)
        to_image.paste(rom_image_1, (width_start_1, 0))
        to_image.paste(rom_image_2, (width_start_2, rom_image_1.height))
        # to_image.save(saving_file,quality=20)
    else:
        NotImplementedError()
    to_image.save(saving_file)
    # if os.path.getsize(saving_file) >= 19 * 1024:
    #     to_image.save(saving_file, quality=10)
    # else:
    #     to_image.save(saving_file)
    return


def image_paste_anywhere(base_image_file: str, operate_image_file: str, saving_file: str, x: int, y: int):
    rom_image_1 = Image.open(base_image_file)
    rom_image_2 = Image.open(operate_image_file)
    to_width = max(rom_image_1.width, rom_image_2.width + x + 10)
    to_height = max(rom_image_1.height, rom_image_2.height + y + 10)
    to_image = Image.new(
        'RGB',
        (to_width, to_height),
        (255, 255, 255)
    )
    to_image.paste(rom_image_1, (0, 0))
    to_image.paste(rom_image_2, (x, y))
    to_image.save(saving_file)
    return


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def get_image_size(image_file: str):
    """

    :param image_file:
    :return: (height, width)
    """
    image = Image.open(image_file)
    return image.height, image.width


def image_resize(image_file: str, scale: float, saving_file: str):
    rate = math.sqrt(scale)
    image = Image.open(image_file)
    size = image.size
    new_size = (int(size[0] * rate), int(size[1] * rate))
    new_image = image.resize(new_size)
    new_image.save(saving_file)


def image_rotating_concat(image_file1: str, image_file2: str, saving_file: str, pos="l"):
    rom_image_1 = Image.open(image_file1)

    rom_image_2 = Image.open(image_file2)
    # 旋转图片，参数为旋转角度
    rotated_rom_image_2 = rom_image_2.rotate(45, expand=True, fillcolor=(255, 255, 255))  # 这里的90表示逆时针旋转90度

    to_width = max(rom_image_1.width, rotated_rom_image_2.width)
    to_height = rom_image_1.height + rotated_rom_image_2.height
    to_image = Image.new(
        'RGB',
        (to_width, to_height),
        (255, 255, 255)
    )
    if pos == "l":
        to_image.paste(rom_image_1, (0, 0))
        to_image.paste(rotated_rom_image_2, (0, rom_image_1.height))
        to_image.save(saving_file)
    elif pos == "c":
        width_start_1 = int((to_width - rom_image_1.width) / 2)
        width_start_2 = int((to_width - rotated_rom_image_2.width) / 2)
        to_image.paste(rom_image_1, (width_start_1, 0))
        to_image.paste(rotated_rom_image_2, (width_start_2, rom_image_1.height))
        to_image.save(saving_file)
    else:
        NotImplementedError()
    return


def images_crop_concat(image_file1: str, image_file2: str, saving_file: str, pos='c'):
    rom_image_1 = Image.open(image_file1)
    rom_image_2 = Image.open(image_file2)
    cropped = rom_image_2.crop((0, 0, rom_image_2.width, int(rom_image_2.height / 3)))
    to_width = max(rom_image_1.width, cropped.width)
    to_height = rom_image_1.height + cropped.height
    to_image = Image.new(
        'RGB',
        (to_width, to_height),
        (255, 255, 255)
    )
    if pos == "l":
        to_image.paste(rom_image_1, (0, 0))
        to_image.paste(cropped, (0, rom_image_1.height))
        to_image.save(saving_file)
    elif pos == "c":
        width_start_1 = int((to_width - rom_image_1.width) / 2)
        width_start_2 = int((to_width - cropped.width) / 2)
        to_image.paste(rom_image_1, (width_start_1, 0))
        to_image.paste(cropped, (width_start_2, rom_image_1.height))
        to_image.save(saving_file)
    else:
        NotImplementedError()
    return


def image_upload(image_file):
    api_key = ''
    url = ''
    # url = "https://api.imgbb.com/1/upload"
    with open(image_file, 'rb') as f:
        files = {'image': (image_file, f)}
        params = {'key': api_key}
        response = requests.post(url, files=files, params=params)

    image_link = response.json()['data']['url']
    return image_link


def copy_image(image_file: str, save_file: str):
    source_image = Image.open(image_file)
    copied_image = source_image.copy()
    copied_image.save(save_file)
    return

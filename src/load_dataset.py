import json
import os
import re
import warnings
from typing import Union
import ast
# from dataset.halllusionbench_text import get_demo_text
import pandas as pd

from src.utils import load_json, text_image_concat, create_dir, images_concat, get_image_size, image_resize, \
    image_paste_anywhere, image_rotating_concat, images_crop_concat


def load_hallusionbench():
    dataset_root = "dataset/hallusion_bench"
    data_file = f"{dataset_root}/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    for raw_idx in range(len(df_datas)):
        data = df_datas[raw_idx]
        if data.get("visual_input"):
            image_file = f"{dataset_root}/{data.get('category')}/{data.get('subcategory')}/{data.get('set_id')}_{data.get('figure_id')}.png"
        else:
            image_file = None
        text = data.get('question')
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}-{data.get('figure_id')}-{data.get('question_id')}"
        if image_file is not None:
            datas.append(
                {"id": id_, "text": text, "image_file": image_file, "label": answer, "label_detail": answer_detail}
            )

    return datas


def load_hallusionbench_ticl():
    """
    text only
    :return:
    """
    pass


def load_hallusionbench_vticl(learning_type: str):
    """
    image&text
    :param learning_type:
    :return:
    """
    dataset_root = "dataset/hallusion_bench"
    data_file = f"{dataset_root}/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    for raw_idx in range(len(df_datas)):
        data = df_datas[raw_idx]
        if data.get("visual_input"):
            test_file = f"{dataset_root}/{data.get('category')}/{data.get('subcategory')}/{data.get('set_id')}_{data.get('figure_id')}.png"
            test_text = data.get('question')
        else:
            test_file = None
            continue
        image_inputs = [test_file]
        text_inputs = [test_text]
        if learning_type == "few_shot":
            # todo: 指定demo图片路径和文本
            demo_file = ""
            demo_text = ""
            image_inputs = [demo_file, test_file]
            text_inputs = [demo_text, test_text]
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}-{data.get('figure_id')}-{data.get('question_id')}"
        if test_file is not None:
            datas.append(
                {"id": id_, "text": test_text, "image_file": test_file, "label": answer, "label_detail": answer_detail,
                 "image_inputs": image_inputs, "text_inputs": text_inputs}
            )

    return datas


def load_hallusionbench_iil(learning_type: str):
    """

    :param learning_type: zero_shot/few_shot
    :return:
    """
    ori_dataset_root = "dataset/hallusion_bench"
    dataset_root = "dataset/hallusion_bench_iil"
    create_dir(dataset_root)
    ori_data_file = f"{ori_dataset_root}/HallusionBench.tsv"
    df_datas = pd.read_csv(ori_data_file, sep='\t').to_dict("records")
    datas = []
    for raw_idx in range(len(df_datas)):
        data = df_datas[raw_idx]
        if data.get("visual_input"):
            ori_image_file = f"{ori_dataset_root}/{data.get('category')}/{data.get('subcategory')}/{data.get('set_id')}_{data.get('figure_id')}.png"
        else:
            ori_image_file = None
            continue
        create_dir(f"{dataset_root}/{data.get('category')}")
        create_dir(f"{dataset_root}/{data.get('category')}/{data.get('subcategory')}")
        zs_image_file = f"{dataset_root}/{data.get('category')}/{data.get('subcategory')}/{data.get('set_id')}_{data.get('figure_id')}.png"
        text = data.get('question')
        text_image_concat(text, ori_image_file, zs_image_file)
        image_file = zs_image_file
        if learning_type == 'few_shot':
            # todo: 初始化demo图片路径和拼接后的图片路径
            demo_file = ""
            fs_image_file = ""
            images_concat(demo_file, zs_image_file, fs_image_file)
            image_file = fs_image_file
        text = data.get('question')
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}-{data.get('figure_id')}-{data.get('question_id')}"
        if image_file is not None:
            datas.append(
                {"pid": id_, "text": text, "image_file": ori_image_file, "label": answer, "label_detail": answer_detail,
                 "image_inputs": [image_file]}
            )

    return datas


def load_hallusionbench_textonly_demo_discription():
    dataset_root = "dataset/hallusion_bench_demonstration"
    data_file = "dataset/hallusion_bench/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    image_file_exists = []
    for raw_idx in range(len(df_datas)):
        data = df_datas[raw_idx]
        image_file = f"{dataset_root}/{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}.png"
        if os.path.exists(image_file):
            pass
        else:
            continue
        question = data.get('question')
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}"
        text = "Generate detailed caption for the image. Use natural language to describe the image"
        if image_file not in image_file_exists:
            image_file_exists.append(image_file)
            if image_file is not None:
                datas.append(
                    {"id": id_, "text": text, "image_file": image_file}
                )

    return datas


def load_hallusionbench_textonly():
    dataset_root = "dataset/hallusion_bench_demonstration"
    demo_captions_file = 'result/expforhb_demo_caption-few_shot-hallusionbench_textonly_demo_discription-text&image.json'
    test_image_captions_file = 'result/expforhb-hallusionbench-text_only.json'
    data_file = "dataset/hallusion_bench/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    demo_captions = load_json(demo_captions_file)
    demos = {}
    for demo_caption in demo_captions:
        demos[demo_caption] = [demo_caption, demo_captions[demo_caption]['prediction']]
    test_image_captions = load_json(test_image_captions_file)
    for test_image_caption in test_image_captions:
        id = test_image_caption
        text = test_image_captions[test_image_caption]['text']
        answer = test_image_captions[test_image_caption]['label']
        answer_detail = test_image_captions[test_image_caption]['prediction']
        image_description = test_image_captions[test_image_caption]['image_description']
        demo_id = id.split('-')[0] + '-' + id.split('-')[1] + '-' + id.split('-')[2]
        if demo_id in demos:
            new_text = 'Demonstration Example:\n' + demos[demo_id][
                1] + '\nTest Example:\n' + image_description + '\n' + text
            datas.append(
                {"id": id, "demo_caption": demos[demo_id][1], "image_description": image_description, "text": new_text,
                 "label": answer, "label_detail": answer_detail}
            )

    return datas


def load_mathvista(args):
    dataset_root = "dataset/mathvista_testmini/testmini"
    # image_root = "dataset/HallusionBench_few_shot_higher"
    data_file = f"{dataset_root}/testmini.json"
    # df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        # data = df_datas[raw_idx]
        if data[idx]['category'] == args.category and data[idx]['context'] == args.context:
            image_file = f"{dataset_root}/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
        else:
            image_file = None
        text = data[idx]['question']
        choices = None
        if 'Choices' in data[idx]['query']:
            choices = 'Choices ' + data[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = data[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append({
                "pid": id_,
                "image_path": data[idx]['image_path'],
                "image_file": image_file,
                "category": data[idx]["category"],
                "context": data[idx]["context"],
                "grade": data[idx]["grade"],
                "img_height": data[idx]["img_height"],
                "img_width": data[idx]["img_width"],
                "language": data[idx]["language"],
                "skills": data[idx]["skills"],
                "source": data[idx]["source"],
                "split": data[idx]["split"],
                "task": data[idx]["task"],
                "question": data[idx]["question"],
                "query": data[idx]["query"],
                "choices": choices,
                "answer": data[idx]["answer"]
            })
    return datas


def load_mathvista_ticl(learning_type: str, category: str, sub_category: str):
    ori_dataset_root = "dataset/mathvista_testmini/testmini"
    data_file = f"{ori_dataset_root}/testmini.json"
    ori_datas = load_json(data_file)
    datas = []
    question_prompt = """Answer the question according to the description.
Description: {description}
{question}
Answer: """
    caption_prompt = "Generate detailed caption for the image."
    for raw_idx in range(len(ori_datas)):
        data = ori_datas[raw_idx]
        if data.get('category') == category and data.get('context') == sub_category:
            test_file = f"{ori_dataset_root}/{data.get('category')}/{data.get('context')}/images/{data.get('pid')}.png"
            test_text = data.get("query", "")
        else:
            continue
        if test_file is not None and not os.path.exists(test_file):
            warnings.warn(f"找不到文件：{test_file}")
            continue
        image_inputs = []
        text_inputs = [question_prompt]
        image_inputs_for_caption = [test_file]
        text_inputs_for_caption = [caption_prompt]
        if learning_type == "few_shot":
            demo_caption_prompt = "Generate detailed caption for the image.\n{caption}"
            # todo: 指定demo图片路径
            fs_question_prompt_dir = "dataset/mathvista_testmini/ticl_prompt"
            with open(f"{fs_question_prompt_dir}/{data.get('category')}_{data.get('context')}.txt", 'r',
                      encoding='utf-8') as f:
                fs_question_prompt = f.read()
                f.close()
            text_inputs = [fs_question_prompt]
            demo_file_dir = "dataset/mathvista_testmini/image"
            demo_caption_dir = "dataset/mathvista_testmini/demo_caption"
            demo_file = f"{demo_file_dir}/{data.get('category')}_{data.get('context')}.png"
            with open(f"{demo_caption_dir}/{data.get('category')}_{data.get('context')}.txt", 'r',
                      encoding='utf-8') as f:
                demo_caption = f.read()
            demo_caption = demo_caption_prompt.format(caption=demo_caption)
            image_inputs_for_caption = [demo_file, test_file]
            text_inputs_for_caption = [demo_caption, caption_prompt]
        choices = None
        if 'Choices' in data['query']:
            choices = 'Choices ' + data['query'].split('Choices')[1].replace('\n', ' ')
        pid = data['pid']
        if test_file is not None:
            datas.append(
                {
                    "pid": pid,
                    "text": test_text,
                    "image_file": test_file,
                    "label": data["answer"],
                    "image_inputs": image_inputs,
                    "text_inputs": text_inputs,
                    "image_inputs_for_caption": image_inputs_for_caption,
                    "text_inputs_for_caption": text_inputs_for_caption,
                    "image_path": data['image_path'],
                    "category": data["category"],
                    "context": data["context"],
                    "grade": data["grade"],
                    "img_height": data["img_height"],
                    "img_width": data["img_width"],
                    "language": data["language"],
                    "skills": data["skills"],
                    "source": data["source"],
                    "split": data["split"],
                    "task": data["task"],
                    "question": data["question"],
                    "query": data["query"],
                    "choices": choices,
                    "answer": data["answer"]
                }
            )
    return datas


def load_mathvista_vticl(learning_type: str, category: str, sub_category: str):
    ori_dataset_root = "dataset/mathvista_testmini/testmini"
    data_file = f"{ori_dataset_root}/testmini.json"
    ori_datas = load_json(data_file)
    datas = []
    for raw_idx in range(len(ori_datas)):
        data = ori_datas[raw_idx]
        if data.get('category') == category and data.get('context') == sub_category:
            test_file = f"{ori_dataset_root}/{data.get('category')}/{data.get('context')}/images/{data.get('pid')}.png"
            test_text = data.get("query", "")
        else:
            continue
        if test_file is not None and not os.path.exists(test_file):
            warnings.warn(f"找不到文件：{test_file}")
            continue
        image_inputs = [test_file]
        text_inputs = [test_text]
        if learning_type == "few_shot":
            demo_file_dir = "dataset/mathvista_testmini/demo_file_vision_cue"
            demo_text_dir = "dataset/mathvista_testmini/demo_text"
            demo_file = f"{demo_file_dir}/{data.get('category')}_{data.get('context')}_demo.png"
            with open(f"{demo_text_dir}/{data.get('category')}_{data.get('context')}.txt", 'r', encoding='utf-8') as f:
                demo_text = f.read()
                f.close()
            image_inputs = [demo_file, test_file]
            text_inputs = [demo_text, test_text]
        choices = None
        if 'Choices' in data['query']:
            choices = 'Choices ' + data['query'].split('Choices')[1].replace('\n', ' ')
        pid = data['pid']
        if test_file is not None:
            datas.append(
                {
                    "pid": pid,
                    "text": test_text,
                    "image_file": test_file,
                    "label": data["answer"],
                    "image_inputs": image_inputs,
                    "text_inputs": text_inputs,
                    "image_path": data['image_path'],
                    "category": data["category"],
                    "context": data["context"],
                    "grade": data["grade"],
                    "img_height": data["img_height"],
                    "img_width": data["img_width"],
                    "language": data["language"],
                    "skills": data["skills"],
                    "source": data["source"],
                    "split": data["split"],
                    "task": data["task"],
                    "question": data["question"],
                    "query": data["query"],
                    "choices": choices,
                    "answer": data["answer"]
                }
            )
    return datas


def load_mathvista_iil(learning_type: str, category: str, sub_category: str):
    ori_dataset_root = "dataset/mathvista_testmini/testmini"
    dataset_root = "dataset/mathvista_testmini/testmini_iil"
    create_dir(dataset_root)
    data_file = f"{ori_dataset_root}/testmini.json"
    ori_datas = load_json(data_file)
    datas = []
    for raw_idx in range(len(ori_datas)):
        data = ori_datas[raw_idx]
        if data.get('category') == category and data.get('context') == sub_category:
            ori_image_file = f"{ori_dataset_root}/{data.get('category')}/{data.get('context')}/images/{data.get('pid')}.png"
        else:
            continue
        if ori_image_file is not None and not os.path.exists(ori_image_file):
            warnings.warn(f"找不到文件：{ori_image_file}")
            continue
        create_dir(f"{dataset_root}/{data.get('category')}/{data.get('context')}/images")
        zs_image_file = f"{dataset_root}/{data.get('category')}/{data.get('context')}/images/{data.get('pid')}.png"
        text = data.get("query", "")
        text_image_concat(text, ori_image_file, zs_image_file)
        image_file = zs_image_file
        choices = None
        if 'Choices' in data['query']:
            choices = 'Choices ' + data['query'].split('Choices')[1].replace('\n', ' ')
        pid = data['pid']
        if learning_type == 'few_shot':
            demo_dir = "dataset/mathvista_testmini/demo_file"
            demo_file = f"{demo_dir}/{data.get('category')}_{data.get('context')}_demo.png"
            fs_image_file = f"{dataset_root}/{data.get('category')}/{data.get('context')}/images/fs_{data.get('pid')}.png"
            images_concat(demo_file, zs_image_file, fs_image_file)
            image_file = fs_image_file
        if image_file is not None:
            datas.append(
                {
                    "pid": pid,
                    "text": text,
                    "image_file": ori_image_file,
                    "label": data["answer"],
                    "image_inputs": [image_file],
                    "image_path": data['image_path'],
                    "category": data["category"],
                    "context": data["context"],
                    "grade": data["grade"],
                    "img_height": data["img_height"],
                    "img_width": data["img_width"],
                    "language": data["language"],
                    "skills": data["skills"],
                    "source": data["source"],
                    "split": data["split"],
                    "task": data["task"],
                    "question": data["question"],
                    "query": data["query"],
                    "choices": choices,
                    "answer": data["answer"]
                }
            )
    return datas


def load_vqa100():
    dataset_root = "dataset/vqa2_sampled100"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    datas = []
    for data in ori_datas:
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        datas.append(
            {"id": str(data.get("question_id")), "text": data.get("question"), "image_file": image_file, "label": ""}
        )
    return datas


def load_counting100():
    dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    datas = []
    for data in ori_datas:
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        answer_file = f"{dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        datas.append(
            {"id": str(data.get("question_id")), "text": data.get("question"), "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50():
    dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    datas = []
    for data in ori_datas:
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        answer_file = f"{dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        datas.append(
            {"id": data.get("question_id"), "text": data.get("question"), "image_file": image_file, "label": answer}
        )
    return datas


def load_yesorno50_imageonly():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        if os.path.exists(image_file):
            pass
        else:
            text = data.get('question')
            text_image_concat(text, ori_image_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        datas.append(
            {"id": data.get("question_id"), "text": data.get("question"), "image_file": image_file, "label": answer}
        )
    return datas


def load_counting100_imageonly():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    create_dir(dataset_root)
    datas = []
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        if os.path.exists(image_file):
            pass
        else:
            text = data.get('question')
            text_image_concat(text, ori_image_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        datas.append(
            {"id": str(data.get("question_id")), "text": data.get("question"), "image_file": image_file,
             "label": answer}
        )
    return datas


def load_vqa100_imageonly():
    ori_dataset_root = "dataset/vqa2_sampled100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/vqa2_sampled100_imageonly"
    create_dir(dataset_root)
    datas = []
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        if os.path.exists(image_file):
            pass
        else:
            text = data.get('question')
            text_image_concat(text, ori_image_file, image_file)
        # answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        # with open(answer_file, mode="r", encoding="utf-8") as f:
        #     answer = f.read()
        #     f.close()
        datas.append(
            {"id": str(data.get("question_id")), "text": data.get("question"), "image_file": image_file, "label": ""}
        )
    return datas


def load_math_equation():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    datas = []
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[50:]
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_path = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"
        image_file = f"{dataset_root}/images/cropped_with_demo_{id_}.jpg"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = r"""# Demonstration Example1
Determine if the sequence is increasing, decreasing, or not monotonic:
$$\left\{\frac{2 n^2-1}{2 n}\right\}_{n=2}^{\infty}$$

Solution:
To determine if the given sequence \(\left\{\frac{2 n^2-1}{2 n}\right\}_{n=2}^{\infty}\) is increasing, decreasing, or not monotonic, we need to examine how the terms of the sequence change as \( n \) increases.

The general term of the sequence is given by:
\[ a_n = \frac{2n^2 - 1}{2n} \]

To check for monotonicity, we'll compare consecutive terms \( a_n \) and \( a_{n+1} \). If \( a_{n+1} > a_n \) for all \( n \geq 2 \), then the sequence is increasing. If \( a_{n+1} < a_n \) for all \( n \geq 2 \), then the sequence is decreasing. If neither condition is satisfied consistently, the sequence is not monotonic.

Let's calculate \( a_{n+1} - a_n \) and analyze the result.

The difference between consecutive terms of the sequence is given by:
\[ \Delta a_n = a_{n+1} - a_n = \frac{n^2 + n + \frac{1}{2}}{n(n + 1)} \]

Now, we need to analyze the sign of \( \Delta a_n \) as \( n \) varies from 2 to infinity.

- The numerator \( n^2 + n + \frac{1}{2} \) is always positive for \( n \geq 2 \) because it's a sum of positive terms.
- The denominator \( n(n + 1) \) is also always positive for \( n \geq 2 \) since it's a product of two positive integers.

Since both the numerator and denominator are positive, \( \Delta a_n \) is positive for all \( n \geq 2 \). This implies that \( a_{n+1} > a_n \) for all \( n \geq 2 \), and therefore, the sequence is increasing.

# Demonstration Example2
Calculate the limit, if it exists:
$$\lim _{x \rightarrow 2}\left(8-3 x+6 x^2\right)$$

Solution:
To find the limit of the given expression as $x$ approaches $2$, we substitute $x=2$ into the expression and evaluate it. 
$$\lim _{x \rightarrow 2}\left(8-3 x+6 x^2\right) = 8-3(2)+6(2)^2 = 8-6+24 = 26$$
Therefore, the limit of the expression as $x$ approaches $2$ is $26$.

Learn from above examples and solve the following test example
# Test Example
""" + f.read()
            # text = f.read()
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        original_img_file = image_path
        demo_file = "dataset/Sampled_Data_For_Experiments/Math/math_equ_demo_v3.jpg"
        # images_concat(demo_file, original_img_file, image_file, pos="c")
        images_crop_concat(demo_file, original_img_file, image_file, pos="c")
        # image_file = remote_imgs[ids.index(id_)]
        # match = re.search("(?<=d/).*?(?=/view)", remote_imgs[ids.index(id_)])
        # file_id = match.group() if match else ""
        # image_file = f"https://drive.google.com/u/0/uc?id={file_id}&export=download"
        datas.append({"id": id_, "text": text, "image_file": image_file, "original_img_file": original_img_file,
                      "label": answer})
    return datas


def load_math_equation_zs():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[50:]
    datas = []
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_path = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = f.read()
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        image_file = image_path
        datas.append({"id": id_, "text": text, "image_file": image_file, "label": answer})
    return datas


def load_math_wp():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    datas = []
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[:50]
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_path = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"
        image_file = f"{dataset_root}/images/with_demo_{id_}.jpg"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = f"""#Demonstration Example1
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

Solution: She bought 5 bagels for \$3 each. This means she spent 5 * \$3 = \$15 on the bagels. 
She had \$23 in beginning, so now she has \$23 - \$15 = \$8. The answer is 8.

#Demonstration Example2
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

Solution: There are 4 days from monday to thursday. 5 computers were added each day. 
That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning.
So now there are 9 + 20 = 29 computers. The answer is 29.

Learn from above examples and solve the following test example:
# Test Example
{f.read()}
"""
            # text = f.read()
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        original_img_file = image_path
        demo_file = "dataset/Sampled_Data_For_Experiments/Math/math_wp_demo.jpg"
        if not os.path.exists(image_file):
            images_concat(demo_file, original_img_file, image_file)
        # image_file = remote_imgs[ids.index(id_)]
        # match = re.search("(?<=d/).*?(?=/view)", remote_imgs[ids.index(id_)])
        # file_id = match.group() if match else ""
        # image_file = f"https://drive.google.com/u/0/uc?id={file_id}&export=download"
        datas.append({"id": id_, "text": text, "image_file": image_file, "original_img_file": original_img_file,
                      "label": answer})
    return datas


def load_math_wp_zs():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[:50]
    datas = []
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_path = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = f.read()
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        image_file = image_path
        datas.append({"id": id_, "text": text, "image_file": image_file, "label": answer})
    return datas


def load_counting100_demo():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: How many different teams in this shot?
Answer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot.

# Demonstration Example 2
Question: How many levels do the buses have?
Answer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_v4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        # if not os.path.exists(image_file):
        #     images_concat(demo_file, image_only_file, image_file)
        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_counting100_demo1():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo1_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo1.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        # if not os.path.exists(image_file):
        #     images_concat(demo_file, image_only_file, image_file)
        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_counting100_demo3():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo3_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo3.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        # if not os.path.exists(image_file):
        #     images_concat(demo_file, image_only_file, image_file)
        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_counting100_demo4():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo4_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        # if not os.path.exists(image_file):
        #     images_concat(demo_file, image_only_file, image_file)
        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: Is the dog hot?
Answer: Let's think step by step. In the image, the dog appears to be panting, which is a common sign that it may be hot or has been active. Panting is a normal response in dogs to help regulate their body temperature since they cannot sweat through their skin like humans do. It's also worth noting that dogs will pant when they are excited or after exercise. Therefore, the answer is yes.

# Demonstration Example 2
Question: Is there a chain on the hydrant?
Answer: Let's think  step by step. In the provided image, there is no visible chain on the hydrant. A chain is sometimes attached to fire hydrants to secure the caps or to link to a hydrant wrench, but in this picture, such a chain is not present. The hydrant appears to have a typical design with a red top and white base, and it is standing alone without any attachments. Therefore, the answer is no.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_ysorno_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo1():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo1_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo1.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo3():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo3_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo3.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo4():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo4_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                         '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_l2r():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_l2r_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                            '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo_l2r.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file)
        image_paste_anywhere(demo_file, image_only_file, image_file, x=1400, y=60)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_t2brot():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_t2brot_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                               '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo_t2brot.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file)
        image_rotating_concat(demo_file, image_only_file, image_file, pos='c')
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_sampled100_demo():
    ori_dataset_root = "dataset/vqa2_sampled100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/vqa2_sampled100_imageonly"
    create_dir(dataset_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: What language is the writing on the truck in?
Answer: Let's think step by step. The writing on the truck is in Spanish. The text "BARBACOA & FRUTAS EL VAQUERO" translates to "Barbecue & Fruits The Cowboy" in English. Therefore, the answer is Spanish.

# Demonstration Example 2
Question: Where is this man?
Answer: Let's think step by step. The man in the image appears to be at a beach. You can tell by the sandy ground he's sitting on, the presence of beach chairs and umbrellas, and other people in the background who seem to be enjoying a day at the beach. The weather looks sunny and clear, which is typical for a beach setting. Therefore, the answer is beach.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_random_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        # answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        # with open(answer_file, mode="r", encoding="utf-8") as f:
        #     answer = f.read()
        #     f.close()
        answer = ""
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_hallusionbench_few_shot():
    dataset_root = "dataset/hallusion_bench"
    image_root = "dataset/HallusionBench_few_shot_higher"
    data_file = f"{dataset_root}/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    for raw_idx in range(len(df_datas)):
        data = df_datas[raw_idx]
        if data.get("visual_input"):
            image_file = f"{image_root}/{data.get('subcategory')}_{data.get('set_id')}_{data.get('figure_id')}_{data.get('question_id')}.png"
        else:
            image_file = None
        text = data.get('question')
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}-{data.get('figure_id')}-{data.get('question_id')}"
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append(
                {"id": id_, "text": text, "image_file": image_file, "label": answer, "label_detail": answer_detail}
            )

    return datas


def load_mathvista(args):
    dataset_root = "dataset/mathvista_testmini/testmini"
    # image_root = "dataset/HallusionBench_few_shot_higher"
    data_file = f"{dataset_root}/testmini.json"
    # df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        # data = df_datas[raw_idx]
        if data[idx]['category'] == args.category and data[idx]['context'] == args.context:
            image_file = f"{dataset_root}/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
            # create_dir('dataset/mathvista_testmini/testmini_text_concat')
            # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}")
            # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}")
            # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images")
            # # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
            # text_concat_image_file = f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
            # text_concat_image_file_resize = f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}_resize.png"
        else:
            image_file = None
        text = data[idx]['question']
        choices = None
        if 'Choices' in data[idx]['query']:
            choices = 'Choices ' + data[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = data[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append({
                "pid": id_,
                "image_path": data[idx]['image_path'],
                "image_file": image_file,
                "category": data[idx]["category"],
                "context": data[idx]["context"],
                "grade": data[idx]["grade"],
                "img_height": data[idx]["img_height"],
                "img_width": data[idx]["img_width"],
                "language": data[idx]["language"],
                "skills": data[idx]["skills"],
                "source": data[idx]["source"],
                "split": data[idx]["split"],
                "task": data[idx]["task"],
                "question": data[idx]["question"],
                "query": data[idx]["query"],
                "choices": choices,
                "answer": data[idx]["answer"]
            })
    return datas


def load_mathvista_sample100(args):
    dataset_root = "dataset/mathvista_testmini/sample_100/mathvista_100"
    # image_root = "dataset/HallusionBench_few_shot_higher"
    data_file = f"{dataset_root}/sample_data.json"

    # df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        # data = df_datas[raw_idx]
        image_file = data[idx]['image_file'].replace("mathvista_100", "dataset/mathvista_testmini/testmini")
        create_dir('dataset/mathvista_testmini/testmini_text_concat')
        create_dir(f"dataset/mathvista_testmini/testmini_text_concat/choose_better")
        text_concat_image_file = f"dataset/mathvista_testmini/testmini_text_concat/choose_better/{data[idx]['pid']}.png"
        text_concat_image_file_resize = f"dataset/mathvista_testmini/testmini_text_concat/choose_better/{data[idx]['pid']}_resize.png"

        datas.append({
            "pid": data[idx]['pid'],
            "image_file": image_file,
            "text_concat_image_file": text_concat_image_file,
            "text_concat_image_file_resize": text_concat_image_file_resize
        })
    return datas


def load_mathvista_sample100_caption(args):
    # dataset_root = "dataset/mathvista_testmini/sample_100/mathvista_100"
    # image_root = "dataset/HallusionBench_few_shot_higher"
    # data_file = f"{dataset_root}/sample_data_caption_output_new.json"
    data_file = 'result/mathvista_choose_better_result/testmini_better_method_labeled_caption.json'
    # df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    data_ = load_json(data_file)
    datas = []
    for idx, data in data_.items():
        # data = df_datas[raw_idx]
        # image_file = data.get('image_file').replace("mathvista_100","dataset/mathvista_testmini/testmini")
        image_file = data.get('image_file').replace("mathvista_100", "dataset/mathvista_testmini/testmini_text_concat")
        # create_dir('dataset/mathvista_testmini/testmini_text_concat')
        # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/choose_better")
        # text_concat_image_file = f"dataset/mathvista_testmini/testmini_text_concat/choose_better/{data[idx]['pid']}.png"
        # text_concat_image_file_resize = f"dataset/mathvista_testmini/testmini_text_concat/choose_better/{data[idx]['pid']}_resize.png"

        datas.append({
            "pid": data.get('pid'),
            "image_file": image_file,
            "image_caption": data.get('image_caption')
        })
    return datas


def load_mathvista_ipy(category, context):
    dataset_root = "dataset/mathvista_testmini/testmini"
    # image_root = "dataset/HallusionBench_few_shot_higher"
    data_file = f"{dataset_root}/testmini.json"
    # df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        # data = df_datas[raw_idx]
        if data[idx]['category'] == category and data[idx]['context'] == context:
            image_file = f"{dataset_root}/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
            create_dir('dataset/mathvista_testmini/testmini_text_concat')
            create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}")
            create_dir(
                f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}")
            create_dir(
                f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images")
            # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
            text_concat_image_file = f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
        else:
            image_file = None
        text = data[idx]['question']
        choices = None
        if 'Choices' in data[idx]['query']:
            choices = 'Choices ' + data[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = data[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append({
                "pid": id_,
                "image_path": data[idx]['image_path'],
                "image_file": image_file,
                "category": data[idx]["category"],
                "context": data[idx]["context"],
                "grade": data[idx]["grade"],
                "img_height": data[idx]["img_height"],
                "img_width": data[idx]["img_width"],
                "language": data[idx]["language"],
                "skills": data[idx]["skills"],
                "source": data[idx]["source"],
                "split": data[idx]["split"],
                "task": data[idx]["task"],
                "question": data[idx]["question"],
                "query": data[idx]["query"],
                "choices": choices,
                "answer": data[idx]["answer"],
                "text_concat_image_file": text_concat_image_file
            })
    return datas


def load_mathvista_few_shot_mix(args):
    general_vqa_abstract_scene_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: Are there at least 3 ducks in this scene?
# Choices:
# (A) Yes
# (B) No
# Answer: Let's think step by step. In this picture, we can see 4 ducks, which is more than 3. Therefore, the answer is yes.

# Test Example
"""
    general_vqa_bar_chart_demo_text = """Hint: Please answer the question requiring an integeranswer and provide the final value, e.g., 1, 2, 3, at theend.
# Demonstration Example
# Question: How many bars have values larger than 8?
# Answer: Let's think step by step. In picture, there is no bar has value larger than 8. The answer is 0.

# Test Example
"""
    general_vqa_document_image_demo_text = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
# Demonstration Example
# Question: Before  week 4, in which week is the cumulative increase in weight the highest, for Group C?
# Answer: Let's think step by step. According to the picture, Step 1: It is observed that the line representing the Group C. Step 2: On this line, observe that the point corresponding to week 3 is the highest before week 4. Therefore,for Group C, week 2 is the cumulative increase in weight the highest for Group C. 

# Test Example
"""
    general_vqa_medical_image_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: Is the cardiac silhouette less than half the lateral diameter of the chest wall?
# Choices:
# (A) Yes
# (B) No
# Answer: Let's think step by step. According to the picture, the cardiac silhouette is less than half the lateral diameter of the chest wall 

# Test Example
    """
    general_vqa_line_plot_demo_text = """Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
# Demonstration Example
# Question: The sum of the lowest points in the two lines equal to what?
# Answer: Let's think step by step. To calculate the sum of the lowest points in the two lines, we must identify those lowest points on the chart for both 'Men' and 'Women'. For "Women' the lowest point is at the beginning of the line in 2000, with a value of 587.16. For 'Men' the lowest point appears to be at the same year in 2000, with a value of 756.59. Now we calculate the sum of these two values587.16(Women)+ 756.59 (Men) = 1,343.75. Therefore, the sum of the lowest points in the two lines is 1,343.75

# Test Example
"""
    general_vqa_map_chart_demo_text = """Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
# Demonstration Example
# Question: Question: Does Virginia have the highest value in the USA ?
# Choices:
# (A) Yes
# (B) No
# Answer: Let's think step by step. Based on the map provided, Virginia is the area of "VA!which is not shown as having the highest value in the USA: in fact, it is colored in a darkershade, indicating a smaller percentage point difference than states that are colored in lightershades, The darker the color, the smaller the number, Therefore, the correct answer is:(B)No

# Test Example
"""
    general_vqa_natural_image_demo_text = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
# Demonstration Example
# Question: How many people in the image were born after the end of World War II?
# Answer: Let's think step by step. World War II ended in 1945. On the left is  Elizabeth Ann Warren, who is born in 1946.6; On the middle is Obama, who is born in 1961.8; On the right is Richard Cordray, who is born in 1959.3. Therefore, the answer is 3.

# Test Example
"""
    general_vqa_pie_chart_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: Is Medium Purple greater than Coral?
# Choices:
# (A) yes
# (B) no
# Answer: Let's think step by step. According to the pie chart, the area of Medium Purple is greater than Coral. The answer is Yes.

# Test Example
"""
    general_vqa_scatter_plot_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: Is Midnight Blue the smoothest?
# Choices:
# (A) yes
# (B) no
# Answer: Let's think step by step. According to the picture, after fitting the curve Midnight Blue, it is not the smoothest. Therefore, the answer is B) No.

# Test Example
"""
    general_vqa_scientific_figure_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g..A, B, C,D,at the end.
# Demonstration Example
# Question: What will happen to the guppies population is the duck population increases?
# Choices:
# (A) It will remain the same
# (B) It will decrease
# (C) It will decrease slightly
# (D) It will increase
# Answer: Let's think step by step. When there is a sudden increase in duckthere could be a sudden increase in the guppies population due to theincreased availability of food. Thereforethe answer is D(lt will increase).

# Test Example
"""
    general_vqa_synthetic_scene_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g, A, B, C, Dat the end.
# Demonstration Example
# Question: ls the number of purple sedans that are on the left side of the big purple thing less than the number of yellow matte biplanes?
# Choices:
# (A) Yes
# (B)No
# Answer: Let's think step by stepon the left side of the big purple thing, there are two purple sedans. In picture, there is one yellow matte blplanes.Then, the answer is B.no

# Test Example
"""
    math_targeted_vqa_abstract_scene_demo_text = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. 
# Demonstration Example
# Question: Move the ruler to measure the length of the line to the nearest centimeter.  The line is about (_) centimeters long.
# Answer: Let's think step by step. The line is about 5 centimeters long.

# Test Example
    """
    math_targeted_vqa_bar_chart_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: For the models shown in the figure, which one has the lowest rate of being 'Refused to Answer'?
# Choices:
# (A) davinci
# (B) OPT-1.3B
# (C) text-davinci-003
# (D) flan-t5-xxl
# (E) ChatGPT
# (F) GPT-4
# Answer: Let's think step by step. According to the picture, the bar of OPT-1.3B is lowest of all. Therefore, the answer is B)OPT-1.3B.

# Test Example
"""
    math_targeted_vqa_function_plot_demo_text = """Hint: Please answer the question and provide the correctoption letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question:ls this a monotonic function?
Choices:
# (A) Yes
# (B)No
# Answer: Let's think step by step.In the domain, the function is monotonically increasing when y is negative for x less than zero, monotonically decreasing for x in the range (0, 4.5), and monotonically increasing for x greater than 4.5. Therefore, the function is not monotonic within its domain. Therefore, theanswer is B(No.

# Test Example
"""
    math_targeted_vqa_geometry_diagram_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
# Demonstration Example
# Question: 如图，在△ABC中，∠C＝90°，AD是∠BAC的平分线，若∠ADC＝65°，则∠BAC的大小为（）
# Choices:
# (A) 25°
# (B) 35°
# (C) 50°
# (D) 70°
# Answer: Let's think step by step. In triangle DAC, ∠DAC=180-∠C(90)-∠ADC(65)=25. AD is the angle bisector of ∠BAC, Then ∠DAC=∠BAD=25,∠BAC=25+25=50. Therefore the answer is C.50

# Test Example
    """
    math_targeted_vqa_line_plot_demo_text = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. 
# Demonstration Example
# Question: Which model exhibits the lowest performance relative to fine-tuned state-of-the-art models when provided with 4 in-context examples?
# Choices:
# (A) Flamingo-3B
# (B) Flamingo-9B
# (C) OF-3B
# (D) OF-3B (I)
# (E) OF-4B
# (F) OF-4B (I)
# (G) OF-9B
# Answer: Let's think step by step. To find out which model exhibits the lowest performance relative to fine-tuned state-of-the-art models when provided with 4 in-context examples
# With 4 in-context examples (which corresponds to 4 on the x-axis), you should look for the line that is lowest at that point. From the graph, it appears that the line representing the "OF-3B" model is the lowest. Therefore, the correct answer is: (C) OF-3B

# Test Example
"""
    math_targeted_vqa_puzzle_test_demo_text = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
# Demonstration Example
# Question: Which numbers is missing?
# Answer: Let's think step by step. Observing the patterns in the rows and columns of the image, in each column, the circle's number is equal to the product of the remaining numbers. In first column: 24=2*3*4, In last column: 32=2*2*?, then, the number is 8. Therefore, the answer is 8.

# Test Example
    """
    math_targeted_vqa_scientific_figure_demo_text = """Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 13, 1.4, at the end.
# Demonstration Example
# Question: Tangent Circle at C.AB: common tangent. ∠0QB=112. What is ∠BAC? Return the numeric value.
# Answer: Let's think step by step. The image presents a scientific figureAccording to the tangent theorem of the circle, ∠OAB=90，CQB=90 ∠AOC=360-90-90-112=68. The radii of the circles are equal,OA=OC.∠AOC=∠ACO=(180-68)/2=56: BAC=90-∠OAC=90-56=34. Therefore the answer is 34.

# Test Example
    """
    math_targeted_vqa_synthetic_scene_demo_text = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1,2, 3, at the end.
# Demonstration Example
# Question: Subtract all tiny gray spheres. Subtract all yellow cubes. How many objects are left?
# Answer: Let's think step by step. In picture, there are5 objects. 1 purple sphere,1 yellow light cube，1green light cube, 1 big gray cube, 1 bright redcylinder. There is 1 yellow cube in picture, no tinygray sphere. Then, subtract all tiny gray spheres.Subtract all yellow cubes,5-1=4.The answer is 4.

# Test Example
    """
    math_targeted_vqa_table_demo_text = """Hint: Please answer the question requiringan integer answer and provide the finalvalue, e.g., 1, 2, 3, at the end.
# Demonstration Example
# Question:A baker wrote down how many pies she made in the past 5 days. What is the median of the numbers?
# Answer: Let's think step by step. The table indicates the number of pies baked each day fromMonday to Friday. To find the median, we mustfirst list the numbers in numeric order:0,0,1,2,3. The median is the middle number in the sortedlist. Since we have five numbers, the middle onewill be the third number:
0,0,1,2,3. The median number of pies baked in the past 5days is 1. The answer is 1.

# Test Example
    """
    math_targeted_vqa_scatter_plot_demo_text = """Hint: Please answer the question requiringan integer answer and provide the finalvalue, e.g., 1, 2, 3, at the end.
# Demonstration Example
# Question: Is Midnight Blue the smoothest?\nChoices: (A) yes (B)no
# Answer: Let's think step by step. According to the picture, after fitting the curveMidnight Blue, it is not the smoothest. Therefore, the answer is B) No.

# Test Example
"""
    dataset_root = "dataset/mathvista_testmini/testmini"
    data_file = "dataset/mathvista_testmini/testmini/testmini.json"
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        if data[idx]['category'] == args.category and data[idx]['context'] == args.context:
            image_file = f"{dataset_root}/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
            demo_file = f"dataset/mathvista_testmini/mix_picture/{args.category}_{args.context}.png"
            temp_category = args.category.replace('-', '_')
            demo_text = eval(f"{temp_category}_{args.context}_demo_text")
        else:
            image_file = None
        text = data[idx]['question']
        choices = None
        if 'Choices' in data[idx]['query']:
            choices = 'Choices ' + data[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = data[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append({
                "pid": id_,
                "image_path": data[idx]['image_path'],
                "image_file": image_file,
                "demo_file": demo_file,
                "demo_text": demo_text,
                "category": data[idx]["category"],
                "context": data[idx]["context"],
                "grade": data[idx]["grade"],
                "img_height": data[idx]["img_height"],
                "img_width": data[idx]["img_width"],
                "language": data[idx]["language"],
                "skills": data[idx]["skills"],
                "source": data[idx]["source"],
                "split": data[idx]["split"],
                "task": data[idx]["task"],
                "question": data[idx]["question"],
                "query": data[idx]["query"],
                "choices": choices,
                "answer": data[idx]["answer"]
            })
    return datas


def load_mathvista_few_shot_text_only(args):
    general_vqa_abstract_scene_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Answer the question according to the description.
Description: The image depicts a stylized outdoor scene with two people and a few animals. On the left, there's a leafless tree, and two individuals appear to be enjoying themselves around it. One person is hanging off a tree branch with one hand, while the other is seemingly seated casually on the ground, leaning against the tree with one hand extended outward. Both individuals have a relaxed and playful demeanor.
In the center-right of the image, there are two birds that look like ducks or perhaps geese walking on the grass. To their right, there's a small pond with clear blue water and one more bird floating on it. In the background, there are a few green bushes or shrubs.
The sky is a clear light blue, suggesting it is a fair weather day. The grass is uniformly green, which gives a calm and peaceful atmosphere to the setting. The overall impression is that of a simple, cartoon-style representation of a day at the park or a natural outdoor environment.
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C Dat the end.
Question:  Are there at least 3 ducks in this scene?
Choices: (A) Yes (B) No
Answer:  Let's think step by step. In this picture, we can see 4 ducks, which is more than 3.Therefore, the answer is (A).yes

# Test Example
Description: {description}
{question}
Answer: 
    """
    general_vqa_bar_chart_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Answer the question according to the description.
Description: The image shows a simple bar chart with a title "Title" at the top. There are six bars representing different categories labeled on the x-axis: "colt," "sweet," "frieze," "egg," and "peril." Each bar's height corresponds to a value on the y-axis, which is labeled "Values" and has a range from 0 to 10. Here's the approximate value for each category based on the height of each bar:
- "colt": 1; - "sweet": 7; - "frieze": 1; - "egg": 5; - "peril": around 8
The background of the chart is light gray, and the bars are colored in a shade of blue. Horizontal lines across the background help to visually estimate the values of the bars.
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C Dat the end.
Question:  Question: How many bars have values larger than 8?
Answer: Let's think step by step. In picture, there is no bar has value larger than 8. The answer is 0.

# Test Example
Description: {}
Query: {}
Answer: 
        """
    general_vqa_document_image_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This is an image of a line graph, labeled as "Fig. 5," which depicts the average increases in weight over a period of six weeks for three different groups. The vertical axis on the left is labeled "CUMULATIVE INCREASE IN WEIGHT IN GRAMS," with values ranging from 0 to 500 grams. The horizontal axis on the bottom is labeled "WEEKS," with time points from 0 to 6 weeks. There are three lines on the graph representing three different groups:
- Group A (42 % POWDER) is represented by a solid line with circular markers. This line shows a steady increase in weight over the six weeks, ending just below 500 grams. - Group B (21% POWDER) is represented by a dashed line with triangular markers. This line shows fluctuations in weight gain, with a sharp increase between weeks 1 and 2, a decrease between weeks 2 and 3, and then another increase up to week 4, followed by a sharp decrease and a final increase, ending around 300 grams. - Group C (CONTROL) is represented by a dotted line with no markers. This line shows a very modest increase in weight, remaining below 100 grams throughout the six weeks.
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Before  week 4, in which week is the cumulative increase in weight the highest, for Group C?
Answer: Let's think step by step. According to the picture, Step 1: It is observed that the line representing the Group C. Step 2: On this line, observe that the point corresponding to week 3 is the highest before week 4. Therefore,for Group C, week 2 is the cumulative increase in weight the highest for Group C.

# Test Example
Description: {}
Query: {}
Answer: 
    """
    general_vqa_map_chart_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image is a color-coded map of the United States showing the percentage point difference by race/ethnicity in the receipt of a COVID-19 vaccine.  The map uses different shades of blue to represent the range of percentage point differences, with darker shades indicating a smaller difference and lighter shades indicating a larger difference. The legend at the bottom categorizes the differences into three ranges: - Dark blue represents states where the difference is between -8 and -1 percentage points. - Medium blue represents states where the difference is between -13 and -9 percentage points. - Light blue represents states where the difference is between -21 and -19 percentage points. Gray states are marked as "N/A," indicating that data is not available for those states.
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: Does Missouri have the highest value in the MidWest ?
Choices:
(A) Yes
(B) No
Answer: Let's think step by step. According to the picture, the 'MO' reprents 'Missouri', the color of Missouri is darkest. The color reprents 25.4%-28.5%. Therefore, Missouri have the highest value in the MidWest the answer is A) Yes.

# Test Example
Description: {}
Query: {}
Answer: 
    """
    general_vqa_medical_image_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This is a chest X-ray image, also known as a chest radiograph. The image is in grayscale, typical of X-ray images, with lighter areas representing denser tissues such as bones, and darker areas indicating less dense tissues like air-filled spaces. In the image, you can see the outline of the rib cage with the ribs appearing as curved, horizontal structures. The central white vertical structure is the spine. The darker areas on either side of the spine are the lung fields, which should normally appear dark due to being filled with air. The heart is usually visible as a denser area in the center of the chest, but specific details about the heart's shape and size cannot be determined without medical expertise.
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: Is the cardiac silhouette less than half the lateral diameter of the chest wall?
Choices:
(A) Yes
(B) No
Answer: Let's think step by step. According to the picture, the cardiac silhouette is less than half the lateral diameter of the chest wall 

# Test Example
Description: {}
Query: {}
Answer: 
    """
    general_vqa_line_plot_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image provided is a line graph showing the number of students in thousands from the year 2000 to 2019, with separate lines representing men and women. The graph indicates a general increase in the number of students for both genders over the years.
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: The sum of the lowest points in the two lines equal to what?
Answer: Let's think step by step. To calculate the sum of the lowest points in the two lines, we must identify the lowest points on the chart for both "Men" and "Women". The lowest point for ''Men" is at 2000. The value is 587.16. The lowest point for ''Women" is also at 2000. The value is 756.59. the sum of the lowest points in the two lines is 587.16+ 756.59 = 1,343.75. Therefore the sum of the lowest points in the two lines is 1,343.75.

# Test Example
Description: {}
Query: {}
   """
    general_vqa_natural_image_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: In this image, we see three individuals standing in front of the White House.  The person in the center is speaking at a podium with the Presidential Seal, indicating that this is a formal event or announcement.  On the left is  Elizabeth Ann Warren;  On the middle is Obama;  On the right is Richard Cordray.
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: How many people in the image were born after the end of World War II?
Answer: Let's think step by step. World War II ended in 1945. On the left is  Elizabeth Ann Warren, who is born in 1946.6; On the middle is Obama, who is born in 1961.8; On the right is Richard Cordray, who is born in 1959.3. Therefore, the answer is 3.

# Test Example
Description: {}
Query: {}
"""
    general_vqa_pie_chart_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a colorful pie chart with a simple legend to the right. The chart is divided into six segments, each representing a different color as indicated by the legend. The colors and their corresponding labels are:1. Coral; 2. Medium Purple; 3. Dark Orchid; 4. Royal Blue; 5. Blue; 6. Dark Turquoise; 7. Dark Orange
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: Is Medium Purple greater than Coral?
Choices:
(A) yes
(B) no
Answer: Let's think step by step. According to the pie chart, the area of Medium Purple is greater than Coral. The answer is Yes.

# Test Example
Description: {}
Query: {}
"""
    general_vqa_scatter_plot_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This is a scatter plot with a variety of colored points plotted across a grid.  The x-axis is labeled "xaxis label" and ranges from 0 to just over 100.  The y-axis is labeled "yaxis label" and ranges from 0 to just over 80. The points are colored differently and correspond to a legend on the right side of the graph, which associates each color with a label:- Light Slate; - Medium Seafoam; - Cornflower; - Dark Gold; - Dark Salmon; - Midnight Blue; - Gold
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: Is Midnight Blue the smoothest?
Choices:
(A) yes
(B) no
Answer: Let's think step by step. According to the picture, after fitting the curve Midnight Blue, it is not the smoothest. Therefore, the answer is B) No.

# Test Example
Description: {}
Query: {}
    """
    general_vqa_scientific_figure_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image depicts a simplified food web, which is a graphical representation of the feeding relationships among various organisms in an ecosystem.    The arrows indicate the direction of energy flow, from the food source to the consumer.    Here's a breakdown of the depicted relationships: - Plants are shown as the base of the food web, providing energy for several types of consumers. - Algae are consumed by Bottom feeders, Zooplankton, snails, Guppies and tadpoles. - Guppies are depicted as a food source for ducks. - Ants are shown consuming plants - Zooplankton are consuming detritus and are a food source for guppies. - Ducks are shown as consumers of plants, guppies
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What will happen to the guppies population is the duck population increases?
Choices:
(A) It will remain the same
(B) It will decrease
(C) It will decrease slightly
(D) It will increase
Answer: Let's think step by step. When there is a sudden increase in duck, there could be a sudden decrease in the guppies population due to the Increase in the number of natural enemies. Therefore.the answer is B(lt will decrease)

# Test Example
Description: {}
Query: {}
        """
    general_vqa_synthetic_scene_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image features a collection of toy vehicles arranged on a neutral background.    There is a purple car in the foreground with yellow wheels.    Behind it, there are two purple sedans.    To the left of the train, there is a yellow toy fighter jet, and right the train, there is a gray propeller airplane in flight.    On the right side of the image, there is a green motorcycle.
Hint: Please answer the question and provide the correct option letter, e.g, A, B, C, Dat the end.
Question: ls the number of purple sedans that are on the left side of the big purple thing less than the number of yellow matte biplanes?
Choices:
(A) Yes
(B)No
Answer: Let's think step by step. On the right side of the big purple thing, there are 2 purple sedans. In picture, there is 1 yellow matte blplanes. Then, the answer is B.No

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_abstract_scene_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a graphical representation of a ruler. The ruler is depicted with a wood grain texture and has measurements in centimeters. It is marked from 0 to 10 centimeters, with each centimeter clearly indicated by a vertical line. The numbers 1 through 10 are written below the corresponding lines to indicate the length in centimeters. There is a horizontal line above the ruler.
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. 
Question: Move the ruler to measure the length of the line to the nearest centimeter.  The line is about (_) centimeters long.
Answer: Let's think step by step. Extend the starting point downward as a dashed line, the start is about 1.5. Extend the end point downward as a dashed line, the end is about 6.5, then the line is 6.5-1.5 = 5 centimeters long. The line is about (5) centimeters long.

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_bar_chart_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image is a bar chart titled "Figure 27: Result of evaluating LLM's hallucination."  The vertical axis is labeled "Refused to Answer (%)" and ranges from 0 to 100%.  The horizontal axis lists different names: "davinci," "OPT-1.3B," "text-davinci-003," "flan-t5-xxl," "ChatGPT," and "GPT-4.". Each model has a corresponding bar that indicates the percentage of times it refused to answer, presumably in a test of its ability to avoid generating incorrect or nonsensical responses (hallucinations).  The bars are filled with diagonal stripes, and the percentages increase from left to right, with "OPT-1.3B" having the lowest refusal rate and "GPT-4" having the highest.
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: For the models shown in the figure, which one has the lowest rate of being 'Refused to Answer'?
Choices:
(A) davinci
(B) OPT-1.3B
(C) text-davinci-003
(D) flan-t5-xxl
(E) ChatGPT
(F) GPT-4
Answer: Let's think step by step. According to the picture, the bar of OPT-1.3B is lowest of all. Therefore, the answer is B)OPT-1.3B.

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_function_plot_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a Cartesian coordinate system with an x-axis (horizontal) and a y-axis (vertical).  Plotted on this system is a blue curve that appears to represent a function or a data set.  The curve starts from the lower left, rises to a peak, then descends to a trough, rises again to a lower peak, and finally continues to the right, slightly descending. There are two notable points marked on the curve with a darker color, which could indicate specific data points, maxima, minima, or points of interest.  The grid in the background helps to determine the position of the curve at various points along the x and y axes.  The arrows at the end of the axes suggest that the axes extend infinitely in both the positive and negative directions.  The curve itself also has arrows at both ends, indicating that it continues beyond the section shown in the graph.
Hint: Please answer the question and provide the correctoption letter, e.g., A, B, C, D, at the end.
Question:ls this a monotonic function?
Choices:
(A) Yes
(B)No
Answer: Let's think step by step. In the domain, the function is monotonically increasing when y is negative for x less than zero, monotonically decreasing for x in the range (0, 4.5), and monotonically increasing for x greater than 4.5. Therefore, the function is not monotonic within its domain. Therefore, the answer is B(No.

# Test Example
Description: {}
Query: {}
    """
    math_targeted_vqa_geometry_diagram_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a geometric figure consisting of a triangle labeled with the vertices A, B, and C. Point A is at the top, and points B and C form the base.  There is also a point labeled D along the line segment BC.  Two line segments, AD and BD, are drawn from points A and B to point D, respectively, indicating that D is a point on the side BC of the triangle. 
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: 如图，在△ABC中，∠C＝90°，AD是∠BAC的平分线，若∠ADC＝65°，则∠BAC的大小为（）
Choices:
(A) 25°
(B) 35°
(C) 50°
(D) 70°
Answer: Let's think step by step. In triangle DAC, ∠DAC=180-∠C(90)-∠ADC(65)=25. AD is the angle bisectorof ∠BAC, Then ∠DAC=∠BAD=25,∠BAC=25+25=50. Therefore the answer is C.50

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_line_plot_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image is a line graph that compares the performance of different models relative to fine-tuned state-of-the-art (SOTA) performance, averaged across datasets.   The y-axis represents the "Aggregated % of fine-tuned SOTA," which ranges from 50% to 70%.   The x-axis represents the "Number of in-context examples," with values ranging from 0 to 32.  There are several lines on the graph, each representing a different model, and they are color-coded for distinction.   The models include Flamingo-3B, Flamingo-9B, OF-3B, OF-3B (I), OF-4B, OF-4B (I), and OF-9B.   The lines for Flamingo-3B and Flamingo-9B are dashed, while the others are solid.  The graph shows that as the number of in-context examples increases, the performance of some models improves, while for others it remains relatively stable or decreases.   The Flamingo models start with higher performance at zero in-context examples and maintain a lead, with Flamingo-9B having the highest performance overall.   The OF models show varying degrees of improvement with more in-context examples, but none surpass the Flamingo models' performance. 
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. 
Question: Which model exhibits the lowest performance relative to fine-tuned state-of-the-art models when provided with 4 in-context examples?
Choices:
(A) Flamingo-3B
(B) Flamingo-9B
(C) OF-3B
(D) OF-3B (I)
(E) OF-4B
(F) OF-4B (I)
(G) OF-9B
Answer: Let's think step by step. To find out which model exhibits the lowest performance relative to fine-tuned state-of-the-art models when provided with 4 in-context examples. With 4 in-context examples (which corresponds to 4 on the x-axis), you should look for the line that is lowest at that point. From the graph, it appears that the line representing the "OF-3B" model is the lowest. Therefore, the correct answer is: (C) OF-3B

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_puzzle_test_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a collection of shapes, each containing a number. The shapes are arranged in three rows and two columns. In the first row, there's a circle with the number 24 and a square with the number 2. In the second row, there's a triangle with the number 2 and an upside-down triangle with the number 2. In the third row, there's a pentagon with the number 3 and a hexagon with the number 32. At the bottom, there's a shape with five sides, resembling a house or pentagon with a bottom flap, and it contains a question mark, indicating that the viewer is likely meant to deduce the number that should be inside this shape based on a pattern or rule governing the other shapes and numbers.
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which numbers is missing?
Answer: Let's think step by step. Observing the patterns in the rows and columns of the image, in each column, the circle's number is equal to the product of the remaining numbers. In first column: 24=2*3*4, In last column: 32=2*2*?, then, the number is 8. Therefore, the answer is 8.

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_scientific_figure_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: This image depicts a geometric problem involving two circles and a triangle.  The larger circle is shaded in green and labeled with the center point O, while the smaller circle is shaded in orange and labeled with the center point Q. There is a triangle formed by points A, B, and C, where point C lies on both circles, point A lies on the larger circle, and point B lies on the smaller circle. The line segments connecting the points are as follows: - OA and OB are radii of the larger and smaller circles, respectively. - AC and BC are chords of the larger and smaller circles, respectively. - AB is a common external tangent to both circles. The angle at point C, which is part of the triangle and lies on the smaller circle, is given as 112 degrees.  
Hint: Please answer the question requiring a floating-point number with onedecimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: Tangent Circle at C.AB: common tangent. OQB=112. What isZBAC? Return the numeric value.
Answer: Let's think step by step.The image presents a scientific figureAccording to the tangent theorem of the circle,∠OAB=90, /CQB=90ZAOC=360-90-90-112=68.The radii of the circles are equal,OA=OC.ZAOC=∠ACO=(180-68)/2=56;∠BAC=90-ZOAC=90-56=34.Therefore, theanswer is 34.

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_synthetic_scene_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a collection of five 3D-rendered geometric shapes on a flat surface with a neutral background.  From left to right, there is a reflective silver cube, a matte purple sphere, a small shiny gold cube, a shiny green cube, and a reflective red cylinder.  
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1,2, 3, at the end.
Question: Subtract all tiny gray spheres. Subtract all yellow cubes. How many objects are left?
Answer: Let's think step by step. In picture, there are 5 objects. 1 big gray cube, 1 purple sphere, 1 yellow light cube, 1 green light cube, 1 bright red cylinder. There is no tiny gray spheres, 1 yellow cube in picture. Then, subtract all tiny gray spheres. Subtract all yellow cubes,5-1=4. The answer is 4.

# Test Example
Description: {}
Query: {}
"""
    math_targeted_vqa_table_text_for_question = """There are a demonstration example and a test example. Following the reasoning format in the demonstration example, reason the answer to the question of the test example. Only output the reasoning of test example. 
# Demonstration Example 1
Description: The image shows a simple table with a pink header and a white background.  The table is titled "Pies baked" and has two columns.  The first column is labeled "Day" and lists the days of the workweek from Monday to Friday.  The second column is labeled "Number of pies" and lists the number of pies baked on each corresponding day.  According to the table, 3 pies were baked on Monday, 2 on Tuesday, and none on Wednesday and Thursday.  On Friday, 1 pie was baked.
Hint: Please answer the question requiringan integer answer and provide the finalvalue, e.g., 1, 2, 3, at the end.
Question:A baker wrote down how many pies she made in the past 5 days. What is the median of the numbers?
Answer: Let's think step by step. The table indicates the number of pies baked each day fromMonday to Friday. To find the median, we mustfirst list the numbers in numeric order:0,0,1,2,3. The median is the middle number in the sortedlist. Since we have five numbers, the middle onewill be the third number: 0,0,1,2,3. The median number of pies baked in the past 5days is 1. The answer is 1.

# Test Example
Description: {}
Query: {}
"""
    dataset_root = "dataset/mathvista_testmini/testmini"
    data_file = "dataset/mathvista_testmini/testmini/testmini.json"
    data = load_json(data_file)
    datas = []
    for idx in range(len(data)):
        if data[idx]['category'] == args.category and data[idx]['context'] == args.context:
            image_file = f"{dataset_root}/{data[idx]['category']}/{data[idx]['context']}/images/{data[idx]['pid']}.png"
            # demo_file = f"dataset/mathvista_testmini/mix_picture/{args.category}_{args.context}.png"
            temp_category = args.category.replace('-', '_')
            text_for_question = eval(f"{temp_category}_{args.context}_text_for_question")
        else:
            image_file = None
        text = data[idx]['question']
        choices = None
        if 'Choices' in data[idx]['query']:
            choices = 'Choices ' + data[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = data[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas.append({
                "pid": id_,
                "image_path": data[idx]['image_path'],
                "image_file": image_file,
                # "demo_file": demo_file,
                "text_for_question": text_for_question,
                "category": data[idx]["category"],
                "context": data[idx]["context"],
                "grade": data[idx]["grade"],
                "img_height": data[idx]["img_height"],
                "img_width": data[idx]["img_width"],
                "language": data[idx]["language"],
                "skills": data[idx]["skills"],
                "source": data[idx]["source"],
                "split": data[idx]["split"],
                "task": data[idx]["task"],
                "question": data[idx]["question"],
                "query": data[idx]["query"],
                "choices": choices,
                "answer": data[idx]["answer"]
            })
    return datas


def load_mathvista_similarity_respectively(args):
    dataset_testmini_root = "dataset/mathvista_testmini/testmini"
    data_similarity_file = "dataset/mathvista_testmini/testmini/test_personal_similarity.json"
    data_test_file = "dataset/mathvista_all/test_directory.json"
    data_testmini_file = "dataset/mathvista_testmini/testmini/testmini.json"
    datas_testmini = load_json(data_testmini_file)
    data_test = load_json(data_test_file)
    data_similarity = load_json(data_similarity_file)
    datas_testmini_ = []
    for idx in range(len(datas_testmini)):
        if datas_testmini[idx]['category'] == args.category and datas_testmini[idx]['context'] == args.context:
            image_file = f"{dataset_testmini_root}/{datas_testmini[idx]['category']}/{datas_testmini[idx]['context']}/images/{datas_testmini[idx]['pid']}.png"
            demo_file_number = data_similarity[datas_testmini[idx]['pid']].get("max_similarity_score")[0]
            demo_query = data_test[demo_file_number].get("query")
            demo_test_file = f"dataset/mathvista_all/{args.context}/images/{demo_file_number}.png"
            create_dir(f"dataset/mathvista_testmini/similarity_picture")
            create_dir(f"dataset/mathvista_testmini/similarity_picture/{args.category}")
            create_dir(f"dataset/mathvista_testmini/similarity_picture/{args.category}/{args.context}")
            demo_image_file = f"dataset/mathvista_testmini/similarity_picture/{args.category}/{args.context}/{demo_file_number}.png"
            temp_category = args.category.replace('-', '_')
            text_concat_image_file = f"dataset/mathvista_testmini/testmini_text_concat/{datas_testmini[idx]['category']}/{datas_testmini[idx]['context']}/images/{datas_testmini[idx]['pid']}.png"
            text_concat_image_file_resize = f"dataset/mathvista_testmini/testmini_text_concat/{datas_testmini[idx]['category']}/{datas_testmini[idx]['context']}/images/{datas_testmini[idx]['pid']}_resize.png"
            # demo_text = eval(f"{temp_category}_{args.context}_demo_text")
        else:
            image_file = None
        text = datas_testmini[idx]['question']
        choices = None
        if 'Choices' in datas_testmini[idx]['query']:
            choices = 'Choices ' + datas_testmini[idx]['query'].split('Choices')[1].replace('\n', ' ')
        id_ = datas_testmini[idx]['pid']
        if image_file is not None and not os.path.exists(image_file):
            warnings.warn(f"找不到文件：{image_file}")
            continue
        if image_file is not None:
            datas_testmini_.append({
                "pid": id_,
                "image_path": datas_testmini[idx]['image_path'],
                "image_file": image_file,
                "demo_test_file": demo_test_file,
                "demo_image_file": demo_image_file,
                "demo_query": demo_query,
                "text_concat_image_file": text_concat_image_file,
                "text_concat_image_file_resize": text_concat_image_file_resize,
                "category": datas_testmini[idx]["category"],
                "context": datas_testmini[idx]["context"],
                "grade": datas_testmini[idx]["grade"],
                "img_height": datas_testmini[idx]["img_height"],
                "img_width": datas_testmini[idx]["img_width"],
                "language": datas_testmini[idx]["language"],
                "skills": datas_testmini[idx]["skills"],
                "source": datas_testmini[idx]["source"],
                "split": datas_testmini[idx]["split"],
                "task": datas_testmini[idx]["task"],
                "question": datas_testmini[idx]["question"],
                "query": datas_testmini[idx]["query"],
                "choices": choices,
                "answer": datas_testmini[idx]["answer"]
            })
    return datas_testmini_


def load_mmmu_validation(args):
    dataset_root = "dataset/MMMU/dataset_validation/MMMU_testset"
    categories = os.listdir(dataset_root)
    for category_i in categories:
        if args.category == category_i:
            cate_root = f"{dataset_root}/{category_i}/images"
            cate_data_file = f"{dataset_root}/{category_i}/testset.json"
            cate_data = load_json(cate_data_file)

            datas = []
            for idx in range(len(cate_data)):
                # data = df_datas[raw_idx]
                id_ = cate_data[idx]['id']
                image_file = []
                if re.sub(r"_\d+", "", cate_data[idx]['id'].split('validation_')[1]) == args.category:
                    image_file_count = sum([1 if cate_data[idx][f'image_{i}'] else 0 for i in range(1, 8)])
                    for i in range(1, image_file_count + 1):
                        temp_image_file = cate_data[idx][f'image_{i}']
                        image_file.append(f"{cate_root}/{temp_image_file}")
                    create_dir(f'{dataset_root}/mmmu_text_concat')
                    create_dir(f"{dataset_root}/mmmu_text_concat/{category_i}")
                    create_dir(f"{dataset_root}/mmmu_text_concat/{category_i}/images")
                    # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
                    text_concat_image_file = f"{dataset_root}/mmmu_text_concat/{category_i}/images/{cate_data[idx]['id']}.png"
                    text_concat_image_file_resize = f"{dataset_root}/mmmu_text_concat/{category_i}/images/{cate_data[idx]['id']}_resize.png"
                else:
                    image_file = None
                question = cate_data[idx]['question']
                options = ast.literal_eval(cate_data[idx]['options'])
                options = str([f"{chr(65 + i)}. '{item}'" for i, item in enumerate(options)])
                options = options.replace("'", "").replace('[', '').replace(']', '')

                # if image_file is not None and not os.path.exists(image_file):
                #     warnings.warn(f"找不到文件：{image_file}")
                #     continue
                if image_file is not None:
                    datas.append({
                        "id": id_,
                        "question": question,
                        "ori_image": image_file,
                        "options": options,
                        "explanation": cate_data[idx]['explanation'],
                        "image_1": cate_data[idx]['image_1'],
                        "image_2": cate_data[idx]['image_2'],
                        "image_3": cate_data[idx]['image_3'],
                        "image_4": cate_data[idx]['image_4'],
                        "image_5": cate_data[idx]['image_5'],
                        "image_6": cate_data[idx]['image_6'],
                        "image_7": cate_data[idx]['image_7'],
                        "img_type": cate_data[idx]['img_type'],
                        "answer": cate_data[idx]['answer'],
                        "topic_difficulty": cate_data[idx]['topic_difficulty'],
                        "question_type": cate_data[idx]['question_type'],
                        "subfield": cate_data[idx]['subfield'],
                        "text_concat_image_file": text_concat_image_file,
                        "text_concat_image_file_resize": text_concat_image_file_resize
                    })
            if datas:
                return datas
            else:
                return None


def load_zero_shot_mmmu_validation(args):
    dataset_root = "dataset/MMMU/dataset_validation/MMMU_testset"
    categories = os.listdir(dataset_root)
    for category_i in categories:
        if args.category == category_i:
            cate_root = f"{dataset_root}/{category_i}/images"
            cate_data_file = f"{dataset_root}/{category_i}/testset.json"
            cate_data = load_json(cate_data_file)

            datas = []
            for idx in range(len(cate_data)):
                # data = df_datas[raw_idx]
                id_ = cate_data[idx]['id']
                image_file = []
                if re.sub(r"_\d+", "", cate_data[idx]['id'].split('validation_')[1]) == args.category:
                    image_file_count = sum([1 if cate_data[idx][f'image_{i}'] else 0 for i in range(1, 8)])
                    for i in range(1, image_file_count + 1):
                        temp_image_file = cate_data[idx][f'image_{i}']
                        image_file.append(f"{cate_root}/{temp_image_file}")
                    # create_dir(f'{dataset_root}/mmmu_text_concat_zero_shot')
                    # create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}")
                    # create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}/images")
                    create_dir('dataset/MMMU/dataset_zero_shot_demo')
                    create_dir(f"dataset/MMMU/dataset_zero_shot_demo/{args.category}")
                    create_dir(
                        f"dataset/MMMU/dataset_zero_shot_demo/{args.category}/images")
                    # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
                    text_concat_image_file = f"dataset/MMMU/dataset_zero_shot_demo/{args.category}/images/{cate_data[idx]['id']}.png"
                    text_concat_image_file_resize = f"dataset/MMMU/dataset_zero_shot_demo/{args.category}/images/{cate_data[idx]['id']}_resize.png"
                else:
                    image_file = None
                question = cate_data[idx]['question']
                options = ast.literal_eval(cate_data[idx]['options'])
                options = str([f"{chr(65 + i)}. '{item}'" for i, item in enumerate(options)])
                options = options.replace("'", "").replace('[', '').replace(']', '')

                # if image_file is not None and not os.path.exists(image_file):
                #     warnings.warn(f"找不到文件：{image_file}")
                #     continue
                if image_file is not None:
                    datas.append({
                        "id": id_,
                        "question": question,
                        "ori_image": image_file,
                        "options": options,
                        "explanation": cate_data[idx]['explanation'],
                        "image_1": cate_data[idx]['image_1'],
                        "image_2": cate_data[idx]['image_2'],
                        "image_3": cate_data[idx]['image_3'],
                        "image_4": cate_data[idx]['image_4'],
                        "image_5": cate_data[idx]['image_5'],
                        "image_6": cate_data[idx]['image_6'],
                        "image_7": cate_data[idx]['image_7'],
                        "img_type": cate_data[idx]['img_type'],
                        "answer": cate_data[idx]['answer'],
                        "topic_difficulty": cate_data[idx]['topic_difficulty'],
                        "question_type": cate_data[idx]['question_type'],
                        "subfield": cate_data[idx]['subfield'],
                        "text_concat_image_file": text_concat_image_file,
                        "text_concat_image_file_resize": text_concat_image_file_resize
                    })
            if datas:
                return datas
            else:
                return None


def load_zero_shot_mixup_mmmu_validation(args):
    dataset_root = "dataset/MMMU/dataset_validation/MMMU_testset"
    categories = os.listdir(dataset_root)
    for category_i in categories:
        if args.category == category_i:
            cate_root = f"{dataset_root}/{category_i}/images"
            cate_data_file = f"{dataset_root}/{category_i}/testset.json"
            cate_data = load_json(cate_data_file)
            datas = []
            for idx in range(len(cate_data)):
                # data = df_datas[raw_idx]
                id_ = cate_data[idx]['id']
                image_file = []
                if re.sub(r"_\d+", "", cate_data[idx]['id'].split('validation_')[1]) == args.category:
                    image_file_count = sum([1 if cate_data[idx][f'image_{i}'] else 0 for i in range(1, 8)])
                    for i in range(1, image_file_count + 1):
                        temp_image_file = cate_data[idx][f'image_{i}']
                        image_file.append(f"{cate_root}/{temp_image_file}")
                    create_dir(f'{dataset_root}/mmmu_text_concat_zero_shot')
                    create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}")
                    create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}/images")
                    create_dir('dataset/MMMU/dataset_zero_shot_imageonly_demo')
                    create_dir(f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}")
                    create_dir(
                        f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images")
                    # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
                    text_concat_image_file = f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images/{cate_data[idx]['id']}.png"
                    text_concat_image_file_resize = f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images/{cate_data[idx]['id']}_resize.png"
                else:
                    image_file = None
                question = cate_data[idx]['question']
                options = ast.literal_eval(cate_data[idx]['options'])
                options = str([f"{chr(65 + i)}. '{item}'" for i, item in enumerate(options)])
                options = options.replace("'", "").replace('[', '').replace(']', '')

                # if image_file is not None and not os.path.exists(image_file):
                #     warnings.warn(f"找不到文件：{image_file}")
                #     continue
                if image_file is not None:
                    datas.append({
                        "id": id_,
                        "question": question,
                        "ori_image": image_file,
                        "options": options,
                        "explanation": cate_data[idx]['explanation'],
                        "image_1": cate_data[idx]['image_1'],
                        "image_2": cate_data[idx]['image_2'],
                        "image_3": cate_data[idx]['image_3'],
                        "image_4": cate_data[idx]['image_4'],
                        "image_5": cate_data[idx]['image_5'],
                        "image_6": cate_data[idx]['image_6'],
                        "image_7": cate_data[idx]['image_7'],
                        "img_type": cate_data[idx]['img_type'],
                        "answer": cate_data[idx]['answer'],
                        "topic_difficulty": cate_data[idx]['topic_difficulty'],
                        "question_type": cate_data[idx]['question_type'],
                        "subfield": cate_data[idx]['subfield'],
                        "text_concat_image_file": text_concat_image_file,
                        "text_concat_image_file_resize": text_concat_image_file_resize
                    })
            if datas:
                return datas
            else:
                return None


def load_mmmu_few_shot_mix(args):
    Accounting = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example. 
# Demonstration Example
# Question: Each of the following situations relates to a different company. For company B, find the missing amounts.
# Choices: [A. '$63,020',  B. '$58,410' ,  C. '$71,320' ,  D. '$77,490]
# Answer: Let's think step by step. For company B, Revenues is 1480500, Expenses is 1518300, Gains is ?, Losses is 0, Net Income or (Loss) is 39690. Net Income or (Losses) = Revenues - Expenses + Gains - Losses. Then, 39690 = 1480500 - 1518300 + ? - 0. Therefore, Gains is 77490. The answer is D. 77490

# Test Example
"""
    Agriculture = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: This isn't related to plant diseases, but it's crucial for the production of potatoes. Can you tell me if either of these two insects, or maybe even both, are harmful to potatoes?
# Choices: A. 'Neither are pest of potato', B. 'The one with black coloured antennae',  C. 'The one with tan coloured antennae', D. 'Both are pests of potato'
# Answer: Let's think step by step.The correct answer is:The one with black coloured antennae. This is the famous Colorado beetle Leptinotarsa decemlineata an innocuous insect which only became a serious pest of potato when the potato was introduced to its native range. The beetle with the tan antennae is the sunflower beetle Zygogramma exclamationis. Photo credit: Marco Verch Black antennae beetle; Keith Roragen, Tan antennae beetle both Flickr. Therefore, the answer is B.

# Test Example
    """
    Art = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: by Mark Gertler can be found in the Touchstones Rochdale museum. Which artist belonging to the Bloomsbury group was Gertler in a relationship with?
# Choices: [A.'Vanessa Bell', B.'Eileen Agar', C.'Dora Carrington', D.'Leonora Carrington']
# Answer: Let's think step by step. Gertler and Carrington met at the Slade School of Fine Art in the early 1910s, alongside their friend and fellow artist Richard Nevinson. When both men fell in love with Carrington, Gertler wrote to Nevinson: 'I am writing here to tell you that our friendship must end from now, my sole reason being that I am in love with Carrington and I have reason to believe that you are so too. Therefore, much as I have tried to overlook it, I have come to the conclusion that rivals, and rivals in love, cannot be friends.' Image: The Bokhara Coat, 1920, Mark Gertler (1891-1939); Bridgeman Images. Therefore, the answer is C.

# Test Example
    """
    Art_Theory = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: Church interiors from this time period typically were covered with 
# Choices: [A.'timber roofs',  B.'quadripartite vaults',  C.'pendentive domes', D.'masonry barrel vaults']
# Answer: Let's think step by step. Looking at the image, we can see the interior of a church with a timber roof, which is characterized by a wooden, coffered ceiling. Therefore, the answer to the question is: A. Timber roofs

# Test Example
        """
    Architecture_and_Engineering = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: Observers stationed at two sections XX and YY, 152 m apart on a highway, record the arrival times of four vehicles as shown in the accompanying table. If the total time of observation at XX was 15 sec, determine the space mean speed. 
# Choices: [A.'41.67mi/hr', B.'45.67mi/hr', C.'45.17mi/hr']
# Answer: Let's think step by step.To calculate the space mean speed, we need to determine the total distance traveled by all vehicles and the total time taken by all vehicles to travel that distance. We can use the times at which the vehicles arrived at section YY to calculate the total time taken by all vehicles to travel from XX to YY.
# Vehicle A: TA =(To + 7.58) - T = 7.58 seconds
# Vehicle B: TB =(To + 9.18) -(Z + 3) = 6.18 seconds
# Vehicle C: TC = (To + 12.36) -(T + 6) = 6.36 seconds
# Vehicle D: TD =(T + 21.74) -(T + 12) = 9.74 seconds
# Space Mean Speed = Total Distances/Total Time
# Total Distance = 152*4 = 608 meters
# Total Time = 7.58 +6.18 + 6.36 + 9.74 = 29.86 seconds
# 1 mi/hr = 1.60934 km/hr
# Space Mean Speed = (3600*608/29.86)/1.60934 = 45.67mi/hr. Therefore, the answer is B.

# Test Example
    """
    Basic_Medical_Science = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: What kind of tissue does this image depict? 
# Choices: [A.'Skeletal muscle', B.'Cartilage', C.'Blood']
# Answer: Let's think step by step. The image depicts a microscopic view of 'C: Blood'. You can identify this by the presence of red blood cells (erythrocytes), which are the numerous pink, disc-shaped cells. The large purple cell in the center with a lobed nucleus is likely a white blood cell (leukocyte), which is part of the body's immune response. This is not characteristic of skeletal muscle or cartilage tissue. Therefore, the answer is C.

# Test Example
    """
    History = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: In which of the following ways did the Estates General transform into the National Assembly? 
# Choices: [A.'The king granted the title of National Assembly to the First Estate after it advocated for greater royal control of church officials', B.'The Second Estate, refusing to shoulder a greater tax burdon, announced the reformation of a completely aristocratic National Assembly', C.'The Third Estate, whose numbers exceeded the combined numbers of the other estates, simply declared itself the National Assembly', D.'The Fourth Estate, refusing to report on the controversies of Estates General, introduced itself as an alternative legislature called the National Assembly']
# Answer: Let's think step by step. The Estates General transformed into the National Assembly through the actions of the Third Estate. The correct answer to the question is: C. 'The Third Estate, whose numbers exceeded the combined numbers of the other estates, simply declared itself the National Assembly'. Therefore, the answer is C.

# Test Example
    """
    Literature = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: Refer to the figure, which term refers to when a challenge is successful and materials are removed from the curriculum or library? 
# Choices: [A.'Writing-in-role', B.'Challenge', C.'Banning', D.'Hot Seat']
# Answer: Let's think step by step. The term that refers to when a challenge is successful and materials are removed from the curriculum or library is C. 'Banning'. Therefore, the answer is C.

# Test Example
    """
    Manage = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: The Image is the Five Forces model. What is the joint function of these five forces? 
# Choices: [A.'Profit structure', B.'cost structure', C.'Industrial structure', D.'Company personnel structure']
# Answer: Let's think step by step.The image shows the Five Forces model, which is a framework for analyzing the competitive forces within an industry. The joint function of these five forces is to determine the competitive intensity and, therefore, the attractiveness of an industry in terms of its profitability. The correct answer from the given choices is: C. 'Industrial structure' This is because the Five Forces model, developed by Michael E. Porter, is used to analyze the industry structure and understand the level of competition within that industry. The forces include the bargaining power of suppliers, the bargaining power of buyers, the threat of new entrants, the threat of substitute products or services, and the rivalry among existing competitors. These forces shape the industry structure and influence the ability of firms within the industry to earn above-average returns. Therefore, the answer is C.

# Test Example
    """
    Marketing = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: The data in Table 8.2 are the result of a random survey of 39 national flags (with replacement between picks) from various countries. Let X = the number of colors on a national flag. Image using the same x_bar, sx, and level of confidence, suppose that n were 69 instead of 39. Would the error bound become larger or smaller? 
# Choices: [A.'Larger.', B.'Smaller.', C.'Not certain.']
# Answer: Let's think step by step. The image shows a table with two columns labeled "X" and "Freq." which represent the number of colors on a national flag and their respective frequencies from a survey of 39 national flags. The question asks whether the error bound would become larger or smaller if the sample size were increased to 69 instead of 39, while keeping the sample mean (\(\bar{x}\)), sample standard deviation (s_x), and level of confidence the same. The error bound in a confidence interval is inversely related to the square root of the sample size (n). This means that as the sample size increases, the error bound decreases, assuming that the level of confidence and the variability of the data (as measured by the standard deviation) remain constant. Therefore, if the sample size were increased to 69, the error bound would become smaller. Therefore, the answer is: B. 'Smaller.'

# Test Example
    """
    Materials = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: Which of the following options is consistent with the unit cell plane depicted below? 
# Choices: [A.'SC {100}', B.'BCC {100}', C.'FCC {100}', D.'SC {110}', E.'BCC {110}', F.'FCC {110}', G.'SC {111}', H.'BCC {111}', I.'FCC {111}']
# Answer: Let's think step by step. The image shows a two-dimensional representation of a crystallographic plane with circular cross-sections of atoms arranged in a square pattern. The question asks which unit cell plane is consistent with this depiction. The correct answer is C. FCC {100}. This is because in a face-centered cubic (FCC) structure, the {100} planes are the planes that cut through the faces of the cube, and the atoms at the face centers and corners form a square pattern when viewed along the <100> directions. Therefore, the answer is C.

# Test Example
    """
    Math = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question: Let f be twice differentiable function on the interval -1 < x < 5 with f(1) = 0 and f(2) = 3.The graph of f' , the derivative of f , is shown above. The graph of f'crosses the x- axis at x=-0.5 and x =4 . Let h be the function given by $h(x)=f({\\sqrt{x+1}})$. which is the equation for the line tangent to the graph of h at x = 3 
# Choices: [A.'y = 5x/12 + 7/4 ', B.'y = 5x/12 + 5/4', C.'y = 7x/12 + 7/4']
# Answer: Let's think step by step. The graph provided is the graph of \( f' \), the derivative of the function \( f \). We are given that \( f \) is twice differentiable on the interval \( -1 < x < 5 \), with \( f(1) = 0 \) and \( f(2) = 3 \). We are also given a function \( h \) defined by \( h(x) = f(\sqrt{x + 1}) \), and we need to find the equation of the tangent line to the graph of \( h \) at \( x = 3 \). To find the equation of the tangent line to \( h \) at \( x = 3 \), we need to determine the slope of the tangent line, which is given by the derivative of \( h \) evaluated at \( x = 3 \), \( h'(3) \).
# First, let's find \( h'(x) \) using the chain rule:
# \[ h'(x) = f'(\sqrt{x + 1}) \cdot \frac{d}{dx}(\sqrt{x + 1}) \]; \[ h'(x) = f'(\sqrt{x + 1}) \cdot \frac{1}{2\sqrt{x + 1}} \]
# Now, we evaluate \( h'(3) \): \[ h'(3) = f'(\sqrt{3 + 1}) * 1/4 f'(2) = 5/3; h'(3) = 5/12
# The slope of the tangent line is h'(3), and the tangent line passes through the point (3,h(3)), which is (3,3). y-3 = h'(3)(x-3); y= 5/12x + 7/4. Therefore, the answer is A.

# Test Example
    """
    Mechanical_Engineering = """There is a demonstration example and a test example. \nFollowing the reasoning format in the demonstration example, reason the answer to the question of the test example.\nOnly output the reasoning of test example.
# Demonstration Example
# Question：In the pinewood-derby event shown, the car is released from rest at the starting position A and then rolls down the incline and on to the finish line C. If the constant acceleration down the incline is 2.75 m/s^2 and the speed from B to C is essentially constant, determine the time duration $t_{AC}$ for the race. The effects of the small transition area at B can be neglected. 
# Choices: [A.'t=1.46s', B.'t=2.46s', C.'t=3.46s']
# Answer: Let's think step by step. To solve this problem, we need to calculate the time it takes for the car to travel from point A to point C in two segments: from A to B (down the incline) and from B to C (on the flat surface).
# First, let's calculate the time it takes to go from A to B down the incline. We can use the kinematic equation for constant acceleration: \[ s = ut + \frac{1}{2}at^2 \] where: - \( s \) is the distance traveled (3 m in this case), - \( u \) is the initial velocity (0 m/s, since the car starts from rest), - \( a \) is the acceleration (2.75 m/s²), - \( t \) is the time.
# Plugging in the values we have: \[ 3 = 0 \cdot t + \frac{1}{2} \cdot 2.75 \cdot t^2 \] Solving for \( t \): \[ t1 \approx 1.477 \text{ s} \ Next, let's calculate the time it takes to go from B to C. v(B) = u(A) + at. v(B) = 2.75*1.47 = 4.0425. t2 = s/v=4/4.0425 = 0.989. t = t1 + t2 = 2.459. Therefore, the answer is B.2.46s

# Test Example
    """
    Music = """
    """
    dataset_root = "dataset/MMMU/dataset_validation/MMMU_testset"
    demo_file_root = "dataset/MMMU/dataset_mix_demo"
    categories = os.listdir(dataset_root)
    for category_i in categories:
        if args.category == category_i:
            cate_root = f"{dataset_root}/{category_i}/images"
            cate_data_file = f"{dataset_root}/{category_i}/testset.json"
            cate_data = load_json(cate_data_file)
            datas = []
            demo_file = f'{demo_file_root}/{category_i}_mix_demo.png'
            demo_text = eval(f'{category_i}')
            for idx in range(len(cate_data)):
                # data = df_datas[raw_idx]
                id_ = cate_data[idx]['id']
                image_file = f"dataset/MMMU/dataset_zero_shot_demo/{category_i}/images/validation_{category_i}_{idx + 1}.png"
                # if re.sub(r"_\d+", "", cate_data[idx]['id'].split('validation_')[1]) == args.category:
                #     image_file_count = sum([1 if cate_data[idx][f'image_{i}'] else 0 for i in range(1, 8)])
                #     for i in range(1, image_file_count + 1):
                #         temp_image_file = cate_data[idx][f'image_{i}']
                #         image_file.append(f"{cate_root}/{temp_image_file}")
                #     create_dir(f'{dataset_root}/mmmu_text_concat_zero_shot')
                #     create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}")
                #     create_dir(f"{dataset_root}/mmmu_text_concat_zero_shot/{category_i}/images")
                #     create_dir('dataset/MMMU/dataset_zero_shot_imageonly_demo')
                #     create_dir(f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}")
                #     create_dir(
                #         f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images")
                #     # create_dir(f"dataset/mathvista_testmini/testmini_text_concat/{data[idx]['category']}/{data[idx]['context']}/images/")
                #     text_concat_image_file = f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images/{cate_data[idx]['id']}.png"
                #     text_concat_image_file_resize = f"dataset/MMMU/dataset_zero_shot_imageonly_demo/{args.category}/images/{cate_data[idx]['id']}_resize.png"
                # else:
                #     image_file = None
                question = cate_data[idx]['question']
                options = ast.literal_eval(cate_data[idx]['options'])
                options = str([f"{chr(65 + i)}. '{item}'" for i, item in enumerate(options)])
                options = options.replace("'", "").replace('[', '').replace(']', '')
                if image_file is not None:
                    datas.append({
                        "id": id_,
                        "question": question,
                        "ori_image": image_file,
                        "demo_file": demo_file,
                        "demo_text": demo_text,
                        "options": options,
                        "explanation": cate_data[idx]['explanation'],
                        "image_1": cate_data[idx]['image_1'],
                        "image_2": cate_data[idx]['image_2'],
                        "image_3": cate_data[idx]['image_3'],
                        "image_4": cate_data[idx]['image_4'],
                        "image_5": cate_data[idx]['image_5'],
                        "image_6": cate_data[idx]['image_6'],
                        "image_7": cate_data[idx]['image_7'],
                        "img_type": cate_data[idx]['img_type'],
                        "answer": cate_data[idx]['answer'],
                        "topic_difficulty": cate_data[idx]['topic_difficulty'],
                        "question_type": cate_data[idx]['question_type'],
                        "subfield": cate_data[idx]['subfield'],
                        # "text_concat_image_file": text_concat_image_file,
                        # "text_concat_image_file_resize": text_concat_image_file_resize
                    })
            if datas:
                return datas
            else:
                return None


def load_yesorno50_demo_to():
    dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    create_dir(dataset_root)
    datas = []
    c_demo_text = """Learn from demonstration examples and generate detailed caption for the image of test example.
# Demonstration Example 1
Caption: A brindle-coated dog with a blue collar stands attentively on a lush green lawn, its tongue hanging out as it looks up with anticipation at a badminton racket held by an unseen player. The dog's posture suggests it is engaged in play and its panting indicates it might be feeling warm from the activity.

# Demonstration Example 2
Caption: This image shows a red fire hydrant with a white base standing on a sidewalk. The hydrant is not attached to a chain. In the background, the blurred motion of a pedestrian's legs can be seen, suggesting movement and the busy nature of the area. The pavement is composed of square tiles, and the time appears to be either dusk or night due to the artificial lighting.

# Test Example
Caption: 
"""
    for data in ori_datas:
        ori_image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        c_image_file = f"{dataset_root}/caption_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                  '0') + ".jpg"
        c_demo_file = f"{dataset_root}/vqa_yesorno_caption_demo.jpg"
        # image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        # demo_h, demo_w = get_image_size(c_demo_file)
        # text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(c_demo_file, ori_image_file, c_image_file, pos="c")
        answer_file = f"{dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text_for_question = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Answer the question according to the description.
Description: A brindle-coated dog with a blue collar stands attentively on a lush green lawn, its tongue hanging out as it looks up with anticipation at a badminton racket held by an unseen player. The dog's posture suggests it is engaged in play and its panting indicates it might be feeling warm from the activity.
Question: Is the dog hot?
Answer: Let's think step by step. Based on the description, it is mentioned that the dog is panting, which is often a behavior exhibited by dogs to cool themselves down when they are warm. Therefore, considering the panting behavior, it can be inferred that the dog might be feeling hot from the activity. So, the answer to the question "Is the dog hot?" would be yes.

# Demonstration Example 2
Answer the question according to the description.
Description: This image shows a red fire hydrant with a white base standing on a sidewalk. The hydrant is not attached to a chain. In the background, the blurred motion of a pedestrian's legs can be seen, suggesting movement and the busy nature of the area. The pavement is composed of square tiles, and the time appears to be either dusk or night due to the artificial lighting.
Question: Is there a chain on the hydrant?
Answer: Let's think step by step. Based on the description, it is mentioned that the red fire hydrant is not attached to a chain. Therefore, the answer to the question "Is there a chain on the hydrant?" is no.

# Test Example
Description: {}
Question: {}
Answer: 
"""
        text = data.get("question")
        text_for_caption = c_demo_text
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": c_image_file,
             "label": answer, "text_for_caption": text_for_caption, "text_for_question": text_for_question}
        )
    return datas


def load_counting_demo_to():
    dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    create_dir(dataset_root)
    datas = []
    c_demo_text = """Learn from demonstration examples and generate detailed caption for the image of test example.
# Demonstration Example 1
Caption: In the image, two female soccer players are in the midst of a competitive match, representing two different teams. One player is wearing a green and white striped jersey with matching shorts, while the other is dressed in a red jersey with the word "MADRID" printed on it, paired with black shorts. Both players are focused intently on the soccer ball at their feet, indicating an ongoing play. The artificial turf field adds a vibrant green backdrop to the action. There are two distinct teams visible in this shot.

# Demonstration Example 2
Caption: In the image, we see a collection of vintage double-decker buses parked on a grassy field. Each bus features two levels, with the front one displaying the number 49 and a sign indicating it's on a private hire. There are at least three buses visible, with the closest two showcasing a classic orange and white color scheme, while the third bus, partially obscured in the background, sports a blue and cream livery. The setting appears to be a bus rally or exhibition where enthusiasts gather to admire these historical vehicles.

# Test Example
Caption: 
"""
    for data in ori_datas:
        ori_image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        c_image_file = f"{dataset_root}/caption_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                  '0') + ".jpg"
        c_demo_file = f"{dataset_root}/vqa_counting_caption_demo.jpg"
        # image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        # demo_h, demo_w = get_image_size(c_demo_file)
        # text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(c_demo_file, ori_image_file, c_image_file, pos="c")
        answer_file = f"{dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text_for_question = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Answer the question according to the description.
Description: In the image, two female soccer players are in the midst of a competitive match, representing two different teams. One player is wearing a green and white striped jersey with matching shorts, while the other is dressed in a red jersey with the word "MADRID" printed on it, paired with black shorts. Both players are focused intently on the soccer ball at their feet, indicating an ongoing play. The artificial turf field adds a vibrant green backdrop to the action. There are two distinct teams visible in this shot.
Question: How many different teams in this shot?
Answer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot.

# Demonstration Example 2
Answer the question according to the description.
Description: In the image, we see a collection of vintage double-decker buses parked on a grassy field. Each bus features two levels, with the front one displaying the number 49 and a sign indicating it's on a private hire. There are at least three buses visible, with the closest two showcasing a classic orange and white color scheme, while the third bus, partially obscured in the background, sports a blue and cream livery. The setting appears to be a bus rally or exhibition where enthusiasts gather to admire these historical vehicles.
Question: How many levels do the buses have?
Answer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6.

# Test Example
Description: {}
Question: {}
Answer: 
"""
        text = data.get("question")
        text_for_caption = c_demo_text
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": c_image_file,
             "label": answer, "text_for_caption": text_for_caption, "text_for_question": text_for_question}
        )
    return datas


def load_sampled100_demo_to():
    dataset_root = "dataset/vqa2_sampled100"
    ori_datas = load_json(f"{dataset_root}/questions.json")
    create_dir(dataset_root)
    datas = []
    c_demo_text = """Learn from demonstration examples and generate detailed caption for the image of test example.
# Demonstration Example 1
Caption: A white food truck is parked under a clear blue sky, with the words "BARBACOA & FRUTAS EL VAQUERO" prominently displayed on its side, indicating that Spanish is the language used for the writing on the truck. The truck offers a variety of barbecued items and fruits, as suggested by the signage. A blue canopy tent is set up next to the truck, providing shade for a seating area with blue chairs arranged around a couple of tables. A menu board is visible under the tent, and a car is parked in the background, partially obscured by the tent. The scene suggests a casual outdoor dining experience, likely in a place where Spanish is spoken or where there is a Spanish-speaking community.

# Demonstration Example 2
Caption: A man is relaxing in a low-slung beach chair on a sandy shore, sporting a casual LA cap and a dark t-shirt with a light emblem. He appears to be enjoying a sunny day at the beach, surrounded by other beachgoers who have set up colorful umbrellas and tents. The clear blue sky suggests it's a perfect day for outdoor activities. In the background, industrial structures can be seen, hinting that this beach might be near a coastal city or port area. The man holds a smartphone in his hand, possibly taking a break from technology to soak in the seaside ambiance.

# Test Example
Caption: 
"""
    for data in ori_datas:
        ori_image_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        c_image_file = f"{dataset_root}/caption_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                  '0') + ".jpg"
        c_demo_file = f"{dataset_root}/vqa_random_caption_demo.jpg"

        images_concat(c_demo_file, ori_image_file, c_image_file, pos="c")
        answer_file = f"{dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text_for_question = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Answer the question according to the description.
Description: A white food truck is parked under a clear blue sky, with the words "BARBACOA & FRUTAS EL VAQUERO" prominently displayed on its side, indicating that Spanish is the language used for the writing on the truck. The truck offers a variety of barbecued items and fruits, as suggested by the signage. A blue canopy tent is set up next to the truck, providing shade for a seating area with blue chairs arranged around a couple of tables. A menu board is visible under the tent, and a car is parked in the background, partially obscured by the tent. The scene suggests a casual outdoor dining experience, likely in a place where Spanish is spoken or where there is a Spanish-speaking community.
Question: What language is the writing on the truck in?
Answer: Let's think step by step. The writing on the truck is in Spanish. The text "BARBACOA & FRUTAS EL VAQUERO" translates to "Barbecue & Fruits The Cowboy" in English. Therefore, the answer is Spanish.

# Demonstration Example 2
Answer the question according to the description.
Description: A man is relaxing in a low-slung beach chair on a sandy shore, sporting a casual LA cap and a dark t-shirt with a light emblem. He appears to be enjoying a sunny day at the beach, surrounded by other beachgoers who have set up colorful umbrellas and tents. The clear blue sky suggests it's a perfect day for outdoor activities. In the background, industrial structures can be seen, hinting that this beach might be near a coastal city or port area. The man holds a smartphone in his hand, possibly taking a break from technology to soak in the seaside ambiance.
Question: Where is this man?
Answer: Let's think step by step. The man in the image appears to be at a beach. You can tell by the sandy ground he's sitting on, the presence of beach chairs and umbrellas, and other people in the background who seem to be enjoying a day at the beach. The weather looks sunny and clear, which is typical for a beach setting. Therefore, the answer is beach.

# Test Example
Description: {}
Question: {}
Answer: 
"""
        text = data.get("question")
        text_for_caption = c_demo_text
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": c_image_file,
             "label": answer, "text_for_caption": text_for_caption, "text_for_question": text_for_question}
        )
    return datas


def load_yesorno50_demo_resize1():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    resize_save_root = "dataset/yesorno_50_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: Is the dog hot?
Answer: Let's think step by step. In the image, the dog appears to be panting, which is a common sign that it may be hot or has been active. Panting is a normal response in dogs to help regulate their body temperature since they cannot sweat through their skin like humans do. It's also worth noting that dogs will pant when they are excited or after exercise. Therefore, the answer is yes.

# Demonstration Example 2
Question: Is there a chain on the hydrant?
Answer: Let's think  step by step. In the provided image, there is no visible chain on the hydrant. A chain is sometimes attached to fire hydrants to secure the caps or to link to a hydrant wrench, but in this picture, such a chain is not present. The hydrant appears to have a typical design with a red top and white base, and it is standing alone without any attachments. Therefore, the answer is no.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_ysorno_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize1_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 0.25, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_resize2():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    resize_save_root = "dataset/yesorno_50_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: Is the dog hot?
Answer: Let's think step by step. In the image, the dog appears to be panting, which is a common sign that it may be hot or has been active. Panting is a normal response in dogs to help regulate their body temperature since they cannot sweat through their skin like humans do. It's also worth noting that dogs will pant when they are excited or after exercise. Therefore, the answer is yes.

# Demonstration Example 2
Question: Is there a chain on the hydrant?
Answer: Let's think  step by step. In the provided image, there is no visible chain on the hydrant. A chain is sometimes attached to fire hydrants to secure the caps or to link to a hydrant wrench, but in this picture, such a chain is not present. The hydrant appears to have a typical design with a red top and white base, and it is standing alone without any attachments. Therefore, the answer is no.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_ysorno_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize2_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 0.5, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_resize4():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    resize_save_root = "dataset/yesorno_50_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: Is the dog hot?
Answer: Let's think step by step. In the image, the dog appears to be panting, which is a common sign that it may be hot or has been active. Panting is a normal response in dogs to help regulate their body temperature since they cannot sweat through their skin like humans do. It's also worth noting that dogs will pant when they are excited or after exercise. Therefore, the answer is yes.

# Demonstration Example 2
Question: Is there a chain on the hydrant?
Answer: Let's think  step by step. In the provided image, there is no visible chain on the hydrant. A chain is sometimes attached to fire hydrants to secure the caps or to link to a hydrant wrench, but in this picture, such a chain is not present. The hydrant appears to have a typical design with a red top and white base, and it is standing alone without any attachments. Therefore, the answer is no.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_ysorno_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize4_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 2, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_counting_demo_resize1():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    resize_save_root = "dataset/counting_100_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: How many different teams in this shot?
Answer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot.

# Demonstration Example 2
Question: How many levels do the buses have?
Answer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_v4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize1_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 0.25, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_counting_demo_resize2():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    resize_save_root = "dataset/counting_100_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: How many different teams in this shot?
Answer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot.

# Demonstration Example 2
Question: How many levels do the buses have?
Answer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_v4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize2_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 0.5, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_counting_demo_resize4():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    resize_save_root = "dataset/counting_100_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Learn from demonstration examples and answer the question of the test example.
# Demonstration Example 1
Question: How many different teams in this shot?
Answer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot.

# Demonstration Example 2
Question: How many levels do the buses have?
Answer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6.

# Test Example
Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_v4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/resize4_with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                     '0') + ".jpg"
        image_resize(image_file, 2, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_counting_demo_resize(scale: Union[int, float]):
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    resize_save_root = "dataset/counting_100_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_v4.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/scale_{scale}_with_demo_COCO_test2015_000000" + str(
            data.get("image_id")).rjust(6, '0') + ".jpg"
        image_resize(image_file, scale, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_counting_demo_nshots(n: int):
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    n_shots_root = "dataset/counting_100_imageonly_nshots"
    create_dir(dataset_root)
    create_dir(n_shots_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_{n}shots_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                 '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_{n}.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_counting_demo_place(place: str):
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/counting_100_imageonly"
    place_root = "dataset/counting_100_imageonly_place"
    create_dir(dataset_root)
    create_dir(place_root)
    datas = []
    demo_text = """Question: {}"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{place_root}/with_demo_{place}_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                              '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_counting_demo_{place}.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file)
        if place == 'l2r':
            image_paste_anywhere(demo_file, image_only_file, image_file, x=1530, y=80)
        elif place == 't2brot':
            image_rotating_concat(demo_file, image_only_file, image_file, pos='c')
        elif place == 'rand':
            image_paste_anywhere(demo_file, image_only_file, image_file, x=160, y=150)
        else:
            raise NotImplementedError()
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_resize(scale: Union[int, float]):
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    resize_save_root = "dataset/yesorno_50_imageonly_resize"
    create_dir(dataset_root)
    create_dir(resize_save_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{dataset_root}/with_demo_COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_ysorno_demo.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)
        if not os.path.exists(image_file):
            images_concat(demo_file, image_only_file, image_file)
        resize_file = f"{resize_save_root}/scale_{scale}_with_demo_COCO_test2015_000000" + str(
            data.get("image_id")).rjust(6, '0') + ".jpg"
        image_resize(image_file, scale, resize_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": resize_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_nshots(n: int):
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    n_shots_root = "dataset/yesorno_50_imageonly_nshots"
    create_dir(dataset_root)
    create_dir(n_shots_root)
    datas = []
    demo_text = """Question: {}
"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{n_shots_root}/with_demo_{n}shots_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                                 '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo_{n}.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        demo_h, demo_w = get_image_size(demo_file)
        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file, width=demo_w)

        images_concat(demo_file, image_only_file, image_file)
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_demo_place(place: str):
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    dataset_root = "dataset/yesorno_50_imageonly"
    place_root = "dataset/yesorno_50_imageonly_place"
    create_dir(dataset_root)
    create_dir(place_root)
    datas = []
    demo_text = """Question: {}"""
    for data in ori_datas:
        ori_image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        image_file = f"{place_root}/with_demo_{place}_COCO_test2015_000000" + str(data.get("image_id")).rjust(6,
                                                                                                              '0') + ".jpg"
        demo_file = f"{dataset_root}/vqa_yesorno_demo_{place}.jpg"
        image_only_file = f"{dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"

        text_image_concat(f"Question: {data.get('question')}", ori_image_file, image_only_file)
        if place == 'l2r':
            image_paste_anywhere(demo_file, image_only_file, image_file, x=1400, y=60)
        elif place == 't2brot':
            image_rotating_concat(demo_file, image_only_file, image_file, pos='c')
        elif place == 'rand':
            image_paste_anywhere(demo_file, image_only_file, image_file, x=1260, y=520)
        else:
            raise NotImplementedError()
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {"id": str(data.get("question_id")), "text": text, "image_file": image_file,
             "label": answer}
        )
    return datas


def load_yesorno50_mixed_input():
    ori_dataset_root = "dataset/yesorno_50"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    datas = []
    demo_text = """Question: {}\nAnswer:"""
    for data in ori_datas:
        image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        demo1_file = f"{ori_dataset_root}/yesorno_mixed_input_demo1.jpg"
        demo2_file = f"{ori_dataset_root}/yesorno_mixed_input_demo2.jpg"
        demo1_prompt = "Question: Is the dog hot?\nAnswer: Let's think step by step. In the image, the dog appears to be panting, which is a common sign that it may be hot or has been active. Panting is a normal response in dogs to help regulate their body temperature since they cannot sweat through their skin like humans do. It's also worth noting that dogs will pant when they are excited or after exercise. Therefore, the answer is yes."
        demo2_prompt = "Question: Is there a chain on the hydrant?\nAnswer: Let's think  step by step. In the provided image, there is no visible chain on the hydrant. A chain is sometimes attached to fire hydrants to secure the caps or to link to a hydrant wrench, but in this picture, such a chain is not present. The hydrant appears to have a typical design with a red top and white base, and it is standing alone without any attachments. Therefore, the answer is no."
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {
                "id": str(data.get("question_id")), "text": text, "image_file": image_file, "label": answer,
                "inputs": [
                    {"image_file": demo1_file, "text": demo1_prompt},
                    {"image_file": demo2_file, "text": demo2_prompt},
                    {"image_file": image_file, "text": text}
                ]
            }
        )
    return datas


def load_counting_mixed_input():
    ori_dataset_root = "dataset/counting_100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    datas = []
    demo_text = """Question: {}\nAnswer:"""
    for data in ori_datas:
        image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        demo1_file = f"{ori_dataset_root}/counting_mixed_input_demo1.jpg"
        demo2_file = f"{ori_dataset_root}/counting_mixed_input_demo2.jpg"
        demo1_prompt = "Question: How many different teams in this shot?\nAnswer: Let's think step by step. Based on the uniforms depicted in the image, there are two different teams shown. One player is wearing a striped green and white jersey, while the other is in a solid red jersey. Each player's uniform is distinctively colored to represent their respective team, a common practice in team sports to differentiate competitors. Therefore, the answer is two different teams are represented in this shot."
        demo2_prompt = "Question: How many levels do the buses have?\nAnswer: Let's think  step by step. The image shows 3 buses, each with 2 levels, commonly known as double-decker buses. The total number of levels across all buses is 3 × 2 = 6.  Therefore, the answer is 6."
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {
                "id": str(data.get("question_id")), "text": text, "image_file": image_file, "label": answer,
                "inputs": [
                    {"image_file": demo1_file, "text": demo1_prompt},
                    {"image_file": demo2_file, "text": demo2_prompt},
                    {"image_file": image_file, "text": text}
                ]
            }
        )
    return datas


def load_random100_mixed_input():
    ori_dataset_root = "dataset/vqa2_sampled100"
    ori_datas = load_json(f"{ori_dataset_root}/questions.json")
    datas = []
    demo_text = """Question: {}\nAnswer:"""
    for data in ori_datas:
        image_file = f"{ori_dataset_root}/COCO_test2015_000000" + str(data.get("image_id")).rjust(6, '0') + ".jpg"
        answer_file = f"{ori_dataset_root}/answers/{data.get('question_id')}.txt"
        demo1_file = f"{ori_dataset_root}/random100_mixed_input_demo1.jpg"
        demo2_file = f"{ori_dataset_root}/random100_mixed_input_demo2.jpg"
        demo1_prompt = "Question: What language is the writing on the truck in?\nAnswer: Let's think step by step. The writing on the truck is in Spanish. The text \"BARBACOA & FRUTAS EL VAQUERO\" translates to \"Barbecue & Fruits The Cowboy\" in English. Therefore, the answer is Spanish."
        demo2_prompt = "Question: Where is this man?\nAnswer: Let's think step by step. The man in the image appears to be at a beach. You can tell by the sandy ground he's sitting on, the presence of beach chairs and umbrellas, and other people in the background who seem to be enjoying a day at the beach. The weather looks sunny and clear, which is typical for a beach setting. Therefore, the answer is beach."
        with open(answer_file, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        text = demo_text.format(data.get("question"))
        datas.append(
            {
                "id": str(data.get("question_id")), "text": text, "image_file": image_file, "label": answer,
                "inputs": [
                    {"image_file": demo1_file, "text": demo1_prompt},
                    {"image_file": demo2_file, "text": demo2_prompt},
                    {"image_file": image_file, "text": text}
                ]
            }
        )
    return datas


def load_math_equation_mixed_input():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    datas = []
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[50:]
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_path = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"
        image_file = f"{dataset_root}/images/with_demo_{id_}.jpg"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = f"""{f.read()}\n\nSolution:"""
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        original_img_file = image_path
        demo1_file = f"{dataset_root}/math_equ_mixed_input_demo1.png"
        demo2_file = f"{dataset_root}/math_equ_mixed_input_demo2.png"
        demo1_prompt = r"""Determine if the sequence is increasing, decreasing, or not monotonic:
$$\left\{\frac{2 n^2-1}{2 n}\right\}_{n=2}^{\infty}$$

Solution:
To determine if the given sequence \(\left\{\frac{2 n^2-1}{2 n}\right\}_{n=2}^{\infty}\) is increasing, decreasing, or not monotonic, we need to examine how the terms of the sequence change as \( n \) increases.

The general term of the sequence is given by:
\[ a_n = \frac{2n^2 - 1}{2n} \]

To check for monotonicity, we'll compare consecutive terms \( a_n \) and \( a_{n+1} \). If \( a_{n+1} > a_n \) for all \( n \geq 2 \), then the sequence is increasing. If \( a_{n+1} < a_n \) for all \( n \geq 2 \), then the sequence is decreasing. If neither condition is satisfied consistently, the sequence is not monotonic.

Let's calculate \( a_{n+1} - a_n \) and analyze the result.

The difference between consecutive terms of the sequence is given by:
\[ \Delta a_n = a_{n+1} - a_n = \frac{n^2 + n + \frac{1}{2}}{n(n + 1)} \]

Now, we need to analyze the sign of \( \Delta a_n \) as \( n \) varies from 2 to infinity.

- The numerator \( n^2 + n + \frac{1}{2} \) is always positive for \( n \geq 2 \) because it's a sum of positive terms.
- The denominator \( n(n + 1) \) is also always positive for \( n \geq 2 \) since it's a product of two positive integers.

Since both the numerator and denominator are positive, \( \Delta a_n \) is positive for all \( n \geq 2 \). This implies that \( a_{n+1} > a_n \) for all \( n \geq 2 \), and therefore, the sequence is increasing.
"""
        demo2_prompt = r"""Calculate the limit, if it exists:
$$\lim _{x \rightarrow 2}\left(8-3 x+6 x^2\right)$$

Solution:
To find the limit of the given expression as $x$ approaches $2$, we substitute $x=2$ into the expression and evaluate it. 
$$\lim _{x \rightarrow 2}\left(8-3 x+6 x^2\right) = 8-3(2)+6(2)^2 = 8-6+24 = 26$$
Therefore, the limit of the expression as $x$ approaches $2$ is $26$.

        """
        datas.append(
            {
                "id": id_, "text": text, "image_file": image_file, "original_img_file": original_img_file,
                "label": answer,
                "inputs": [
                    {"image_file": demo1_file, "text": demo1_prompt},
                    {"image_file": demo2_file, "text": demo2_prompt},
                    {"image_file": image_file, "text": text}
                ]
            }
        )
    return datas


def load_math_wp_mixed_input():
    dataset_root = "dataset/Sampled_Data_For_Experiments/Math"
    files = os.listdir(f"{dataset_root}/text")
    datas = []
    ids = []
    for file in files:
        id_ = re.search("\d+", file).group()
        ids.append(id_)
    ids = sorted(ids, key=lambda x: int(x))[:50]
    for file in files:
        id_ = re.search("\d+", file).group()
        if id_ not in ids:
            continue
        text_path = f"{dataset_root}/text/{id_}.txt"
        image_file = f"{dataset_root}/images/{id_}.jpg"
        answer_path = f"{dataset_root}/answers/{id_}.txt"
        # image_file = f"{dataset_root}/images/with_demo_{id_}.jpg"

        with open(text_path, mode="r", encoding="utf-8") as f:
            text = f"""{f.read()}\n\nSolution:"""
            f.close()
        with open(answer_path, mode="r", encoding="utf-8") as f:
            answer = f.read()
            f.close()
        original_img_file = image_file
        demo1_file = f"{dataset_root}/math_wp_mixed_input_demo1.png"
        demo2_file = f"{dataset_root}/math_wp_mixed_input_demo2.png"
        demo1_prompt = r"""Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

Solution: She bought 5 bagels for \$3 each. This means she spent 5 * \$3 = \$15 on the bagels. 
She had \$23 in beginning, so now she has \$23 - \$15 = \$8. The answer is 8.
"""
        demo2_prompt = r"""There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

Solution: There are 4 days from monday to thursday. 5 computers were added each day. 
That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning.
So now there are 9 + 20 = 29 computers. The answer is 29.
"""
        datas.append(
            {
                "id": id_, "text": text, "image_file": image_file, "original_img_file": original_img_file,
                "label": answer,
                "inputs": [
                    {"image_file": demo1_file, "text": demo1_prompt},
                    {"image_file": demo2_file, "text": demo2_prompt},
                    {"image_file": image_file, "text": text}
                ]
            }
        )
    return datas


def load_hallusionbench_mixed_input():
    dataset_root = "dataset/hallusion_bench"
    data_file = f"{dataset_root}/HallusionBench.tsv"
    df_datas = pd.read_csv(data_file, sep='\t').to_dict("records")
    datas = []
    for raw_idx in range(len(df_datas)):
        demo_root = "dataset/hallusionbench_mixed_image"
        data = df_datas[raw_idx]
        if data.get("visual_input"):
            image_file = f"{dataset_root}/{data.get('category')}/{data.get('subcategory')}/{data.get('set_id')}_{data.get('figure_id')}.png"
        else:
            image_file = None
        if image_file is None:
            continue
        text = data.get('question')
        answer = data.get('gt_answer')
        answer_detail = data.get('gt_answer_details')
        id_ = f"{data.get('category')}-{data.get('subcategory')}-{data.get('set_id')}-{data.get('figure_id')}-{data.get('question_id')}"
        demo1_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo0.png"
        text1_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo0_text.txt"
        with open(f"{demo_root}/text/{text1_file}", mode="r", encoding="utf-8") as f:
            demo1_prompt = f.read()
        inputs = [{"image_file": f"{demo_root}/{demo1_file}", "text": demo1_prompt}]

        demo2_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo1.png"
        demo3_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo2.png"
        demo4_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo3.png"
        text2_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo1_text.txt"
        text3_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo2_text.txt"
        text4_file = f"{data.get('category')}_{data.get('subcategory')}_{data.get('set_id')}_demo3_text.txt"
        for demo_file, text_file in zip([demo2_file, demo3_file, demo4_file], [text2_file, text3_file, text4_file]):
            if os.path.exists(f"{demo_root}/{demo_file}"):
                with open(f"{demo_root}/text/{text_file}", mode="r", encoding="utf-8") as f:
                    demo_prompt = f.read()
                inputs.append({"image_file": f"{demo_root}/{demo_file}", "text": demo_prompt})
        inputs.append({"image_file": image_file, "text": text})

        if image_file is not None:
            datas.append(
                {"id": id_, "text": text, "image_file": image_file, "label": answer, "label_detail": answer_detail,
                 "inputs": inputs}
            )

    return datas

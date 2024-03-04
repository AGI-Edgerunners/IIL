import argparse
import copy
import os.path
import random

import numpy as np
import openai

from src.apis import gpt4v
from src.utils import create_dir, write_json, load_json, encode_image
from src.load_dataset import load_hallusionbench_iil, load_mathvista_iil, load_vqa_iil

random.seed(2023)
np.random.seed(2023)

api_key = ""
base_url = None
# Adjust accordingly based on your current proxy settings.
proxy = 'http://127.0.0.1:4780'



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_proxy", type=bool, default=False)
    parser.add_argument("--input_modal", type=str, default="iil",
                        choices=["iil"])
    parser.add_argument("--test_sample", type=int, default=None)

    parser.add_argument("--dataset", type=str, default="vqa", choices=['hallusionbench','mathvista','vqa'])
    # parser.add_argument("--category", type=str, default='math-targeted-vqa',
    #                     choices=['general-vqa', 'math-targeted-vqa'])
    # parser.add_argument("--sub_category", type=str, default='table')
    parser.add_argument("--exp_name", type=str, default="public_code_test01",
                        help="automatically resume experiment by matching the same experiment name")
    parser.add_argument("--lt", type=str, default="few_shot", choices=["zero_shot", "few_shot"],
                        help="zero_shot or few_shot")

    #
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # 文件夹
    result_dir = f"result/iil/{args.lt}"
    create_dir(result_dir)
    if args.dataset == "mathvista":
        category = {
            'math-targeted-vqa': ['abstract_scene', 'bar_chart', 'function_plot', 'geometry_diagram', 'line_plot',
                                  'puzzle_test', 'scientific_figure', 'scatter_plot', 'synthetic_scene', 'table'],
            'general-vqa': ['abstract_scene', 'bar_chart', 'document_image', 'line_plot', 'map_chart', 'medical_image',
                            'natural_image', 'pie_chart', 'scatter_plot', 'scientific_figure', 'synthetic_scene']
        }
    elif args.dataset == 'hallusionbench':
        category = {'all': [None]}
    elif args.dataset == 'vqa':
        category = {'counting50': [None], 'yesorno50': [None], 'random50': [None]}
    else:
        raise NotImplementedError(f"not support dataset: {args.dataset}")
    for cate, sub_cates in category.items():
        args.category = cate
        for i in range(len(sub_cates)):
            args.sub_category = sub_cates[i]
            result_file = f"{result_dir}/{args.exp_name}-{args.category}-{args.sub_category}-{args.lt}-{args.input_modal}.json"

            if args.use_proxy:
                openai.proxy = proxy
                os.environ["ALL_PROXY"] = proxy

            # 数据集
            if args.dataset == "mathvista":
                datas = eval(f"load_{args.dataset}_iil")(args.lt, args.category, args.sub_category)
            elif args.dataset == 'hallusionbench':
                datas = eval(f"load_{args.dataset}_iil")(args.lt)
            elif args.dataset == 'vqa':
                datas = eval(f"load_{args.dataset}_iil")(args.lt, args.category)
            else:
                raise NotImplementedError(f"not support dataset: {args.dataset}")

            if args.test_sample:
                datas = random.sample(datas, k=args.test_sample)
            results = {}
            if os.path.exists(result_file):
                results = load_json(result_file)
            for data in datas:
                id_ = data.get("pid", "")
                if id_ in results:
                    continue
                image_inputs = data.get("image_inputs", [])
                text_inputs = []
                messages = [
                    {
                        "role": "user",
                        "content": [],
                    },
                ]
                type_mapping = {"jpg": "jpeg", "png": "png"}
                for image_file in image_inputs:
                    base64_image = encode_image(image_file)
                    messages[0]['content'].append({
                                "type": "image_url",
                                "image_url": f"data:image/{type_mapping[image_file.split('.')[-1]]};base64,{base64_image}",
                            })
                text = data.get("text", "")
                test_file = data.get("image_file")

                n = 3
                error_flag = False
                output = ""
                while n > 0 and not error_flag:
                    output = gpt4v(
                        messages=messages,
                        temperature=0,
                        api_key=api_key,
                        base_url=base_url
                    )
                    if output:
                        error_flag = True
                    else:
                        n -= 1

                print("\n")
                print(f"ID: {id_}")
                print(f"user: {test_file} {text}")
                print(f"chatgpt: {output}")
                print("-" * 20)

                temp_data = copy.deepcopy(data)
                temp_data["prediction"] = output
                results[id_] = temp_data

                write_json(results, result_file)
                # time.sleep(2)


if __name__ == '__main__':
    main()

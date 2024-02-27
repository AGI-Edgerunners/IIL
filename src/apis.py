import warnings

import openai
from openai import OpenAI


def chatgpt_base(model, messages, temperature, api_key, base_url=None):
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    retry = 0
    flag = False
    out = ''
    while retry < 10 and not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            out = response.choices[0].message.content
            flag = True
        except openai.APIStatusError as e:
            if e.message == "Error code: 307":
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                continue
            else:
                raise e
        except Exception as e:
            raise e
    client.close()
    return out


def gpt4v(messages, temperature, api_key, base_url=None):
    model = "gpt-4-vision-preview"
    out = chatgpt_base(model, messages, temperature, api_key, base_url)
    return out

# def gpt4(messages, temperature, api_key, base_url=None):
#     # todo:指定gpt4模型名称
#     model = ""
#     out = chatgpt_base(model, messages, temperature, api_key, base_url)
#     return out

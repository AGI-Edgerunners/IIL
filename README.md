# <img src="logo_i2l.jpg" width="30" height="40"> In-Image Learning 
Code for the paper "[All in an Aggregated Image for In-Image Learning](https://arxiv.org/abs/2402.17971)".

![IIL case](Figure1.jpg)


# Requirement
```shell
pip install -r requirements.txt
```

# Download Dataset
The processed dataset and demonstration examples are available from this [link](https://drive.google.com/file/d/1eL3BnWPMLJxm3iAptX_QQRuijW9dl9AM/view?usp=sharing).
Unzip the file after downloading and keep the ```dataset``` directory in the root directory of the project.
```text
----IIL
    |----dataset
    |----src
    ...
```
# Run In-Image Learning and Baselines
## In-Image Learning
```shell
python run_iil.py --exp_name exp_on_mv --dataset mathvista --lt few_shot
```

## Visual-text interleaved in-context learning
```shell
python run_vticl.py --exp_name exp_on_mv --dataset mathvista --lt few_shot
```

## Text-only in-context learning
```shell
python run_ticl.py --exp_name exp_on_mv --dataset mathvista --lt few_shot
```

# Cite
If you find **In-Image Learning** useful for your research and applications, please kindly cite using this BibTeX:
```latex
@misc{wang2024single,
      title={All in a Single Image: Large Multimodal Models are In-Image Learners}, 
      author={Lei Wang and Wanyu Xu and Zhiqiang Hu and Yihuai Lan and Shan Dong and Hao Wang and Roy Ka-Wei Lee and Ee-Peng Lim},
      year={2024},
      eprint={2402.17971},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

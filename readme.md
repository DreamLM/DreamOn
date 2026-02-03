
# ✏️ DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas

[![Static Badge](https://img.shields.io/badge/📰-Notion-grey)](https://foremost-success-91b.notion.site/DreamOn-Diffusion-Language-Models-For-Code-Infilling-Beyond-Fixed-Size-Canvas-228be544bdbb80cc991ef540e7805bd7)
[![Static Badge](https://img.shields.io/badge/📰-Report-yellow)](https://arxiv.org/abs/2602.01326)
[![Static Badge](https://img.shields.io/badge/📰-Blog-red)](https://hkunlp.github.io/blog/2025/dreamon/)
[![Static Badge](https://img.shields.io/badge/📰-Demo-green)](https://huggingface.co/spaces/ZiruiWu/DreamOn-v0-7B)
[![Static Badge](https://img.shields.io/badge/Hugging%20Face%20🤗-DreamOn%207B-blue)
](https://huggingface.co/Dream-org/DreamOn-v0-7B)

## Overview

DreamOn is a novel discrete diffusion algorithm designed to address the variable-length generation challenge in code infilling. Unlike current discrete diffusion language models, our approach enables dynamic expansion and contraction of mask tokens during inference, providing flexible length control without requiring predetermined canvas sizes.

This work is done as part of the [HKU NLP Group](https://hkunlp.github.io/) and [Klear Team@Kuaishou Technology](https://github.com/Kwai-Klear).

<div style="display: flex">
  <figure style="margin: 0; text-align: center;">
    <img src="figs/code_infilling_dreamon_from_short.gif" width="400" />
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="figs/code_infilling_dreamon_from_long.gif" width="400" />

  </figure>
</div>

## News
[2025/7/25] We open-source our code for training and evaluation. We also release a demo for DreamOn on Hugging Face Spaces. You can try it out [here](https://huggingface.co/spaces/ZiruiWu/DreamOn-v0-7B).

[2025/7/15] We release our model [DreamOn](https://huggingface.co/Dream-org/DreamOn-v0-7B) and its accompanying model [DreamCoder](https://github.com/DreamLM/Dream-Coder).

## Installation
Our implementation follows our previous work [Dream](https://github.com/HKUNLP/Dream/). Please install transformers by `pip install transformers==4.46.2` and `torch==2.5.1` as Dream uses the [SdpaAttention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) built in torch. 


## Quickstart
```python
import torch
import time
from transformers import AutoModel, AutoTokenizer

def process_infilling_prompt(prefix, suffix, tokenizer, number_of_mask):
    prefix = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    middle = [tokenizer.mask_token_id] * number_of_mask
    suffix = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix + middle + suffix

prefix = '''from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

suffix = '''        for idx2, elem2 in enumerate(numbers):
        if idx != idx2:
            distance = abs(elem - elem2)
            if distance < threshold:
                return True

return False
'''
model_path = 'Dream-org/DreamOn-v0-7B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

## Set the initial mask token length when processing the prompt
input_ids = process_infilling_prompt(prefix, suffix, tokenizer, 4)
input_ids = torch.LongTensor([input_ids]).to("cuda")

model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to("cuda").eval()


output = model.diffusion_generate(
    input_ids,
    temperature=0.2,
    alg = 'entropy',
    alg_temp = 0,
    top_p = 0.9,
    max_new_tokens = 64, ## Set the maximum number of new tokens for infilling
    return_dict_in_generate = True,
    output_history = True,
    number_transfer_tokens = 1
)


history = output.history
for i, h in enumerate(history):
    print(f"########################")
    time.sleep(0.2)
    print(tokenizer.decode(h.tolist()), end="\n\n")   
```

## Parameters
- `input_ids`: The input token ids.
- `max_new_tokens`: The maximum tokens to generate. Note that the context length (input+output) of Dream currently is 2048. And the mask added to the prompt is counted as new tokens. Therefore, `max_new_tokens` can not be set to a value smaller than the number of mask tokens in the prompt. For example, if you set `number_of_mask` to 4, then `max_new_tokens` should be at least 4.
- `output_history`: Whether to return the output at each intermediate step.
- `return_dict_in_generate`: The output format, mostly set to True.
- `number_of_transfer_tokens`: The number of tokens to predict at each denoising step. We mainly test our model with `number_of_transfer_tokens` set to 1. Other settings are not fully tested.
- `temperature`: The value used to module the next token probabilities. By default 0.0. The smaller the value, the more accurate the results (e.g., in math or coding). The larger the value, the more diverse the results (e.g., in general conversation). If you notice repeated results, you might consider increasing the temperature.
- `top_p`: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. By default None. Control the diversity of generation. 
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. By default None. Control the diversity of generation.
- `alg`: The remasking strategy in diffusion sampling, controlling the token generation order. Support one random strategy and three confidence-based strategies:
    - `maskgit_plus`: Token will be generated based on the top1 confidence from https://arxiv.org/abs/2202.04200. 
    - `topk_margin`: Token will be generated based on the margin confidence by taking `top1 - top2` from https://arxiv.org/abs/2502.06768. 
    - `entropy`: Token will be generated based on the entropy of each token distribution. 
- `alg_temp`: Add some randomness to `alg` when using confidence-based strategies. By default None. 

Note: We currently do not support attention mask, as we recompute attention mask each denoising step to support variable-length generation.

## Evaluation
Use the following command to replicate our results.
```
git clone https://github.com/openai/human-eval-infilling
pip install -e human-eval-infilling
pip install omegaconf
```
```
bash eval/eval_humaneval_infilling.sh
bash eval/eval_santa_fim.sh
```

## Training
Our training implementation is built upon the SFT trainer from [verl](https://github.com/volcengine/verl). To train DreamOn, please install verl first, and then execute the following command:
```
python data/prepare_data.py
bash run_dreamon.sh
```

## Citation
```bibtex
@misc{wu2026dreamon,
      title={DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas}, 
      author={Zirui Wu and Lin Zheng and Zhihui Xie and Jiacheng Ye and Jiahui Gao and Shansan Gong and Yansong Feng and Zhenguo Li and Wei Bi and Guorui Zhou and Lingpeng Kong},
      year={2026},
      eprint={2602.01326},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.01326}, 
}
```

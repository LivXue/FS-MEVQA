# FS-MEVQA
Few-Shot Multimodal Explanation for Visual Question Answering (FS-MEVQA)

### Dataset
We release the SME dataset in `dataset/dataset.zip`, inlcuding questions, answers, and multimodal explanations. 
You can also download the dataset from [Huggingface Datasets](https://huggingface.co/datasets/LivXue/SME).
The images should be downloaded from the [GQA source](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip).

If you are interesting in our dataset construction, you can refer to `1extract_semantic_structure.py`, `2semantic_tree_to_text.py`, `3correct_some_issues.py`, `4separate_text_and_box.py`, `5complete_grammar.py`, `6GPT_check_grammar.py`, `7Add_mannual_corrections.py` in the `dataset` folder.


### Method
We provide an easy-to-use end-to-end pipeline in `pipeline.py`. You can just set the image path and question in the script and run it.

If you want to reimplement our experiments on the SME dataset, you can just run `1generate_program.py`, `2generate_process.py`, `3intepretation.py`, `4revover_boxes.py`, which is much faster than run samples one-by-one.

### Our Results
We provide our results in `results/MEAgent_results.json`.

### Evaluation
We provide the evaluation script in `evaluation.py`. The detailed metrics are implemented in `language_metrics.py`, `visual_metrics.py`, `attribution_metric.py` in the `metrics` folder.

Furthermore, you should download the [pycocoevalcap package](https://github.com/sks3i/pycocoevalcap) and put it in `metrics`, which is needed for language evaluation.

### Baseline
We provide the implementation of GPT-4V for our task in `GPT4V_baseline.py`.

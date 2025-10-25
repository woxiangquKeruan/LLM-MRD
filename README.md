## Requirements
You can run pip install -r requirements.txt to deploy the environment.
## Data Preparation
1.  **Data Splitting:** In the experiments, we maintain the same data splitting scheme as the benchmarks.
2.  **Dataset:** We follow the same data preprocessing pipeline of Bootstrapping Your Own Representations for Fake News Detection. Please use the data prepare scripts provided in preprocessing or the preprocessing scripts in prior work to prepare data for each datasets. For all datasets (Weibo/Weibo-21/GossipCop), please download from the official source.

## LLM Prompts
1.  **Prompt 1 (Text-based):** "Analyze this text for misleading language, emotional manipulation, or logical fallacies. Please reason step by step to determine the likelihood of falsity and manipulation."
2.  **Prompt 2 (Image-based):** "Examine this image for signs of editing, unreasonable elements, or scenes that contradict common sense. Please reason step by step to determine the likelihood of falsity and manipulation."
3.  **Prompt 3 (Cross-modal):** "Compare the text and image. Are there any contradictions, inconsistencies, or deliberately created false connections between them? Please reason step by step to determine the likelihood of falsity and manipulation."
   
## Pretrained Models

1.  **Roberta:** You can download the pretrained Roberta model from [Roberta](<link-to-roberta>) and move all files into the `./pretrained_model` directory.
2.  **MAE:** Download the pretrained MAE model from "[Masked Autoencoders: A PyTorch Implementation](<link-to-mae>)" and move all files into the root directory.
3.  **CLIP:** Download the pretrained CLIP model from "[Chinese-CLIP](<link-to-clip>)" and move all files into the root directory.

## Training
* **Preparation:** First, download the Weibo21, Weibo, and GossipCop datasets, along with the Qwen2.5-VL model. Then, run the `generate_reasoning_*.py` scripts (e.g., `generate_reasoning_gossipcop.py`) to generate reasoning data, followed by the `encode_reasoning.py` scripts to encode that data.
* **Start Training:** After processing the data, train the model by running `python main.py --dataset gossipcop` or `python main.py --dataset weibo21` or `python main.py --dataset weibo`.


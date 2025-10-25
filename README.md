## Requirements
You can run pip install -r requirements.txt to deploy the environment.
## Data Preparation
1.  **Data Splitting:** In the experiments, we maintain the same data splitting scheme as the benchmarks.
2.  **Weibo21 Dataset:** For the Weibo21 dataset, we follow the work from (Ying et al., 2023). You should send an email to Dr. Qiong Nan to get the complete multimodal multi-domain dataset Weibo21.
3.  **Weibo Dataset:** For the Weibo dataset, we adhere to the work from (Wang et al., 2022). In addition, we have incorporated domain labels into this dataset. You can download the final processed data from the link below. By using this data, you will bypass the data preparation step. Link: https://pan.baidu.com/s/1TGc-8RUH6BiHO1rjnzuPxQ code: qwer
4.  **GossipCop Dataset:** For the GossipCop dataset, we followed the work of (XXX et al, 2023). You can download the final processed data from the link below. By using this data, you will bypass the data preparation step. Link: XXXX code: 1123
4.  **Data Storage:**
    * Place the processed Weibo data in the `./data` directory.
    * Place the Weibo21 data in the `./Weibo_21` directory.
    * Place the GossipCop data in the `./gossipcop` directory.
5.  **Data preparation:** Use `clip_data_pre`, `data_pre`, `weibo21_data_pre` and `weibo21_clip_data_pre` to preprocess the data of Weibo and Weibo21, respectively, in order to save time during the data loading phase.

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


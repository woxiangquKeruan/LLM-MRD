# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# import pandas as pd
# import os
# from tqdm import tqdm

# # --- 步骤 1: 设置模型路径并加载模型 (此部分无需修改) ---
# MODEL_PATH = "./Qwen-VL"
# print("正在加载Tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# print("正在加载Qwen-VL模型，这可能需要一些时间...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map="auto",
#     trust_remote_code=True,
#     load_in_8bit=True
# ).eval()
# model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
# print("模型加载完成！")

# # --- 步骤 2: 定义Prompts (此部分无需修改) ---
# PROMPT_TEXT_BASED    = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_IMAGE_BASED   = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_CROSS_MODAL   = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图."

# # --- 步骤 3: 准备并遍历gossipcop数据集 (此部分无需修改) ---
# print("正在加载和准备gossipcop数据集...")

# # 定义新的数据集路径
# DATA_DIR = "./gossipcop"
# TRAIN_IMG_DIR = os.path.join(DATA_DIR, "gossip_train")
# TEST_IMG_DIR  = os.path.join(DATA_DIR, "gossip_test")
# TRAIN_CSV = os.path.join(DATA_DIR, "gossip_train.csv")
# TEST_CSV  = os.path.join(DATA_DIR, "gossip_test.csv")

# news_dataset_for_processing = []
# all_files_found = True

# try:
#     # 分别加载 train 和 test 的CSV文件
#     train_df = pd.read_csv(TRAIN_CSV)
#     test_df  = pd.read_csv(TEST_CSV)
#     print(f"成功加载了 train ({len(train_df)}条) 和 test ({len(test_df)}条) CSV文件。")

#     # --- 单独处理训练集 ---
#     print("正在处理训练集路径...")
#     for index, row in train_df.iterrows():
#         image_id = row['image_id']
#         image_path = None
#         # 检查image_id是否有效
#         if pd.notna(image_id) and isinstance(image_id, str):
#             # 路径是 train 图片文件夹 + image_id
#             image_path = os.path.join(TRAIN_IMG_DIR, image_id)

#         news_dataset_for_processing.append({
#             "post_id": row['post_id'],
#             "image_id": image_id,
#             "label": row['label'],
#             "content": row['post_text'],
#             "image_path": image_path
#         })

#     # --- 单独处理测试集 ---
#     print("正在处理测试集路径...")
#     for index, row in test_df.iterrows():
#         image_id = row['image_id']
#         image_path = None
#         # 检查image_id是否有效
#         if pd.notna(image_id) and isinstance(image_id, str):
#             # 路径是 test 图片文件夹 + image_id
#             image_path = os.path.join(TEST_IMG_DIR, image_id)
        
#         news_dataset_for_processing.append({
#             "post_id": row['post_id'],
#             "image_id": image_id,
#             "label": row['label'],
#             "content": row['post_text'],
#             "image_path": image_path
#         })
    
#     print(f"数据集路径准备完成，总计 {len(news_dataset_for_processing)} 条数据。")

# except FileNotFoundError as e:
#     print(f"错误：找不到CSV文件 - {e}。请确保文件路径正确。")
#     all_files_found = False

# # --- 步骤 4: 遍历数据集，生成Reasoning (此部分已按正确格式修改) ---
# all_final_results = []
# if all_files_found:
#     print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")
#     for news_item in tqdm(news_dataset_for_processing, desc="Generating Reasoning"):
#         news_text  = news_item["content"]
#         image_path = news_item["image_path"]

#         # 1) text-only (此部分调用方式正确，无需修改)
#         query_text = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{news_text}\""
#         response_text, _ = model.chat(tokenizer, query=query_text, history=None)

#         # 2) image-only (!!! 已修改查询格式 !!!)
#         if image_path and os.path.exists(image_path):
#             # 将图片路径和prompt文本合并成一个字符串
#             query_image = f'<img>{image_path}</img>{PROMPT_IMAGE_BASED}'
#             response_image, _ = model.chat(tokenizer, query=query_image, history=None)
#         else:
#             response_image = "N/A (No Image)"

#         # 3) cross-modal (!!! 已修改查询格式 !!!)
#         if image_path and os.path.exists(image_path):
#             # 将图片路径和prompt文本合并成一个字符串
#             query_cross = f'<img>{image_path}</img>{PROMPT_CROSS_MODAL}\n\n文本内容：\"{news_text}\"'
#             response_cross, _ = model.chat(tokenizer, query=query_cross, history=None)
#         else:
#             response_cross = "N/A (No Image)"

#         all_final_results.append({
#             "post_id": news_item["post_id"],
#             "image_id": news_item["image_id"],
#             "label": news_item["label"],
#             "text_reasoning": response_text,
#             "image_reasoning": response_image,
#             "cross_modal_reasoning": response_cross
#         })

# # --- 步骤 5: 将所有结果保存到新的CSV文件 (此部分无需修改) ---
# if all_files_found:
#     output_filename = "./gossipcop_with_reasoning.csv"
#     results_df = pd.DataFrame(all_final_results)
#     results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
#     print(f"\n全部处理完毕！结果已保存到 {output_filename}")



# import torch
# import os
# import pandas as pd
# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     GenerationConfig,
#     BitsAndBytesConfig
# )

# # -------------------------------------------------
# # 1. 4-bit 量化 + CPU offload
# # -------------------------------------------------
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     llm_int8_enable_fp32_cpu_offload=True  # 允许 CPU 层
# )

# MODEL_PATH = "./Qwen-VL"
# print("正在加载Tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# print("正在加载 Qwen-VL 4-bit 量化模型（允许 CPU offload）...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     device_map="auto",
#     max_memory={0: "20GiB", "cpu": "50GiB"}  # 限制显存 20 GB
# ).eval()
# model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
# print("模型加载完成！")

# # -------------------------------------------------
# # 2. Prompts（不变）
# # -------------------------------------------------
# PROMPT_TEXT_BASED  = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_IMAGE_BASED = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_CROSS_MODAL = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图。"

# # -------------------------------------------------
# # 3. 读取并合并 gossipcop 数据集（不变）
# # -------------------------------------------------
# DATA_DIR       = "./gossipcop"
# TRAIN_IMG_DIR  = os.path.join(DATA_DIR, "gossip_train")
# TEST_IMG_DIR   = os.path.join(DATA_DIR, "gossip_test")
# TRAIN_CSV      = os.path.join(DATA_DIR, "gossip_train.csv")
# TEST_CSV       = os.path.join(DATA_DIR, "gossip_test.csv")

# news_dataset_for_processing = []

# train_df = pd.read_csv(TRAIN_CSV)
# test_df  = pd.read_csv(TEST_CSV)

# for _, row in train_df.iterrows():
#     img_id = row['image_id']
#     img_p  = os.path.join(TRAIN_IMG_DIR, img_id) if pd.notna(img_id) else None
#     news_dataset_for_processing.append({
#         "post_id": row['post_id'],
#         "image_id": img_id,
#         "label": row['label'],
#         "content": row['post_text'],
#         "image_path": img_p
#     })

# for _, row in test_df.iterrows():
#     img_id = row['image_id']
#     img_p  = os.path.join(TEST_IMG_DIR, img_id) if pd.notna(img_id) else None
#     news_dataset_for_processing.append({
#         "post_id": row['post_id'],
#         "image_id": img_id,
#         "label": row['label'],
#         "content": row['post_text'],
#         "image_path": img_p
#     })

# # -------------------------------------------------
# # 4. 生成 Reasoning（不变）
# # -------------------------------------------------
# all_final_results = []
# print(f"开始生成 Reasoning，共 {len(news_dataset_for_processing)} 条数据...")

# for item in tqdm(news_dataset_for_processing, desc="Generating"):
#     text  = item["content"]
#     img_p = item["image_path"]

#     # 1) 纯文本
#     query_txt = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{text}\""
#     resp_txt, _ = model.chat(tokenizer, query=query_txt, history=None)

#     # 2) 纯图片
#     if img_p and os.path.exists(img_p):
#         query_img = f"<img>{img_p}</img>{PROMPT_IMAGE_BASED}"
#         resp_img, _ = model.chat(tokenizer, query=query_img, history=None)
#     else:
#         resp_img = "N/A (No Image)"

#     # 3) 图文跨模态
#     if img_p and os.path.exists(img_p):
#         query_cross = f"<img>{img_p}</img>{PROMPT_CROSS_MODAL}\n\n文本内容：\"{text}\""
#         resp_cross, _ = model.chat(tokenizer, query=query_cross, history=None)
#     else:
#         resp_cross = "N/A (No Image)"

#     all_final_results.append({
#         "post_id": item["post_id"],
#         "image_id": item["image_id"],
#         "label": item["label"],
#         "text_reasoning": resp_txt,
#         "image_reasoning": resp_img,
#         "cross_modal_reasoning": resp_cross
#     })

# # -------------------------------------------------
# # 5. 保存结果
# # -------------------------------------------------
# output_csv = "./gossipcop_with_reasoning_4bit.csv"
# pd.DataFrame(all_final_results).to_csv(output_csv, index=False, encoding='utf-8-sig')
# print(f"\n全部处理完毕！结果已保存到 {output_csv}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pandas as pd
import os
from tqdm import tqdm

# --- 步骤 1: 设置模型路径并加载模型 ---
MODEL_PATH = "./Qwen-VL"
print("正在加载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("正在加载Qwen-VL模型，这可能需要一些时间...")
# !!! 修改点 1: 将模型加载方式改回 fp16 !!!
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    fp16=True  # 使用fp16确保视觉模块兼容性
).eval()
model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("模型加载完成！")

# --- 步骤 2: 定义Prompts (此部分无需修改) ---
PROMPT_TEXT_BASED    = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_IMAGE_BASED   = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_CROSS_MODAL   = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图."

# --- 步骤 3: 准备并遍历gossipcop数据集 (此部分无需修改) ---
print("正在加载和准备gossipcop数据集...")
# (代码与上一版相同，无需修改)
DATA_DIR = "./gossipcop"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "gossip_train")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "gossip_test")
TRAIN_CSV = os.path.join(DATA_DIR, "gossip_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "gossip_test.csv")
news_dataset_for_processing = []
all_files_found = True
try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"成功加载了 train ({len(train_df)}条) 和 test ({len(test_df)}条) CSV文件。")
    print("正在处理训练集路径...")
    for index, row in train_df.iterrows():
        image_id = row['image_id']
        image_path = None
        if pd.notna(image_id) and isinstance(image_id, str):
            image_path = os.path.join(TRAIN_IMG_DIR, image_id)
        news_dataset_for_processing.append({
            "post_id": row['post_id'],"image_id": image_id,"label": row['label'],"content": row['post_text'],"image_path": image_path
        })
    print("正在处理测试集路径...")
    for index, row in test_df.iterrows():
        image_id = row['image_id']
        image_path = None
        if pd.notna(image_id) and isinstance(image_id, str):
            image_path = os.path.join(TEST_IMG_DIR, image_id)
        news_dataset_for_processing.append({
            "post_id": row['post_id'],"image_id": image_id,"label": row['label'],"content": row['post_text'],"image_path": image_path
        })
    print(f"数据集路径准备完成，总计 {len(news_dataset_for_processing)} 条数据。")
except FileNotFoundError as e:
    print(f"错误：找不到CSV文件 - {e}。请确保文件路径正确。")
    all_files_found = False

# --- 步骤 4: 遍历数据集，生成Reasoning ---
all_final_results = []
if all_files_found:
    print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")
    for news_item in tqdm(news_dataset_for_processing, desc="Generating Reasoning"):
        news_text  = news_item["content"]
        image_path = news_item["image_path"]

        # !!! 修改点 2: 增加文本截断，预防显存不足 !!!
        if isinstance(news_text, str):
            news_text = news_text[:2048] # 将文本截断为最大2048字符

        # 1) text-only
        query_text = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{news_text}\""
        response_text, _ = model.chat(tokenizer, query=query_text, history=None)

        # 2) image-only
        if image_path and os.path.exists(image_path):
            query_image = f'<img>{image_path}</img>{PROMPT_IMAGE_BASED}'
            response_image, _ = model.chat(tokenizer, query=query_image, history=None)
        else:
            response_image = "N/A (No Image)"

        # 3) cross-modal
        if image_path and os.path.exists(image_path):
            query_cross = f'<img>{image_path}</img>{PROMPT_CROSS_MODAL}\n\n文本内容：\"{news_text}\"'
            response_cross, _ = model.chat(tokenizer, query=query_cross, history=None)
        else:
            response_cross = "N/A (No Image)"

        all_final_results.append({
            "post_id": news_item["post_id"],"image_id": news_item["image_id"],"label": news_item["label"],"text_reasoning": response_text,"image_reasoning": response_image,"cross_modal_reasoning": response_cross
        })

# --- 步骤 5: 将所有结果保存到新的CSV文件 ---
if all_files_found:
    output_filename = "./gossipcop_with_reasoning.csv"
    results_df = pd.DataFrame(all_final_results)
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n全部处理完毕！结果已保存到 {output_filename}")
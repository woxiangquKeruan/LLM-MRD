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
#     fp16=True
# ).eval()
# model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
# print("模型加载完成！")

# # --- 步骤 2: 定义Prompts (此部分无需修改) ---
# PROMPT_TEXT_BASED    = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_IMAGE_BASED   = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_CROSS_MODAL   = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图."

# # --- 步骤 3: 准备并遍历Weibo_21数据集 (此部分已按新要求修改) ---
# print("正在加载和准备Weibo_21数据集...")

# # 定义新的数据集路径 (已移除VAL_XLSX)
# DATA_DIR = "./Weibo_21"
# TRAIN_XLSX = os.path.join(DATA_DIR, "train_datasets.xlsx")
# TEST_XLSX  = os.path.join(DATA_DIR, "test_datasets.xlsx")

# news_dataset_for_processing = []
# all_files_found = True

# # 加载Excel文件并合并 (已移除val_df)
# try:
#     # 使用 pd.read_excel 读取 .xlsx 文件
#     train_df = pd.read_excel(TRAIN_XLSX)
#     test_df  = pd.read_excel(TEST_XLSX)
#     # 合并 train 和 test
#     combined_df = pd.concat([train_df, test_df], ignore_index=True)
#     print(f"成功加载并合并了 train 和 test XLSX文件。总计 {len(combined_df)} 条数据。")

# except FileNotFoundError as e:
#     print(f"错误：找不到XLSX文件 - {e}。请确保文件路径正确。")
#     combined_df = pd.DataFrame()
#     all_files_found = False

# # 遍历合并后的DataFrame来构建标准化的待处理数据集
# if all_files_found:
#     for index, row in combined_df.iterrows():
#         # 从Excel行中提取信息
#         content = row['content']
#         label = row['label']
#         relative_image_path = row['image'] # 例如 "rumor_images/xyz.jpg"
#         category = row['category']

#         image_path = None
#         image_id = None

#         # 检查 image 列是否有效
#         if pd.notna(relative_image_path) and isinstance(relative_image_path, str):
#             # 正确构建图片位置: "./Weibo_21/" + "image列的数值"
#             full_path = os.path.join(DATA_DIR, relative_image_path)
#             image_path = full_path
#             # 从路径中提取文件名作为image_id
#             image_id = os.path.basename(relative_image_path)
        
#         # 将所有需要的信息都放入待处理列表
#         news_dataset_for_processing.append({
#             "post_id": f"weibo21_{index}", # 使用索引生成唯一ID
#             "image_id": image_id,
#             "label": label,
#             "category": category,
#             "content": content,
#             "image_path": image_path # 这是模型需要访问的完整路径
#         })

# # --- 步骤 4: 遍历数据集，生成Reasoning (此部分无需修改) ---
# all_final_results = []
# print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")
# for news_item in tqdm(news_dataset_for_processing, desc="Generating Reasoning"):
#     news_text  = news_item["content"]
#     image_path = news_item["image_path"]

#     # 1) text-only
#     query_text = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{news_text}\""
#     response_text, _ = model.chat(tokenizer, query=query_text, history=None)

#     # 2) image-only
#     if image_path and os.path.exists(image_path):
#         response_image, _ = model.chat(
#             tokenizer,
#             query=[{"image": image_path}, {"text": PROMPT_IMAGE_BASED}],
#             history=None
#         )
#     else:
#         response_image = "N/A (No Image)"

#     # 3) cross-modal
#     if image_path and os.path.exists(image_path):
#         response_cross, _ = model.chat(
#             tokenizer,
#             query=[{"image": image_path},
#                    {"text": f"{PROMPT_CROSS_MODAL}\n\n文本内容：\"{news_text}\""}],
#             history=None
#         )
#     else:
#         response_cross = "N/A (No Image)"

#     all_final_results.append({
#         "post_id": news_item["post_id"],
#         "image_id": news_item["image_id"],
#         "label": news_item["label"],
#         "category": news_item["category"],
#         "text_reasoning": response_text,
#         "image_reasoning": response_image,
#         "cross_modal_reasoning": response_cross
#     })

# # --- 步骤 5: 将所有结果保存到新的CSV文件 (此部分无需修改) ---
# output_filename = "./weibo_21_with_reasoning.csv"
# results_df = pd.DataFrame(all_final_results)
# results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
# print(f"\n全部处理完毕！结果已保存到 {output_filename}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pandas as pd
import os
from tqdm import tqdm

# -------------------------------------------------
# 1. 加载模型
# -------------------------------------------------
MODEL_PATH = "./Qwen-VL"
print("正在加载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("正在加载Qwen-VL模型，这可能需要一些时间...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    fp16=True
).eval()
model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("模型加载完成！")

# -------------------------------------------------
# 2. 定义 Prompts
# -------------------------------------------------
PROMPT_TEXT_BASED  = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_IMAGE_BASED = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_CROSS_MODAL = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图。"

# -------------------------------------------------
# 3. 读取并合并数据集
# -------------------------------------------------
DATA_DIR   = "./Weibo_21"
TRAIN_XLSX = os.path.join(DATA_DIR, "train_datasets.xlsx")
TEST_XLSX  = os.path.join(DATA_DIR, "test_datasets.xlsx")

news_dataset_for_processing = []
all_files_found = True

try:
    train_df = pd.read_excel(TRAIN_XLSX)
    test_df  = pd.read_excel(TEST_XLSX)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"成功加载并合并了 train 和 test XLSX文件。总计 {len(combined_df)} 条数据。")
except FileNotFoundError as e:
    print(f"错误：找不到XLSX文件 - {e}。请确保文件路径正确。")
    combined_df = pd.DataFrame()
    all_files_found = False

if all_files_found:
    for idx, row in combined_df.iterrows():
        content = row['content']
        label   = row['label']
        rel_img = row['image']
        cate    = row['category']

        img_path = None
        img_id   = None
        if pd.notna(rel_img) and isinstance(rel_img, str):
            full_path = os.path.join(DATA_DIR, rel_img)
            if os.path.exists(full_path):
                img_path = full_path
                img_id   = os.path.basename(rel_img)

        news_dataset_for_processing.append({
            "post_id": f"weibo21_{idx}",
            "image_id": img_id,
            "label": label,
            "category": cate,
            "content": content,
            "image_path": img_path
        })

# -------------------------------------------------
# 4. 生成 Reasoning
# -------------------------------------------------
all_final_results = []
print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")

for item in tqdm(news_dataset_for_processing, desc="Generating"):
    text  = item["content"]
    img_p = item["image_path"]

    # 1) 纯文本
    query_txt = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{text}\""
    resp_txt, _ = model.chat(tokenizer, query=query_txt, history=None)

    # 2) 纯图片
    if img_p and os.path.exists(img_p):
        # 关键修复：把 <img> 占位符写进 prompt
        query_img = f"<img>{img_p}</img>\n{PROMPT_IMAGE_BASED}"
        resp_img, _ = model.chat(tokenizer, query=query_img, history=None)
    else:
        resp_img = "N/A (No Image)"

    # 3) 图文跨模态
    if img_p and os.path.exists(img_p):
        query_cross = f"<img>{img_p}</img>\n{PROMPT_CROSS_MODAL}\n\n文本内容：\"{text}\""
        resp_cross, _ = model.chat(tokenizer, query=query_cross, history=None)
    else:
        resp_cross = "N/A (No Image)"

    all_final_results.append({
        "post_id": item["post_id"],
        "image_id": item["image_id"],
        "label": item["label"],
        "category": item["category"],
        "text_reasoning": resp_txt,
        "image_reasoning": resp_img,
        "cross_modal_reasoning": resp_cross
    })

# -------------------------------------------------
# 5. 保存结果
# -------------------------------------------------
output_csv = "./weibo_21_with_reasoning.csv"
pd.DataFrame(all_final_results).to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n全部处理完毕！结果已保存到 {output_csv}")


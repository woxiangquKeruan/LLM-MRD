# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# import pandas as pd
# import os
# from tqdm import tqdm

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

# PROMPT_TEXT_BASED   = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_IMAGE_BASED  = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
# PROMPT_CROSS_MODAL  = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图."

# DATA_DIR = "./data"
# NONRUMOR_IMG_DIR = os.path.join(DATA_DIR, "nonrumor_images")
# RUMOR_IMG_DIR    = os.path.join(DATA_DIR, "rumor_images")
# TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
# VAL_CSV   = os.path.join(DATA_DIR, "val.csv")
# TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# print("正在加载和准备Weibo数据集...")
# train_df = pd.read_csv(TRAIN_CSV)
# val_df   = pd.read_csv(VAL_CSV)
# test_df  = pd.read_csv(TEST_CSV)
# combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
# print(f"成功加载并合并了 train, val, test CSV文件。总计 {len(combined_df)} 条数据。")

# news_dataset_for_processing = []
# for _, row in combined_df.iterrows():
#     image_id = row['image_id']
#     label    = row['label']
#     image_path = None
#     if pd.notna(image_id) and isinstance(image_id, str) and image_id.lower() != 'null':
#         image_path = os.path.join(NONRUMOR_IMG_DIR if label == 1 else RUMOR_IMG_DIR, image_id)
#     news_dataset_for_processing.append({
#         "post_id": row['post_id'],
#         "image_id": image_id,
#         "label": label,
#         "category": row['category'],
#         "content": row['content'],
#         "image_path": image_path
#     })

# all_final_results = []
# print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")

# for news_item in tqdm(news_dataset_for_processing, desc="Generating Reasoning"):
#     news_text  = news_item["content"]
#     image_path = news_item["image_path"]

#     # ------------------- 1. text-only -------------------
#     query_text = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{news_text}\""
#     response_text, _ = model.chat(tokenizer, query=query_text, history=None)

#     # ------------------- 2. image-only -------------------
#     if image_path and os.path.exists(image_path):
#         query_image = [{"image": image_path}, {"text": PROMPT_IMAGE_BASED}]
#         response_image, _ = model.chat(tokenizer, query=query_image, history=None)
#     else:
#         response_image = "N/A (No Image)"

#     # ------------------- 3. cross-modal ------------------
#     if image_path and os.path.exists(image_path):
#         query_cross = [{"image": image_path},
#                        {"text": f"{PROMPT_CROSS_MODAL}\n\n文本内容：\"{news_text}\""}]
#         response_cross, _ = model.chat(tokenizer, query=query_cross, history=None)
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

# output_filename = "./weibo_dataset_with_reasoning.csv"
# results_df = pd.DataFrame(all_final_results)
# results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
# print(f"\n全部处理完毕！结果已保存到 {output_filename}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import pandas as pd
import os
from tqdm import tqdm

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

PROMPT_TEXT_BASED   = "分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_IMAGE_BASED  = "检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图。"
PROMPT_CROSS_MODAL  = "对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图."

DATA_DIR          = "./data"
NONRUMOR_IMG_DIR  = os.path.join(DATA_DIR, "nonrumor_images")
RUMOR_IMG_DIR     = os.path.join(DATA_DIR, "rumor_images")
TRAIN_CSV         = os.path.join(DATA_DIR, "train.csv")
TEST_CSV          = os.path.join(DATA_DIR, "test.csv")

print("正在加载数据集（仅 train + test）...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"已合并 train 和 test，共 {len(combined_df)} 条数据。")

# 构建待处理列表
news_dataset_for_processing = []
for _, row in combined_df.iterrows():
    image_id = row['image_id']
    label    = row['label']
    image_path = None
    if pd.notna(image_id) and isinstance(image_id, str) and image_id.lower() != 'null':
        image_path = os.path.join(NONRUMOR_IMG_DIR if label == 1 else RUMOR_IMG_DIR, image_id)
    news_dataset_for_processing.append({
        "post_id": row['post_id'],
        "image_id": image_id,
        "label": label,
        "category": row['category'],
        "content": row['content'],
        "image_path": image_path
    })

all_final_results = []
print(f"开始生成Reasoning，共 {len(news_dataset_for_processing)} 条数据...")
for news_item in tqdm(news_dataset_for_processing, desc="Generating"):
    news_text  = news_item["content"]
    image_path = news_item["image_path"]

    # 1) text-only
    query_text = f"{PROMPT_TEXT_BASED}\n\n文本内容：\"{news_text}\""
    response_text, _ = model.chat(tokenizer, query=query_text, history=None)

    # 2) image-only
    if image_path and os.path.exists(image_path):
        response_image, _ = model.chat(
            tokenizer,
            query=[{"image": image_path}, {"text": PROMPT_IMAGE_BASED}],
            history=None
        )
    else:
        response_image = "N/A (No Image)"

    # 3) cross-modal
    if image_path and os.path.exists(image_path):
        response_cross, _ = model.chat(
            tokenizer,
            query=[{"image": image_path},
                   {"text": f"{PROMPT_CROSS_MODAL}\n\n文本内容：\"{news_text}\""}],
            history=None
        )
    else:
        response_cross = "N/A (No Image)"

    all_final_results.append({
        "post_id": news_item["post_id"],
        "image_id": news_item["image_id"],
        "label": news_item["label"],
        "category": news_item["category"],
        "text_reasoning": response_text,
        "image_reasoning": response_image,
        "cross_modal_reasoning": response_cross
    })

output_filename = "./weibo_dataset_with_reasoning.csv"
results_df = pd.DataFrame(all_final_results)
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n全部处理完毕！结果已保存到 {output_filename}")
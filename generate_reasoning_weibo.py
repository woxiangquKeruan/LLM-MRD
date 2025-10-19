import torch, os, re, warnings
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    BitsAndBytesConfig
)

warnings.filterwarnings("ignore", message=".*attention mask.*")

MODEL_PATH = "./Qwen-VL"

# 1. 8-bit 量化
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading Qwen-VL-7B (8-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    # ❗ 先去掉 Flash-Attn：官方未支持
    # attn_implementation="flash_attention_2"
)

# 2. 限制生成长度
model.generation_config = GenerationConfig.from_pretrained(
    MODEL_PATH, trust_remote_code=True, max_new_tokens=256
)

# ---------- 3. 数据 ----------
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
NONRUMOR_IMG_DIR = os.path.join(DATA_DIR, "nonrumor_images")
RUMOR_IMG_DIR    = os.path.join(DATA_DIR, "rumor_images")

df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
print(f"Total samples: {len(df)}")

samples = []
for _, r in df.iterrows():
    img_id = str(r["image_id"]) if pd.notna(r["image_id"]) else "null"
    img_path = None
    if img_id.lower() != "null":
        img_path = os.path.join(NONRUMOR_IMG_DIR if r["label"] == 1 else RUMOR_IMG_DIR, img_id)
    samples.append({
        "post_id": r["post_id"],
        "image_id": img_id,
        "label": r["label"],
        "category": r["category"],
        "content": r["content"],
        "image_path": img_path,
    })

# ---------- 4. 合并 prompt ----------
PROMPT_ALL = (
    "请一次性完成下面三个子任务，并以换行+===+换行分隔三段输出：\n"
    "1) 分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图？\n"
    "2) 检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图？\n"
    "3) 对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图？\n\n"
    "文本：{content}"
)

results = []
for item in tqdm(samples, desc="Generating"):
    content, img_path = item["content"], item["image_path"]
    prompt = PROMPT_ALL.format(content=content)

    if img_path and os.path.exists(img_path):
        query = [{"image": img_path}, {"text": prompt}]
    else:
        query = prompt  # 无图

    resp, _ = model.chat(tokenizer, query=query, history=None)

    # 切三段
    parts = [p.strip() for p in re.split(r"\n===\n", resp) if p.strip()]
    while len(parts) < 3:
        parts.append("")
    txt_r, img_r, cross_r = parts

    results.append({
        **item,
        "text_reasoning": txt_r,
        "image_reasoning": img_r,
        "cross_modal_reasoning": cross_r,
    })

# ---------- 5. 保存 ----------
out_csv = "./weibo_with_reasoning.csv"
pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Done → {out_csv}")
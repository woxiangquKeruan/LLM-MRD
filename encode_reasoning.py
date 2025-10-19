import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import numpy as np

# --- 1. 配置模型和文件路径 ---
# (这些路径与您在问题中提供的一致)
class Args:
    # GossipCop
    bert_model_path_gossipcop = './pretrained_model/bert-base-uncased'
    
    # Weibo
    bert_model_path_weibo = './pretrained_model/chinese_roberta_wwm_base_ext_pytorch'

    # CSV 文件路径
    weibo21_csv = './weibo_21_with_reasoning.csv'
    weibo_csv = './weibo_dataset_with_reasoning.csv'
    gossipcop_csv = './gossipcop_with_reasoning.csv'

args = Args()

# 要处理的列
REASONING_COLUMNS = ['text_reasoning', 'image_reasoning', 'cross_modal_reasoning']

# --- 2. 定义编码函数 ---

def get_bert_embedding(texts, model, tokenizer, device, batch_size=32):
    """
    使用BERT模型对文本列表进行编码，返回[CLS] token的向量表示。
    
    Args:
        texts (list): 待编码的文本字符串列表。
        model: 加载好的BERT模型。
        tokenizer: 加载好的BERT分词器。
        device: 'cuda' or 'cpu'。
        batch_size (int): 每批处理的文本数量。

    Returns:
        numpy.ndarray: 所有文本的嵌入向量，形状为 (len(texts), hidden_size)。
    """
    model.eval()
    all_embeddings = []
    
    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches"):
        batch_texts = texts[i:i+batch_size]
        
        # 分词，并移动到指定设备
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # 在无梯度的模式下进行前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 提取[CLS] token的向量表示 (batch_size, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
    return np.vstack(all_embeddings)


def process_dataframe(df_path, model, tokenizer, device):
    """
    加载CSV，对指定列进行编码，并将结果添加为新列后保存。
    """
    if not os.path.exists(df_path):
        print(f"文件未找到: {df_path}，跳过处理。")
        return

    print(f"\n--- 正在处理文件: {df_path} ---")
    df = pd.read_csv(df_path)
    
    for col in REASONING_COLUMNS:
        if col in df.columns:
            print(f"开始编码列: '{col}'...")
            
            # 处理可能存在的NaN值，替换为空字符串
            texts_to_encode = df[col].fillna('').tolist()
            
            # 获取嵌入向量
            embeddings = get_bert_embedding(texts_to_encode, model, tokenizer, device)
            
            # 将numpy数组转换为list，方便存储在CSV中
            new_col_name = f"{col}_embedding"
            df[new_col_name] = list(embeddings)
            
            print(f"编码完成，结果已添加到新列: '{new_col_name}'")
        else:
            print(f"警告: 在 {df_path} 中未找到列 '{col}'。")

    # 保存带有嵌入向量的新CSV文件
    output_path = df_path.replace('.csv', '_encoded.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"处理完成，已保存到: {output_path}")


# --- 3. 主执行逻辑 ---
# 额外导入 BertTokenizer 和 BertModel
from transformers import BertTokenizer, BertModel

if __name__ == "__main__":
    # 设置设备 (如果可用，优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # --- 处理 Weibo 和 Weibo21 数据集 ---
    print("\n加载中文BERT模型 (RoBERTa)...")
    try:
        # --- 核心修改在这里 ---
        # 1. 强制使用 BertTokenizer
        tokenizer_weibo = BertTokenizer.from_pretrained(args.bert_model_path_weibo)
        
        # 2. 强制使用 BertModel 来加载模型，以匹配权重文件
        model_weibo = BertModel.from_pretrained(args.bert_model_path_weibo).to(device)
        # --- 修改结束 ---
        
        # 依次处理两个weibo文件
        process_dataframe(args.weibo_csv, model_weibo, tokenizer_weibo, device)
        process_dataframe(args.weibo21_csv, model_weibo, tokenizer_weibo, device)

    except (OSError, ValueError) as e:
        print(f"错误: 无法在 '{args.bert_model_path_weibo}' 加载中文BERT模型。请检查路径是否正确。")
        print(f"具体错误: {e}")


    # --- 处理 GossipCop 数据集 ---
    print("\n加载英文BERT模型 (bert-base-uncased)...")
    try:
        # 英文模型使用 AutoClass 没有问题，无需修改
        tokenizer_gossipcop = AutoTokenizer.from_pretrained(args.bert_model_path_gossipcop)
        model_gossipcop = AutoModel.from_pretrained(args.bert_model_path_gossipcop).to(device)
        
        # 处理gossipcop文件
        process_dataframe(args.gossipcop_csv, model_gossipcop, tokenizer_gossipcop, device)
    
    except (OSError, ValueError) as e:
        print(f"错误: 无法在 '{args.bert_model_path_gossipcop}' 加载英文BERT模型。请检查路径是否正确。")
        print(f"具体错误: {e}")

    print("\n--- 所有任务已完成！ ---")



# import argparse
# import os
# import torch
# import pandas as pd
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm

# # ---------- 1. argparse ----------
# parser = argparse.ArgumentParser(description="Encode reasoning columns with BERT")
# # GossipCop
# parser.add_argument('--bert_model_path_gossipcop',
#                     default='./pretrained_model/bert-base-uncased',
#                     help='GossipCop 使用的英文 BERT 模型本地路径')
# parser.add_argument('--clip_model_path_gossipcop',
#                     default='./pretrained_model/clip-vit-base-patch16',
#                     help='GossipCop 使用的英文 CLIP 模型本地路径（当前脚本未使用）')
# # Weibo / Weibo21
# parser.add_argument('--bert_model_path_weibo',
#                     default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
#                     help='Weibo/Weibo21 使用的中文 BERT 模型本地路径')
# parser.add_argument('--bert_vocab_file_weibo',
#                     default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
#                     help='Weibo/Weibo21 BERT 词汇表文件（当前脚本未显式使用）')
# # CSV 文件路径
# parser.add_argument('--weibo21_csv',
#                     default='./weibo_21_with_reasoning.csv')
# parser.add_argument('--weibo_csv',
#                     default='./weibo_dataset_with_reasoning.csv')
# parser.add_argument('--gossipcop_csv',
#                     default='./gossipcop_with_reasoning.csv')

# args = parser.parse_args()

# # 需要编码的列
# REASONING_COLUMNS = ['text_reasoning', 'image_reasoning', 'cross_modal_reasoning']

# # ---------- 2. 工具函数 ----------
# def get_bert_embedding(texts, model, tokenizer, device, batch_size=32):
#     """
#     使用BERT模型对文本列表进行编码，返回[CLS] token的向量表示。
#     """
#     model.eval()
#     all_embeddings = []

#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Batches"):
#             batch_texts = texts[i:i+batch_size]
#             inputs = tokenizer(batch_texts,
#                                return_tensors='pt',
#                                padding=True,
#                                truncation=True,
#                                max_length=512).to(device)
#             outputs = model(**inputs)
#             cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#             all_embeddings.append(cls_embeddings)

#     return np.vstack(all_embeddings)


# def process_dataframe(df_path, model, tokenizer, device):
#     """
#     读取CSV，编码指定列，结果写入 *_encoded.csv
#     """
#     if not os.path.exists(df_path):
#         print(f"文件未找到: {df_path}，跳过处理。")
#         return

#     print(f"\n--- 正在处理文件: {df_path} ---")
#     df = pd.read_csv(df_path)

#     for col in REASONING_COLUMNS:
#         if col in df.columns:
#             print(f"开始编码列: '{col}' ...")
#             texts_to_encode = df[col].fillna('').tolist()
#             embeddings = get_bert_embedding(texts_to_encode, model, tokenizer, device)

#             # 将 numpy 数组转成 list 方便存储
#             new_col_name = f"{col}_embedding"
#             df[new_col_name] = list(embeddings)
#             print(f"编码完成，已添加到新列: '{new_col_name}'")
#         else:
#             print(f"警告: 在 {df_path} 中未找到列 '{col}'。")

#     output_path = df_path.replace('.csv', '_encoded.csv')
#     df.to_csv(output_path, index=False, encoding='utf-8-sig')
#     print(f"处理完成，已保存到: {output_path}")

# # ---------- 3. 主流程 ----------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用的设备: {device}")

#     # --- 处理 Weibo / Weibo21 ---
#     print("\n加载中文BERT模型 (RoBERTa)...")
#     try:
#         tokenizer_zh = AutoTokenizer.from_pretrained(args.bert_model_path_weibo)
#         model_zh = AutoModel.from_pretrained(args.bert_model_path_weibo).to(device)
#     except OSError as e:
#         tokenizer_zh = model_zh = None
#         print(f"错误: 无法在 '{args.bert_model_path_weibo}' 加载中文BERT模型。{e}")

#     if tokenizer_zh and model_zh:
#         process_dataframe(args.weibo_csv, model_zh, tokenizer_zh, device)
#         process_dataframe(args.weibo21_csv, model_zh, tokenizer_zh, device)

#     # --- 处理 GossipCop ---
#     print("\n加载英文BERT模型 (bert-base-uncased)...")
#     try:
#         tokenizer_en = AutoTokenizer.from_pretrained(args.bert_model_path_gossipcop)
#         model_en = AutoModel.from_pretrained(args.bert_model_path_gossipcop).to(device)
#     except OSError as e:
#         tokenizer_en = model_en = None
#         print(f"错误: 无法在 '{args.bert_model_path_gossipcop}' 加载英文BERT模型。{e}")

#     if tokenizer_en and model_en:
#         process_dataframe(args.gossipcop_csv, model_en, tokenizer_en, device)

#     print("\n--- 所有任务已完成！ ---")
# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from transformers import BertTokenizer
# import ast # 用于安全地将字符串形式的列表转换为列表

# class WeiboReasoningDataset(Dataset):
#     def __init__(self, csv_path, tokenizer, max_len=197):
#         print(f"Loading data from: {csv_path}")
#         self.df = pd.read_csv(csv_path)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         content = str(row.get('content', '')) # 使用.get保证列不存在时不报错
#         label = torch.tensor(row.get('label', 0), dtype=torch.float32)

#         encoding = self.tokenizer.encode_plus(
#             content,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )

#         try:
#             text_emb_str = row.get('text_reasoning_embedding', '[0.0]*768')
#             image_emb_str = row.get('image_reasoning_embedding', '[0.0]*768')
#             cross_emb_str = row.get('cross_modal_reasoning_embedding', '[0.0]*768')
            
#             text_reasoning_emb = torch.tensor(ast.literal_eval(text_emb_str), dtype=torch.float32)
#             image_reasoning_emb = torch.tensor(ast.literal_eval(image_emb_str), dtype=torch.float32)
#             cross_modal_reasoning_emb = torch.tensor(ast.literal_eval(cross_emb_str), dtype=torch.float32)
#         except (ValueError, SyntaxError) as e:
#             print(f"Error parsing embedding string at index {idx}, using zeros. Error: {e}")
#             text_reasoning_emb = torch.zeros(768, dtype=torch.float32)
#             image_reasoning_emb = torch.zeros(768, dtype=torch.float32)
#             cross_modal_reasoning_emb = torch.zeros(768, dtype=torch.float32)
        
#         dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
#         dummy_clip_image = torch.zeros(3, 224, 224, dtype=torch.float32)
#         dummy_clip_text = torch.zeros(77, dtype=torch.int64)

#         return {
#             'content': encoding['input_ids'].flatten(),
#             'content_masks': encoding['attention_mask'].flatten(),
#             'label': label,
#             'image': dummy_image,
#             'clip_image': dummy_clip_image,
#             'clip_text': dummy_clip_text,
#             'teacher_reasoning_text_emb': text_reasoning_emb,
#             'teacher_reasoning_image_emb': image_reasoning_emb,
#             'teacher_reasoning_cross_emb': cross_modal_reasoning_emb
#         }

# def create_reasoning_dataloader(csv_path, bert_model_path, batch_size, max_len=197, num_workers=4, shuffle=True):
#     tokenizer = BertTokenizer.from_pretrained(bert_model_path)
#     dataset = WeiboReasoningDataset(
#         csv_path=csv_path,
#         tokenizer=tokenizer,
#         max_len=max_len
#     )
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers
#     )
#     return dataloader


import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class WeiboWithReasoningDataset(Dataset):
    """
    扩展数据集类，为原始微博数据集附加教师模型输出的推理嵌入（文本、图像、跨模态）
    需配合原始数据集使用，通过 post_id 匹配教师推理嵌入
    
    Args:
        original_dataset (Dataset): 原始数据集（需包含 'post_id' 字段）
        reasoning_csv_path (str): 教师推理嵌入 CSV 文件路径，需包含以下字段：
            - post_id: 用于匹配原始数据的唯一标识
            - text_reasoning_embedding: 教师模型输出的文本推理嵌入（字符串格式如 "[x1, x2, ..., xn]"）
            - image_reasoning_embedding: 教师模型输出的图像推理嵌入（同上格式）
            - cross_modal_reasoning_embedding: 教师模型输出的跨模态推理嵌入（同上格式）
    """
    def __init__(self, original_dataset, reasoning_csv_path):
        self.original_dataset = original_dataset
        self.reasoning_df = pd.read_csv(reasoning_csv_path)
        
        # 预处理嵌入字段：将字符串格式的嵌入转为 numpy 数组
        embedding_cols = [
            'text_reasoning_embedding', 
            'image_reasoning_embedding', 
            'cross_modal_reasoning_embedding'
        ]
        for col in embedding_cols:
            self.reasoning_df[col] = self.reasoning_df[col].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=', ', dtype=np.float32)
            )
        
        # 构建 post_id 到教师嵌入的映射字典
        self.post_id_to_emb = {
            row['post_id']: (
                row['text_reasoning_embedding'], 
                row['image_reasoning_embedding'], 
                row['cross_modal_reasoning_embedding']
            ) 
            for _, row in self.reasoning_df.iterrows()
        }

    def __len__(self):
        """数据集总样本数，与原始数据集一致"""
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """
        获取单个样本：
        1. 从原始数据集获取基础样本
        2. 通过 post_id 匹配教师推理嵌入
        3. 若匹配失败，用零向量填充（需确保模型训练时兼容）
        """
        # 获取原始样本（需确保样本包含 'post_id' 字段）
        sample = self.original_dataset[idx]
        post_id = sample['post_id']
        
        # 匹配教师嵌入
        if post_id in self.post_id_to_emb:
            text_emb, img_emb, cross_emb = self.post_id_to_emb[post_id]
        else:
            # 若未匹配到，用 768 维零向量填充（需与教师嵌入维度一致）
            emb_dim = 768
            text_emb = np.zeros(emb_dim, dtype=np.float32)
            img_emb = np.zeros(emb_dim, dtype=np.float32)
            cross_emb = np.zeros(emb_dim, dtype=np.float32)
        
        # 将教师嵌入添加到样本中，供训练时使用
        sample['teacher_text_emb'] = text_emb
        sample['teacher_image_emb'] = img_emb
        sample['teacher_cross_emb'] = cross_emb
        
        return sample
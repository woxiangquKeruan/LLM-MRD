# # # run.py (整合后的版本)  原版

# # import os
# # import torch
# # from torch.utils.data import DataLoader
# # from transformers import BertTokenizer, CLIPProcessor # For GossipCop
# # import logging

# # # --- Logger Setup ---
# # logger = logging.getLogger(__name__)
# # if not logger.hasHandlers(): # 防止重复添加处理器
# #     handler = logging.StreamHandler()
# #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# #     handler.setFormatter(formatter)
# #     logger.addHandler(handler)
# #     logger.setLevel(logging.INFO)

# # # --- GossipCop 相关导入 ---
# # FakeNet_dataset = None # Initialize to None
# # try:
# #     from FakeNet_dataset import FakeNet_dataset #
# #     logger.info("FakeNet_dataset 导入成功。")
# # except ImportError as e:
# #     logger.error(f"无法导入 FakeNet_dataset: {e}. 请确保 FakeNet_dataset.py 在当前工作目录或 Python 路径下。")
# #     # Not raising error here; will be checked if self.dataset == "gossipcop"

# # # --- Weibo/Weibo21 相关导入 ---
# # WeiboDataLoaderClass = None
# # Weibo21DataLoaderClass = None
# # try:
# #     from utils.clip_dataloader import bert_data as WeiboDataLoaderClass # For weibo dataset
# #     logger.info("utils.clip_dataloader.bert_data (for Weibo) 导入成功。")
# # except ImportError:
# #     logger.warning("无法从 utils.clip_dataloader 导入 Weibo 数据加载器。")

# # try:
# #     from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass # For weibo21 dataset
# #     logger.info("utils.weibo21_clip_dataloader.bert_data (for Weibo21) 导入成功。")
# # except ImportError:
# #     logger.warning("无法从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器。")


# # # --- 根据模型名称选择合适的Trainer ---
# # DOMAINTrainerGossipCop = None
# # DOMAINTrainerWeibo = None

# # try:
# #     from model.domain_gossipcop import Trainer as DOMAINTrainerGossipCop
# #     logger.info("model.domain_gossipcop.Trainer 导入成功。")
# # except ImportError:
# #     logger.warning("无法导入 model.domain_gossipcop.Trainer。如果未使用GossipCop数据集则忽略。")


# # try:
# #     # from model.domain_weibo import Trainer as DOMAINTrainerWeibo
# #     from model.domain_weibo import DOMAINTrainerWeibo 
# #     logger.info("model.domain_weibo.Trainer 导入成功。")
# # except ImportError:
# #     logger.warning("无法导入 model.domain_weibo.Trainer。如果未使用Weibo/Weibo21数据集则忽略。")


# # # --- 全局 Tokenizer/Processor 实例 (仅GossipCop需要全局预加载并传递) ---
# # bert_tokenizer_gossipcop = None
# # clip_processor_gossipcop = None

# # # --- GossipCop 使用的 Collate Function ---
# # def collate_fn_gossipcop(batch): #
# #     batch_before_filtering_len = len(batch)
# #     batch = [item for item in batch if item is not None]
# #     if batch_before_filtering_len > 0 and not batch:
# #         logger.warning(f"Collate (gossipcop): 所有 {batch_before_filtering_len} 个项目在此批次中均为 None，返回 None。这可能表示数据加载问题。")
# #         return None
# #     if not batch:
# #         logger.warning("Collate (gossipcop): 接收到空批次或所有项目过滤后批次为空，返回 None。")
# #         return None

# #     keys = batch[0].keys()
# #     collated_batch = {}
# #     for key in keys:
# #         values = [item[key] for item in batch] #
# #         if all(isinstance(v, torch.Tensor) for v in values):
# #             try:
# #                 collated_batch[key] = torch.stack(values, dim=0)
# #             except RuntimeError as e:
# #                 logger.error(f"Collate (gossipcop) 无法为键 '{key}' 堆叠张量 (可能是尺寸不匹配): {e}")
# #                 for i, v_item in enumerate(values): #
# #                     logger.error(f"  Item {i} ('{key}') shape: {v_item.shape}")
# #                 logger.error(f"  批次中导致错误的张量列表 ('{key}'): {[v.shape for v in values]}")
# #                 return None
# #             except Exception as e:
# #                 logger.error(f"Collate (gossipcop) 为键 '{key}' 堆叠张量时发生未知错误: {e}")
# #                 return None
# #         else:
# #             collated_batch[key] = values #
# #     return collated_batch


# # class Run():
# #     def __init__(self, config):
# #         self.config = config
# #         self.use_cuda = config.get('use_cuda', torch.cuda.is_available())
# #         self.dataset = config['dataset']
# #         self.model_name = config['model_name']
# #         self.lr = config['lr']
# #         self.batchsize = config['batchsize']
# #         self.emb_dim = config['emb_dim']
# #         self.max_len = config['max_len'] # BERT max_len
# #         self.num_workers = config.get('num_workers', 0)
# #         self.early_stop = config['early_stop']
# #         self.epoch = config['epoch']
# #         self.save_param_dir = config['save_param_dir']

# #         logger.info(f"Run initialized with config: {config}")
# #         logger.info(f"PyTorch version: {torch.__version__}")
# #         logger.info(f"CUDA available: {torch.cuda.is_available()}, Using CUDA: {self.use_cuda}")

# #         if self.dataset == "gossipcop":
# #             self.root_path = self.config.get('gossipcop_data_dir')
# #             self.bert_model_path = self.config.get('bert_model_path_gossipcop')
# #             self.clip_model_path = self.config.get('clip_model_path_gossipcop')
# #             self.category_dict = {"gossip": 0}
# #             self.early_stop_metric_key = self.config.get('early_stop_metric', 'F1')

# #             if not self.root_path or not self.bert_model_path or not self.clip_model_path:
# #                 logger.error("GossipCop: 'gossipcop_data_dir', 'bert_model_path_gossipcop', or 'clip_model_path_gossipcop' 未在配置中提供。")
# #                 raise ValueError("GossipCop 配置路径缺失。")

# #             global bert_tokenizer_gossipcop, clip_processor_gossipcop
# #             try:
# #                 logger.info(f"尝试从本地路径加载 BERT tokenizer (GossipCop): {self.bert_model_path}")
# #                 bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
# #             except Exception as e:
# #                 logger.error(f"从本地路径加载 BERT tokenizer (GossipCop) 失败: {e}. Path: {self.bert_model_path}")
# #             try:
# #                 logger.info(f"尝试从本地路径加载 CLIP processor (GossipCop): {self.clip_model_path}")
# #                 clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
# #             except Exception as e:
# #                 logger.error(f"从本地路径加载 CLIP processor (GossipCop) 失败: {e}. Path: {self.clip_model_path}")

# #             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
# #                 logger.warning("GossipCop 所需的 BERT tokenizer 或 CLIP processor 未能成功加载。将在 get_dataloader 中进一步检查。")

# #         elif self.dataset == "weibo":
# #             self.root_path = self.config.get('weibo_data_dir')
# #             if not self.root_path:
# #                 logger.error("Weibo: 'weibo_data_dir' 未在配置中提供。")
# #                 raise ValueError("Weibo 数据根目录 'weibo_data_dir' 配置缺失。")
# #             self.train_path = os.path.join(self.root_path, 'train_origin.csv')
# #             self.val_path = os.path.join(self.root_path, 'val_origin.csv')
# #             self.test_path = os.path.join(self.root_path, 'test_origin.csv')
# #             self.category_dict = { "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4, "国际": 5, "教育": 6, "娱乐": 7, "社会": 8 }
# #             self.bert_model_path = self.config.get('bert_model_path_weibo')
# #             self.vocab_file = self.config.get('vocab_file')
# #             self.early_stop_metric_key = 'metric'
# #             if not self.bert_model_path or not self.vocab_file:
# #                 logger.error("Weibo: 'bert_model_path_weibo' or 'vocab_file' 未在配置中提供。")
# #                 raise ValueError("Weibo BERT 模型路径或词汇表文件配置缺失。")

# #         elif self.dataset == "weibo21":
# #             self.root_path = self.config.get('weibo21_data_dir')
# #             if not self.root_path:
# #                 logger.error("Weibo21: 'weibo21_data_dir' 未在配置中提供。")
# #                 raise ValueError("Weibo21 数据根目录 'weibo21_data_dir' 配置缺失。")
# #             self.train_path = os.path.join(self.root_path, 'train_datasets.xlsx') # 假设xlsx
# #             self.val_path = os.path.join(self.root_path, 'val_datasets.xlsx')
# #             self.test_path = os.path.join(self.root_path, 'test_datasets.xlsx')
# #             self.category_dict = { "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8 }
# #             self.bert_model_path = self.config.get('bert_model_path_weibo')
# #             self.vocab_file = self.config.get('vocab_file')
# #             self.early_stop_metric_key = 'metric'
# #             if not self.bert_model_path or not self.vocab_file:
# #                 logger.error("Weibo21: 'bert_model_path_weibo' (for weibo21) or 'vocab_file' 未在配置中提供。")
# #                 raise ValueError("Weibo21 BERT 模型路径或词汇表文件配置缺失。")
# #         else:
# #             logger.error(f"未知数据集: {self.dataset}. 请检查配置。")
# #             raise ValueError(f"未知数据集: {self.dataset}")

# #         logger.info(f"数据集: {self.dataset}, 根目录: {self.root_path}")
# #         if self.dataset in ["weibo", "weibo21"]:
# #              logger.info(f"  训练数据原始文件路径 (预期): {getattr(self, 'train_path', 'N/A')}")
# #              logger.info(f"  验证数据原始文件路径 (预期): {getattr(self, 'val_path', 'N/A')}")
# #              logger.info(f"  测试数据原始文件路径 (预期): {getattr(self, 'test_path', 'N/A')}")
# #              logger.info(f"  BERT 模型路径: {self.bert_model_path}")
# #              logger.info(f"  词汇表文件: {self.vocab_file}")
# #         elif self.dataset == "gossipcop":
# #              logger.info(f"  BERT 模型路径: {self.bert_model_path}")
# #              logger.info(f"  CLIP 模型路径: {self.clip_model_path}")


# #     def get_dataloader(self):
# #         train_loader, val_loader, test_loader = None, None, None
# #         logger.info(f"开始为数据集 '{self.dataset}' 获取 Dataloaders...")

# #         if self.dataset == "gossipcop":
# #             logger.info("为 GossipCop 加载 FakeNet_dataset 和 collate_fn_gossipcop")
# #             if FakeNet_dataset is None: # Check if FakeNet_dataset was imported
# #                 logger.error("FakeNet_dataset 未成功导入，无法为GossipCop加载数据。请检查之前的导入错误。")
# #                 raise ImportError("FakeNet_dataset 未成功导入，无法为GossipCop加载数据。")
# #             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
# #                 logger.error("GossipCop 数据集所需的全局 BERT tokenizer 或 CLIP processor 未加载。请检查 __init__ 中的加载日志。")
# #                 raise RuntimeError("GossipCop 数据集所需的全局 tokenizer/processor 未加载。")

# #             img_size = 224
# #             clip_max_len = 77
            
# #             logger.info(f"GossipCop root_path: {self.root_path}")
# #             logger.info("Initializing GossipCop train_dataset...")
# #             train_dataset = FakeNet_dataset(
# #                 root_path=self.root_path,
# #                 bert_tokenizer_instance=bert_tokenizer_gossipcop,
# #                 clip_processor_instance=clip_processor_gossipcop,
# #                 dataset_name="gossip",
# #                 image_size=img_size,
# #                 is_train=True,
# #                 bert_max_len=self.max_len,
# #                 clip_max_len=clip_max_len
# #             )
# #             if len(train_dataset) == 0:
# #                 logger.warning("GossipCop 训练数据集加载后长度为 0。请检查数据源和 FakeNet_dataset 实现。")
# #             train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True,
# #                                       collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
# #                                       drop_last=True, pin_memory=self.use_cuda)

# #             logger.info("Initializing GossipCop validation_dataset...")
# #             val_dataset = FakeNet_dataset(
# #                 root_path=self.root_path,
# #                 bert_tokenizer_instance=bert_tokenizer_gossipcop,
# #                 clip_processor_instance=clip_processor_gossipcop,
# #                 dataset_name="gossip",
# #                 image_size=img_size,
# #                 is_train=False, # Test mode
# #                 bert_max_len=self.max_len,
# #                 clip_max_len=clip_max_len
# #             )
# #             if len(val_dataset) == 0:
# #                 logger.warning("GossipCop 验证/测试数据集加载后长度为 0。请检查数据源和 FakeNet_dataset 实现。")
# #             val_loader = DataLoader(val_dataset, batch_size=self.batchsize, shuffle=False,
# #                                     collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
# #                                     drop_last=False, pin_memory=self.use_cuda)
# #             test_loader = val_loader

# #         elif self.dataset == "weibo":
# #             # Ensuring "weibo" dataset uses WeiboDataLoaderClass from utils.clip_dataloader
# #             if WeiboDataLoaderClass is None:
# #                 logger.error("Weibo 数据加载器 (WeiboDataLoaderClass from utils.clip_dataloader) 未成功导入。")
# #                 raise ImportError("Weibo 数据加载器 (WeiboDataLoaderClass from utils.clip_dataloader) 未成功导入。")
            
# #             logger.info(f"为 Weibo 数据集加载 utils.clip_dataloader.bert_data (root: {self.root_path})")
# #             loader = WeiboDataLoaderClass(
# #                 max_len=self.max_len,
# #                 batch_size=self.batchsize,
# #                 vocab_file=self.vocab_file,
# #                 category_dict=self.category_dict,
# #                 num_workers=self.num_workers,
# #                 clip_model_name="ViT-B-16", # As per user's provided script
# #                 clip_download_root='./'      # As per user's provided script
# #             )
            
# #             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl') 
# #             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
# #             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
# #             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
# #             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
# #             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

# #             logger.info(f"Weibo (using utils.clip_dataloader): 调用 load_data for train. Raw: {self.train_path}, PKL: {train_pkl_path}, Clip PKL: {train_clip_pkl_path}")
# #             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True) 
# #             if train_loader is None: logger.warning("Weibo (using utils.clip_dataloader) train_loader is None after load_data call.")

# #             logger.info(f"Weibo (using utils.clip_dataloader): 调用 load_data for validation. Raw: {self.val_path}, PKL: {val_pkl_path}, Clip PKL: {val_clip_pkl_path}")
# #             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
# #             if val_loader is None: logger.warning("Weibo (using utils.clip_dataloader) val_loader is None after load_data call.")

# #             logger.info(f"Weibo (using utils.clip_dataloader): 调用 load_data for test. Raw: {self.test_path}, PKL: {test_pkl_path}, Clip PKL: {test_clip_pkl_path}")
# #             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)
# #             if test_loader is None: logger.warning("Weibo (using utils.clip_dataloader) test_loader is None after load_data call.")

# #         elif self.dataset == "weibo21":
# #             if Weibo21DataLoaderClass is None:
# #                 logger.error("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
# #                 raise ImportError("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
            
# #             logger.info(f"为 Weibo21 数据集加载 utils.weibo21_clip_dataloader.bert_data (root: {self.root_path})") 
# #             loader = Weibo21DataLoaderClass(
# #                 max_len=self.max_len, 
# #                 batch_size=self.batchsize,
# #                 vocab_file=self.vocab_file, 
# #                 category_dict=self.category_dict,
# #                 num_workers=self.num_workers
# #                 # clip_model_name and clip_download_root are commented out in user's provided script for weibo21
# #                 # If needed, they can be added here from self.config similar to Weibo or GossipCop
# #             )
# #             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
# #             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl') 
# #             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
# #             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
# #             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
# #             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

# #             logger.info(f"Weibo21: 调用 load_data for train. Raw: {self.train_path}, PKL: {train_pkl_path}, Clip PKL: {train_clip_pkl_path}")
# #             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
# #             if train_loader is None: logger.warning("Weibo21 train_loader is None after load_data call.")

# #             logger.info(f"Weibo21: 调用 load_data for validation. Raw: {self.val_path}, PKL: {val_pkl_path}, Clip PKL: {val_clip_pkl_path}")
# #             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
# #             if val_loader is None: logger.warning("Weibo21 val_loader is None after load_data call.")

# #             logger.info(f"Weibo21: 调用 load_data for test. Raw: {self.test_path}, PKL: {test_pkl_path}, Clip PKL: {test_clip_pkl_path}")
# #             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False) 
# #             if test_loader is None: logger.warning("Weibo21 test_loader is None after load_data call.")
# #         else:
# #             logger.error(f"数据集 '{self.dataset}' 的 Dataloader 逻辑未定义。")
# #             raise ValueError(f"数据集 {self.dataset} 的 Dataloader 逻辑未定义。")

# #         error_messages = []
# #         if train_loader is None:
# #             error_messages.append("train_loader is None")
# #         if val_loader is None:
# #             error_messages.append("val_loader is None")
# #         if test_loader is None:
# #             error_messages.append("test_loader is None")

# #         if error_messages:
# #             full_error_message = f"Dataloader 初始化失败: {', '.join(error_messages)} for dataset '{self.dataset}'. 请检查上述日志以获取详细信息，特别是文件路径和外部 dataloader 的行为。"
# #             logger.error(full_error_message)
# #             raise RuntimeError(full_error_message)

# #         logger.info(f"Dataloaders for '{self.dataset}' created successfully.")
# #         logger.info(f"  训练加载器批次数: {len(train_loader) if hasattr(train_loader, '__len__') else 'N/A (可能是None或无长度)'}") 
# #         logger.info(f"  验证加载器批次数: {len(val_loader) if hasattr(val_loader, '__len__') else 'N/A (可能是None或无长度)'}")
# #         logger.info(f"  测试加载器批次数: {len(test_loader) if hasattr(test_loader, '__len__') else 'N/A (可能是None或无长度)'}")
# #         return train_loader, val_loader, test_loader

# #     def main(self):
# #         logger.info(f"开始运行主程序: 数据集 '{self.dataset}', 模型配置名 '{self.model_name}'")
# #         try:
# #             train_loader, val_loader, test_loader = self.get_dataloader()
# #         except (RuntimeError, ImportError, ValueError) as e: # Catching errors from get_dataloader
# #             logger.error(f"在 get_dataloader 期间发生错误: {e}")
# #             logger.error("无法继续进行训练。请检查配置和数据。")
# #             return # Exit if dataloaders fail

# #         trainer = None
# #         model_save_path = os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
# #         logger.info(f"模型参数将保存到: {model_save_path}")
# #         os.makedirs(model_save_path, exist_ok=True) 

# #         if self.dataset == "gossipcop":
# #             if DOMAINTrainerGossipCop is None: 
# #                 logger.error(f"DOMAINTrainerGossipCop 未加载，无法训练 {self.dataset}。")
# #                 raise ImportError(f"DOMAINTrainerGossipCop 未加载，无法训练 {self.dataset}。")
# #             if self.model_name != 'domain_gossipcop':
# #                 logger.warning(f"GossipCop 通常使用 'domain_gossipcop' 模型, 当前为 '{self.model_name}'.")
            
# #             if 'model_params' not in self.config or 'mlp' not in self.config['model_params']:
# #                 logger.error("GossipCop trainer: 'model_params.mlp' 未在配置中找到。")
# #                 raise ValueError("'model_params.mlp' (包含 'dims' 和 'dropout') 配置缺失。")

# #             trainer = DOMAINTrainerGossipCop(
# #                 emb_dim=self.emb_dim,
# #                 mlp_dims=self.config['model_params']['mlp']['dims'], 
# #                 bert_path_or_name=self.bert_model_path,
# #                 clip_path_or_name=self.clip_model_path,
# #                 use_cuda=self.use_cuda,
# #                 lr=self.lr,
# #                 train_loader=train_loader,
# #                 val_loader=val_loader, 
# #                 test_loader=test_loader,
# #                 dropout=self.config['model_params']['mlp']['dropout'],
# #                 weight_decay=self.config.get('weight_decay', 5e-5),
# #                 category_dict=self.category_dict,
# #                 early_stop=self.early_stop,
# #                 metric_key_for_early_stop=self.early_stop_metric_key,
# #                 epoches=self.epoch, 
# #                 save_param_dir=model_save_path
# #             )
# #         elif self.dataset == "weibo" or self.dataset == "weibo21": 
# #             if DOMAINTrainerWeibo is None:
# #                 logger.error(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
# #                 raise ImportError(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
# #             if self.model_name != 'domain_weibo': 
# #                 logger.warning(f"{self.dataset} 通常使用 'domain_weibo' 模型, 当前为 '{self.model_name}'.")

# #             if 'model_params' not in self.config or 'mlp' not in self.config['model_params']:
# #                 logger.error(f"{self.dataset} trainer: 'model_params.mlp' 未在配置中找到。")
# #                 raise ValueError("'model_params.mlp' (包含 'dims' 和 'dropout') 配置缺失。")

# #             trainer = DOMAINTrainerWeibo(
# #                 emb_dim=self.emb_dim,
# #                 mlp_dims=self.config['model_params']['mlp']['dims'],
# #                 bert=self.bert_model_path, 
# #                 use_cuda=self.use_cuda, 
# #                 lr=self.lr,
# #                 dropout=self.config['model_params']['mlp']['dropout'],
# #                 train_loader=train_loader,
# #                 val_loader=val_loader,
# #                 test_loader=test_loader,
# #                 category_dict=self.category_dict, 
# #                 weight_decay=self.config.get('weight_decay', 5e-5),
# #                 save_param_dir=model_save_path,
# #                 early_stop=self.early_stop,
# #                 epoches=self.epoch
# #             )
# #         else:
# #             # This block is now correctly indented
# #             logger.error(f"数据集 '{self.dataset}' 没有匹配的 Trainer 初始化逻辑。") 
# #             raise ValueError(f"数据集 {self.dataset} 没有匹配的 Trainer 初始化逻辑。") 

# #         if trainer:
# #             logger.info(f"Trainer for '{self.dataset}' 初始化完成。开始训练...")
# #             try:
# #                 trainer.train()
# #                 logger.info(f"训练过程已成功完成 for {self.dataset} - {self.model_name}.")
# #             except Exception as train_e:
# #                 # This block is now correctly indented
# #                 logger.exception(f"训练过程中发生严重错误 for {self.dataset} - {self.model_name}: {train_e}") 
# #         else:
# #             # This block is now correctly indented
# #             logger.error(f"Trainer for '{self.dataset}' 未能初始化。训练无法开始。")

# # if __name__ == '__main__':
# #     logger.info("run.py executed directly (example). 通常此文件作为模块导入。")
# #     logger.info("要运行，请从 main.py 调用此类并提供有效的配置。")
# #     # Example configuration (replace with your actual config loading from main.py)
# #     # Ensure pandas is imported if you create a direct test here that uses the conceptual dataloader example.
# #     # import pandas 
# #     example_config_weibo = {
# #         'use_cuda': torch.cuda.is_available(),
# #         'dataset': 'weibo', 
# #         'model_name': 'domain_weibo', 
# #         'lr': 1e-4,
# #         'batchsize': 32,
# #         'emb_dim': 768, 
# #         'max_len': 150, 
# #         'num_workers': 2,
# #         'early_stop': 10,
# #         'epoch': 50,
# #         'save_param_dir': './saved_models',
# #         # --- Paths for 'weibo' dataset ---
# #         'weibo_data_dir': './data/weibo_example', #  <-- !!! IMPORTANT: ADJUST THIS PATH !!!
# #         'bert_model_path_weibo': './model_weights/bert-base-chinese', #  <-- !!! IMPORTANT: ADJUST THIS PATH !!!
# #         'vocab_file': './model_weights/bert-base-chinese/vocab.txt', #  <-- !!! IMPORTANT: ADJUST THIS PATH !!!
# #         # --- Common model params ---
# #         'model_params': {
# #             'mlp': {
# #                 'dims': [768, 256, 2], 
# #                 'dropout': 0.3
# #             }
# #         },
# #         'weight_decay': 1e-5,
# #         # --- Optional CLIP params (if your WeiboDataLoaderClass uses them from config) ---
# #         # 'cn_clip_model_name': "ViT-B-16", 
# #         # 'cn_clip_download_root': './model_weights/clip_cn/'
# #     }
# #     # To run this example directly, you would:
# #     # 1. Ensure all paths in example_config_weibo are correct and files exist.
# #     # 2. Ensure `utils.clip_dataloader.py` and `model.domain_weibo.py` are available and correct.
# #     # 3. Uncomment the following:
# #     # logger.info(f"使用示例配置: {example_config_weibo['dataset']}")
# #     # try:
# #     #     if not os.path.exists(example_config_weibo["weibo_data_dir"]):
# #     #         logger.warning(f"示例数据目录不存在: {example_config_weibo['weibo_data_dir']}. Dataloader 可能失败。")
# #     #         # Create dummy files for testing if necessary, e.g.:
# #     #         # os.makedirs(example_config_weibo["weibo_data_dir"], exist_ok=True)
# #     #         # with open(os.path.join(example_config_weibo["weibo_data_dir"], "train_origin.csv"), "w") as f:
# #     #         #     f.write("text_column,label_column\ndummy text,0\n") # Adjust dummy csv content
# #     #         # Similar for val_origin.csv, test_origin.csv

# #     #     runner = Run(example_config_weibo)
# #     #     runner.main()
# #     # except ValueError as ve:
# #     #     logger.error(f"配置错误或值错误: {ve}")
# #     # except ImportError as ie:
# #     #     logger.error(f"导入错误: {ie}")
# #     # except Exception as e:
# #     #     logger.exception(f"运行示例时发生意外错误: {e}")






    
# # run.py (修改后的导入部分)
# # run.py (整合后的版本)

# # import os
# # import torch
# # from torch.utils.data import DataLoader
# # from transformers import BertTokenizer, CLIPProcessor # For GossipCop
# # import logging

# # logger = logging.getLogger(__name__)
# # if not logger.hasHandlers(): # 防止重复添加处理器
# #     handler = logging.StreamHandler()
# #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# #     handler.setFormatter(formatter)
# #     logger.addHandler(handler)
# #     logger.setLevel(logging.INFO)

# # # --- GossipCop 相关导入 ---
# # try:
# #     from FakeNet_dataset import FakeNet_dataset #
# #     logger.info("FakeNet_dataset 导入成功。")
# # except ImportError as e:
# #     logger.error(f"无法导入 FakeNet_dataset: {e}. 请确保 FakeNet_dataset.py 在当前工作目录或 Python 路径下。")
# #     FakeNet_dataset = None # 设置为None，以便后续检查

# # # --- Weibo/Weibo21 相关导入 ---
# # WeiboDataLoaderClass = None
# # Weibo21DataLoaderClass = None

# # # DataLoader for "weibo" dataset
# # try:
# #     from utils.weibo_clip_dataloader import bert_data as WeiboDataLoaderForWeibo
# #     logger.info("成功从 utils.weibo_clip_dataloader 导入 Weibo 数据加载器 (WeiboDataLoaderForWeibo)。")
# #     WeiboDataLoaderClass = WeiboDataLoaderForWeibo
# # except ImportError as e:
# #     logger.error(f"错误：无法从 utils.weibo_clip_dataloader 导入 Weibo 数据加载器。错误: {e}")
# #     # 此处不raise，让后续的 if WeiboDataLoaderClass is None: 处理

# # # DataLoader for "weibo21" dataset
# # try:
# #     from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderForWeibo21 # [cite: 39]
# #     logger.info("成功从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器 (Weibo21DataLoaderForWeibo21)。")
# #     Weibo21DataLoaderClass = Weibo21DataLoaderForWeibo21
# # except ImportError as e:
# #     logger.error(f"错误：无法从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器。错误: {e}")
# #     # 此处不raise，让后续的 if Weibo21DataLoaderClass is None: 处理


# # # --- 根据模型名称选择合适的Trainer ---
# # DOMAINTrainerGossipCop = None
# # DOMAINTrainerWeibo = None

# # try:
# #     from model.domain_gossipcop import Trainer as DOMAINTrainerGossipCop
# #     logger.info("model.domain_gossipcop.Trainer 导入成功。")
# # except ImportError as e:
# #     logger.warning(f"无法导入 model.domain_gossipcop.Trainer。如果未使用GossipCop数据集则忽略。错误: {e}")

# # try:
# #     from model.domain_weibo import Trainer as DOMAINTrainerWeibo
# #     logger.info("model.domain_weibo.Trainer 导入成功。")
# # except ImportError as e:
# #     logger.warning(f"无法导入 model.domain_weibo.Trainer。如果未使用Weibo/Weibo21数据集则忽略。错误: {e}")


# # # --- 全局 Tokenizer/Processor 实例 (仅GossipCop需要全局预加载并传递) ---
# # bert_tokenizer_gossipcop = None # [cite: 39, 40]
# # clip_processor_gossipcop = None

# # # --- GossipCop 使用的 Collate Function ---
# # def collate_fn_gossipcop(batch): #
# #     batch = [item for item in batch if item is not None]
# #     if not batch:
# #         return None
# #     keys = batch[0].keys()
# #     collated_batch = {}
# #     for key in keys:
# #         values = [item[key] for item in batch] #
# #         if all(isinstance(v, torch.Tensor) for v in values):
# #             try:
# #                 collated_batch[key] = torch.stack(values, dim=0) # [cite: 41]
# #             except RuntimeError as e:
# #                 logger.error(f"Collate (gossipcop) 无法为键 '{key}' 堆叠张量 (可能是尺寸不匹配): {e}")
# #                 for i, v_item in enumerate(values): #
# #                     logger.error(f"  Item {i} shape: {v_item.shape}") # [cite: 42]
# #                 return None
# #             except Exception as e:
# #                  logger.error(f"Collate (gossipcop) 为键 '{key}' 堆叠张量时发生未知错误: {e}")
# #                  return None
# #         else:
# #             collated_batch[key] = values # [cite: 43]
# #     return collated_batch


# # class Run():
# #     def __init__(self, config):
# #         self.config = config
# #         self.use_cuda = config['use_cuda']
# #         self.dataset = config['dataset']
# #         self.model_name = config['model_name']
# #         self.lr = config['lr']
# #         self.batchsize = config['batchsize']
# #         self.emb_dim = config['emb_dim']
# #         self.max_len = config['max_len'] # BERT max_len
# #         self.num_workers = config['num_workers'] # [cite: 44]
# #         self.early_stop = config['early_stop']
# #         self.epoch = config['epoch']
# #         self.save_param_dir = config['save_param_dir']

# #         if self.dataset == "gossipcop":
# #             self.root_path = self.config.get('gossipcop_data_dir')
# #             self.bert_model_path = self.config.get('bert_model_path_gossipcop')
# #             self.clip_model_path = self.config.get('clip_model_path_gossipcop')
# #             self.category_dict = {"gossip": 0} # [cite: 45]
# #             self.early_stop_metric_key = self.config.get('early_stop_metric', 'F1')

# #             global bert_tokenizer_gossipcop, clip_processor_gossipcop
# #             try:
# #                 logger.info(f"尝试从本地路径加载 BERT tokenizer (GossipCop): {self.bert_model_path}")
# #                 bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
# #             except Exception as e:
# #                 logger.error(f"从本地路径加载 BERT tokenizer (GossipCop) 失败: {e}") # [cite: 46]
# #             try:
# #                 logger.info(f"尝试从本地路径加载 CLIP processor (GossipCop): {self.clip_model_path}")
# #                 clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
# #             except Exception as e:
# #                 logger.error(f"从本地路径加载 CLIP processor (GossipCop) 失败: {e}") # [cite: 47]

# #             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
# #                  logger.warning("GossipCop 所需的 BERT tokenizer 或 CLIP processor 未能成功加载。")

# #         elif self.dataset == "weibo":
# #             self.root_path = self.config.get('weibo_data_dir')
# #             self.train_path = os.path.join(self.root_path, 'train_origin.csv')
# #             self.val_path = os.path.join(self.root_path, 'val_origin.csv') # [cite: 48]
# #             self.test_path = os.path.join(self.root_path, 'test_origin.csv')
# #             self.category_dict = { "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4, "国际": 5, "教育": 6, "娱乐": 7, "社会": 8 }
# #             self.bert_model_path = self.config.get('bert_model_path_weibo') # [cite: 48]
# #             self.vocab_file = self.config.get('vocab_file') # This should be bert_vocab_file_weibo from main.py config
# #             self.early_stop_metric_key = 'metric'


# #         elif self.dataset == "weibo21": # [cite: 49]
# #             self.root_path = self.config.get('weibo21_data_dir')
# #             self.train_path = os.path.join(self.root_path, 'train_datasets.xlsx') # 假设xlsx, 如果是csv请修改
# #             self.val_path = os.path.join(self.root_path, 'val_datasets.xlsx')
# #             self.test_path = os.path.join(self.root_path, 'test_datasets.xlsx')
# #             self.category_dict = { "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8 }
# #             self.bert_model_path = self.config.get('bert_model_path_weibo') # [cite: 50]
# #             self.vocab_file = self.config.get('vocab_file') # This should be bert_vocab_file_weibo from main.py config
# #             self.early_stop_metric_key = 'metric'
# #         else:
# #             raise ValueError(f"未知数据集: {self.dataset}")

# #         logger.info(f"数据集: {self.dataset}, 根目录: {self.root_path}")


# #     def get_dataloader(self):
# #         train_loader, val_loader, test_loader = None, None, None

# #         if self.dataset == "gossipcop": # [cite: 51]
# #             logger.info("为 GossipCop 加载 FakeNet_dataset 和 collate_fn_gossipcop")
# #             if FakeNet_dataset is None:
# #                 raise ImportError("FakeNet_dataset 未成功导入，无法为GossipCop加载数据。")
# #             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
# #                 raise RuntimeError("GossipCop 数据集所需的全局 tokenizer/processor 未加载。")

# #             img_size = 224 # [cite: 52]
# #             clip_max_len = 77

# #             train_dataset = FakeNet_dataset(
# #                 root_path=self.root_path,
# #                 bert_tokenizer_instance=bert_tokenizer_gossipcop,
# #                 clip_processor_instance=clip_processor_gossipcop,
# #                 dataset_name="gossip",
# #                 image_size=img_size, # [cite: 53]
# #                 is_train=True,
# #                 bert_max_len=self.max_len,
# #                 clip_max_len=clip_max_len
# #             )
# #             train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True,
# #                                       collate_fn=collate_fn_gossipcop, num_workers=self.num_workers, # [cite: 54]
# #                                       drop_last=True, pin_memory=True)

# #             val_dataset = FakeNet_dataset(
# #                 root_path=self.root_path,
# #                 bert_tokenizer_instance=bert_tokenizer_gossipcop, # [cite: 55]
# #                 clip_processor_instance=clip_processor_gossipcop,
# #                 dataset_name="gossip",
# #                 image_size=img_size,
# #                 is_train=False, # Test mode
# #                 bert_max_len=self.max_len,
# #                 clip_max_len=clip_max_len # [cite: 56]
# #             )
# #             val_loader = DataLoader(val_dataset, batch_size=self.batchsize, shuffle=False,
# #                                     collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
# #                                     drop_last=False, pin_memory=True) # [cite: 56, 57]
# #             test_loader = val_loader

# #         elif self.dataset == "weibo":
# #             if WeiboDataLoaderClass is None:
# #                 raise ImportError("Weibo 数据加载器 (WeiboDataLoaderClass from utils.weibo_clip_dataloader) 未成功导入。")
# #             logger.info("为 Weibo 数据集加载 utils.weibo_clip_dataloader.bert_data")
# #             loader = WeiboDataLoaderClass(max_len=self.max_len, # BERT max_len [cite: 57]
# #                                           batch_size=self.batchsize, # [cite: 58]
# #                                           vocab_file=self.vocab_file, # [cite: 58]
# #                                           category_dict=self.category_dict, # [cite: 59]
# #                                           num_workers=self.num_workers, # [cite: 59]
# #                                           # 使用 weibo_clip_dataloader.py 中定义的参数名
# #                                           cn_clip_model_name=self.config.get('cn_clip_model_name', "ViT-B-16"), # [cite: 5]
# #                                           cn_clip_download_root=self.config.get('cn_clip_download_root', './') # [cite: 5]
# #                                           )
# #             # PKL文件路径相对于项目根目录
# #             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl') # [cite: 61]
# #             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
# #             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
# #             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
# #             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
# #             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

# #             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True) # [cite: 62]
# #             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
# #             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)


# #         elif self.dataset == "weibo21":
# #             if Weibo21DataLoaderClass is None:
# #                 raise ImportError("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
# #             logger.info("为 Weibo21 数据集加载 utils.weibo21_clip_dataloader.bert_data") # [cite: 63]
# #             loader = Weibo21DataLoaderClass(max_len=self.max_len, # BERT max_len
# #                                              batch_size=self.batchsize,
# #                                              vocab_file=self.vocab_file, # [cite: 64]
# #                                              category_dict=self.category_dict,
# #                                              num_workers=self.num_workers, # [cite: 65]
# #                                              # cn_clip_model_name 和 cn_clip_download_root 会使用 dataloader 中的默认值
# #                                              # cn_clip_model_name=self.config.get('cn_clip_model_name', "ViT-B-16"), # [cite: 5]
# #                                              # cn_clip_download_root=self.config.get('cn_clip_download_root', './') # [cite: 5]
# #                                              )
# #             # PKL文件路径相对于项目根目录
# #             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
# #             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl') # [cite: 67]
# #             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
# #             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
# #             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
# #             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

# #             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
# #             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
# #             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False) # [cite: 68]
# #         else:
# #             raise ValueError(f"数据集 {self.dataset} 的 Dataloader 逻辑未定义。")

# #         if train_loader is None or val_loader is None or test_loader is None:
# #             logger.error("一个或多个 dataloader 初始化失败。正在退出。")
# #             raise RuntimeError("Dataloader 初始化失败。")

# #         logger.info(f"训练加载器批次数: {len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'}") # [cite: 68, 69]
# #         logger.info(f"验证加载器批次数: {len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'}")
# #         logger.info(f"测试加载器批次数: {len(test_loader) if hasattr(test_loader, '__len__') else 'N/A'}")
# #         return train_loader, val_loader, test_loader

# #     def main(self):
# #         logger.info(f"开始运行: 数据集 {self.dataset}, 模型配置名 {self.model_name}")
# #         train_loader, val_loader, test_loader = self.get_dataloader()

# #         trainer = None
# #         if self.dataset == "gossipcop":
# #             if DOMAINTrainerGossipCop is None: # [cite: 70]
# #                 raise ImportError(f"DOMAINTrainerGossipCop 未加载，无法训练 {self.dataset}。")
# #             if self.model_name != 'domain_gossipcop':
# #                 logger.warning(f"GossipCop 通常使用 'domain_gossipcop' 模型, 当前为 '{self.model_name}'.")
# #             trainer = DOMAINTrainerGossipCop(
# #                 emb_dim=self.emb_dim,
# #                 mlp_dims=self.config['model_params']['mlp']['dims'], # [cite: 71]
# #                 bert_path_or_name=self.bert_model_path,
# #                 clip_path_or_name=self.clip_model_path,
# #                 use_cuda=self.use_cuda,
# #                 lr=self.lr,
# #                 train_loader=train_loader,
# #                 val_loader=val_loader, # [cite: 72]
# #                 test_loader=test_loader,
# #                 dropout=self.config['model_params']['mlp']['dropout'],
# #                 weight_decay=self.config.get('weight_decay', 5e-5),
# #                 category_dict=self.category_dict,
# #                 early_stop=self.early_stop,
# #                 metric_key_for_early_stop=self.early_stop_metric_key,
# #                 epoches=self.epoch, # [cite: 73]
# #                 save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
# #             )
# #         elif self.dataset == "weibo" or self.dataset == "weibo21":
# #             if DOMAINTrainerWeibo is None:
# #                 raise ImportError(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
# #             if self.model_name != 'domain_weibo': # [cite: 74]
# #                  logger.warning(f"Weibo/Weibo21 通常使用 'domain_weibo' 模型, 当前为 '{self.model_name}'.")
# #             trainer = DOMAINTrainerWeibo(
# #                 emb_dim=self.emb_dim,
# #                 mlp_dims=self.config['model_params']['mlp']['dims'],
# #                 bert=self.bert_model_path, # domain_weibo.py 内部使用此键名
# #                 use_cuda=self.use_cuda, # [cite: 75]
# #                 lr=self.lr,
# #                 dropout=self.config['model_params']['mlp']['dropout'],
# #                 train_loader=train_loader,
# #                 val_loader=val_loader,
# #                 test_loader=test_loader,
# #                 category_dict=self.category_dict, # [cite: 76]
# #                 weight_decay=self.config.get('weight_decay', 5e-5),
# #                 save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}"),
# #                 early_stop=self.early_stop,
# #                 epoches=self.epoch
# #             )
# #         else:
# #             raise ValueError(f"数据集 {self.dataset} 没有匹配的 Trainer 初始化逻辑。") # [cite: 77]

# #         if trainer:
# #             logger.info("Trainer 初始化完成。开始训练...")
# #             try:
# #                 trainer.train()
# #                 logger.info("训练过程结束。")
# #             except Exception as train_e:
# #                 logger.exception(f"训练过程中发生严重错误: {train_e}") # [cite: 78]
# #         else:
# #             logger.error("Trainer 未初始化。")


# # run.py (修改后，用于加载Reasoning Dataloader)

# # run.py (修改后，适配 weibo_reasoning_dataloader.py)

# # run.py (修改后，适配 weibo_reasoning_dataloader.py)

# # main.py (Corrected version)
# # -*-coding = utf-8 -*-

# # run.py (修正版)
# # -*-coding = utf-8 -*-

# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, CLIPProcessor  # For GossipCop
# import logging

# # --- Logger Setup ---
# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# # --- GossipCop 相关导入 ---
# FakeNet_dataset = None
# try:
#     from FakeNet_dataset import FakeNet_dataset
#     logger.info("FakeNet_dataset 导入成功。")
# except ImportError as e:
#     logger.error(f"无法导入 FakeNet_dataset: {e}. 请确保 FakeNet_dataset.py 在当前工作目录或 Python 路径下。")

# # --- Weibo/Weibo21 相关导入 ---
# WeiboDataLoaderClass = None
# Weibo21DataLoaderClass = None
# try:
#     from utils.clip_dataloader import bert_data as WeiboDataLoaderClass
#     logger.info("utils.clip_dataloader.bert_data (for Weibo) 导入成功。")
# except ImportError:
#     logger.warning("无法从 utils.clip_dataloader 导入 Weibo 数据加载器。")

# try:
#     from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass
#     logger.info("utils.weibo21_clip_dataloader.bert_data (for Weibo21) 导入成功。")
# except ImportError:
#     logger.warning("无法从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器。")

# # --- 根据模型名称选择合适的Trainer ---
# DOMAINTrainerGossipCop = None
# DOMAINTrainerWeibo = None

# try:
#     # MODIFICATION: Import the modified Trainer that supports distillation loss
#     from model.domain_gossipcop import Trainer as DOMAINTrainerGossipCop
#     logger.info("model.domain_gossipcop.Trainer 导入成功。")
# except ImportError:
#     logger.warning("无法导入 model.domain_gossipcop.Trainer。如果未使用GossipCop数据集则忽略。")

# try:
#     from model.domain_weibo import DOMAINTrainerWeibo
#     logger.info("model.domain_weibo.Trainer 导入成功。")
# except ImportError:
#     logger.warning("无法导入 model.domain_weibo.Trainer。如果未使用Weibo/Weibo21数据集则忽略。")

# # --- 全局 Tokenizer/Processor 实例 (仅GossipCop需要全局预加载并传递) ---
# bert_tokenizer_gossipcop = None
# clip_processor_gossipcop = None

# # --- GossipCop 使用的 Collate Function ---
# def collate_fn_gossipcop(batch):
#     batch_before_filtering_len = len(batch)
#     batch = [item for item in batch if item is not None]
#     if batch_before_filtering_len > 0 and not batch:
#         logger.warning(f"Collate (gossipcop): 所有 {batch_before_filtering_len} 个项目在此批次中均为 None，返回 None。这可能表示数据加载问题。")
#         return None
#     if not batch:
#         logger.warning("Collate (gossipcop): 接收到空批次或所有项目过滤后批次为空，返回 None。")
#         return None

#     keys = batch[0].keys()
#     collated_batch = {}
#     for key in keys:
#         values = [item[key] for item in batch]
#         # MODIFICATION: Handle non-tensor data like reasoning embeddings which might be numpy arrays or lists from the dataloader
#         # This part of the collate_fn is crucial for handling the reasoning embeddings correctly.
#         # We assume the embeddings are returned as tensors from the dataset's __getitem__ method.
#         if all(isinstance(v, torch.Tensor) for v in values):
#             try:
#                 collated_batch[key] = torch.stack(values, dim=0)
#             except RuntimeError as e:
#                 logger.error(f"Collate (gossipcop) 无法为键 '{key}' 堆叠张量 (可能是尺寸不匹配): {e}")
#                 for i, v_item in enumerate(values):
#                     logger.error(f"  Item {i} ('{key}') shape: {v_item.shape}")
#                 logger.error(f"  批次中导致错误的张量列表 ('{key}'): {[v.shape for v in values]}")
#                 return None
#         else:
#             collated_batch[key] = values
#     return collated_batch


# class Run():
#     def __init__(self, config):
#         self.config = config
#         self.use_cuda = config.get('use_cuda', torch.cuda.is_available())
#         self.dataset = config['dataset']
#         self.model_name = config['model_name']
#         self.lr = config['lr']
#         self.batchsize = config['batchsize']
#         self.emb_dim = config['emb_dim']
#         self.max_len = config['max_len']
#         self.num_workers = config.get('num_workers', 0)
#         self.early_stop = config['early_stop']
#         self.epoch = config['epoch']
#         self.save_param_dir = config['save_param_dir']

#         logger.info(f"Run initialized with config: {config}")
#         logger.info(f"PyTorch version: {torch.__version__}")
#         logger.info(f"CUDA available: {torch.cuda.is_available()}, Using CUDA: {self.use_cuda}")

#         if self.dataset == "gossipcop":
#             self.root_path = self.config.get('gossipcop_data_dir')
#             # MODIFICATION: Get the reasoning CSV path from config
#             self.reasoning_csv_path = self.config.get('gossipcop_reasoning_csv_path')
#             self.bert_model_path = self.config.get('bert_model_path_gossipcop')
#             self.clip_model_path = self.config.get('clip_model_path_gossipcop')
#             self.category_dict = {"gossip": 0}
#             self.early_stop_metric_key = self.config.get('early_stop_metric', 'F1')

#             if not self.root_path or not self.bert_model_path or not self.clip_model_path or not self.reasoning_csv_path:
#                 logger.error("GossipCop: 'gossipcop_data_dir', 'bert_model_path_gossipcop', 'clip_model_path_gossipcop', or 'gossipcop_reasoning_csv_path' 未在配置中提供。")
#                 raise ValueError("GossipCop 配置路径缺失。")

#             global bert_tokenizer_gossipcop, clip_processor_gossipcop
#             try:
#                 logger.info(f"尝试从本地路径加载 BERT tokenizer (GossipCop): {self.bert_model_path}")
#                 bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
#             except Exception as e:
#                 logger.error(f"从本地路径加载 BERT tokenizer (GossipCop) 失败: {e}. Path: {self.bert_model_path}")
#             try:
#                 logger.info(f"尝试从本地路径加载 CLIP processor (GossipCop): {self.clip_model_path}")
#                 clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
#             except Exception as e:
#                 logger.error(f"从本地路径加载 CLIP processor (GossipCop) 失败: {e}. Path: {self.clip_model_path}")

#             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
#                 logger.warning("GossipCop 所需的 BERT tokenizer 或 CLIP processor 未能成功加载。将在 get_dataloader 中进一步检查。")

#         # ... (rest of the __init__ method for weibo/weibo21 remains the same)
#         elif self.dataset == "weibo":
#             self.root_path = self.config.get('weibo_data_dir')
#             if not self.root_path:
#                 logger.error("Weibo: 'weibo_data_dir' 未在配置中提供。")
#                 raise ValueError("Weibo 数据根目录 'weibo_data_dir' 配置缺失。")
#             self.train_path = os.path.join(self.root_path, 'train_origin.csv')
#             self.val_path = os.path.join(self.root_path, 'val_origin.csv')
#             self.test_path = os.path.join(self.root_path, 'test_origin.csv')
#             self.category_dict = { "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4, "国际": 5, "教育": 6, "娱乐": 7, "社会": 8 }
#             self.bert_model_path = self.config.get('bert_model_path_weibo')
#             self.vocab_file = self.config.get('vocab_file')
#             self.early_stop_metric_key = 'metric'
#             if not self.bert_model_path or not self.vocab_file:
#                 logger.error("Weibo: 'bert_model_path_weibo' or 'vocab_file' 未在配置中提供。")
#                 raise ValueError("Weibo BERT 模型路径或词汇表文件配置缺失。")
#         elif self.dataset == "weibo21":
#             self.root_path = self.config.get('weibo21_data_dir')
#             if not self.root_path:
#                 logger.error("Weibo21: 'weibo21_data_dir' 未在配置中提供。")
#                 raise ValueError("Weibo21 数据根目录 'weibo21_data_dir' 配置缺失。")
#             self.train_path = os.path.join(self.root_path, 'train_datasets.xlsx')
#             self.val_path = os.path.join(self.root_path, 'val_datasets.xlsx')
#             self.test_path = os.path.join(self.root_path, 'test_datasets.xlsx')
#             self.category_dict = { "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8 }
#             self.bert_model_path = self.config.get('bert_model_path_weibo')
#             self.vocab_file = self.config.get('vocab_file')
#             self.early_stop_metric_key = 'metric'
#             if not self.bert_model_path or not self.vocab_file:
#                 logger.error("Weibo21: 'bert_model_path_weibo' (for weibo21) or 'vocab_file' 未在配置中提供。")
#                 raise ValueError("Weibo21 BERT 模型路径或词汇表文件配置缺失。")
#         else:
#             logger.error(f"未知数据集: {self.dataset}. 请检查配置。")
#             raise ValueError(f"未知数据集: {self.dataset}")

#     def get_dataloader(self):
#         train_loader, val_loader, test_loader = None, None, None
#         logger.info(f"开始为数据集 '{self.dataset}' 获取 Dataloaders...")

#         if self.dataset == "gossipcop":
#             logger.info("为 GossipCop 加载 FakeNet_dataset 和 collate_fn_gossipcop")
#             if FakeNet_dataset is None:
#                 logger.error("FakeNet_dataset 未成功导入，无法为GossipCop加载数据。请检查之前的导入错误。")
#                 raise ImportError("FakeNet_dataset 未成功导入，无法为GossipCop加载数据。")
#             if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
#                 logger.error("GossipCop 数据集所需的全局 BERT tokenizer 或 CLIP processor 未加载。请检查 __init__ 中的加载日志。")
#                 raise RuntimeError("GossipCop 数据集所需的全局 tokenizer/processor 未加载。")

#             img_size = 224
#             clip_max_len = 77
            
#             logger.info(f"GossipCop root_path: {self.root_path}")
#             logger.info("Initializing GossipCop train_dataset...")
#             # MODIFICATION: Pass the reasoning_csv_path to the dataset loader.
#             # IMPORTANT: Your FakeNet_dataset class must be updated to accept and use this parameter.
#             train_dataset = FakeNet_dataset(
#                 root_path=self.root_path,
#                 reasoning_csv_path=self.reasoning_csv_path,  # Pass the new path here
#                 bert_tokenizer_instance=bert_tokenizer_gossipcop,
#                 clip_processor_instance=clip_processor_gossipcop,
#                 dataset_name="gossip",
#                 image_size=img_size,
#                 is_train=True,
#                 bert_max_len=self.max_len,
#                 clip_max_len=clip_max_len
#             )
#             if len(train_dataset) == 0:
#                 logger.warning("GossipCop 训练数据集加载后长度为 0。请检查数据源和 FakeNet_dataset 实现。")
#             train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True,
#                                       collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
#                                       drop_last=True, pin_memory=self.use_cuda)

#             logger.info("Initializing GossipCop validation_dataset...")
#             # MODIFICATION: Pass the reasoning_csv_path to the validation dataset loader as well.
#             val_dataset = FakeNet_dataset(
#                 root_path=self.root_path,
#                 reasoning_csv_path=self.reasoning_csv_path,  # Pass the new path here
#                 bert_tokenizer_instance=bert_tokenizer_gossipcop,
#                 clip_processor_instance=clip_processor_gossipcop,
#                 dataset_name="gossip",
#                 image_size=img_size,
#                 is_train=False,
#                 bert_max_len=self.max_len,
#                 clip_max_len=clip_max_len
#             )
#             if len(val_dataset) == 0:
#                 logger.warning("GossipCop 验证/测试数据集加载后长度为 0。请检查数据源和 FakeNet_dataset 实现。")
#             val_loader = DataLoader(val_dataset, batch_size=self.batchsize, shuffle=False,
#                                     collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
#                                     drop_last=False, pin_memory=self.use_cuda)
#             test_loader = val_loader

#         # ... (rest of the get_dataloader method for weibo/weibo21 remains the same)
#         elif self.dataset == "weibo":
#             if WeiboDataLoaderClass is None:
#                 logger.error("Weibo 数据加载器 (WeiboDataLoaderClass from utils.clip_dataloader) 未成功导入。")
#                 raise ImportError("Weibo 数据加载器 (WeiboDataLoaderClass from utils.clip_dataloader) 未成功导入。")
#             loader = WeiboDataLoaderClass(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file, category_dict=self.category_dict, num_workers=self.num_workers, clip_model_name="ViT-B-16", clip_download_root='./')
#             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
#             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
#             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
#             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
#             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
#             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')
#             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
#             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
#             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)
#         elif self.dataset == "weibo21":
#             if Weibo21DataLoaderClass is None:
#                 logger.error("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
#                 raise ImportError("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
#             loader = Weibo21DataLoaderClass(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file, category_dict=self.category_dict, num_workers=self.num_workers)
#             train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
#             train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl') 
#             val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
#             val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
#             test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
#             test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')
#             train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
#             val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
#             test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)
#         else:
#             raise ValueError(f"数据集 {self.dataset} 的 Dataloader 逻辑未定义。")

#         if not all([train_loader, val_loader, test_loader]):
#             error_messages = [name for name, loader in [("train_loader", train_loader), ("val_loader", val_loader), ("test_loader", test_loader)] if loader is None]
#             full_error_message = f"Dataloader 初始化失败: {', '.join(error_messages)} for dataset '{self.dataset}'. 请检查上述日志以获取详细信息，特别是文件路径和外部 dataloader 的行为。"
#             logger.error(full_error_message)

#         return train_loader, val_loader, test_loader

#     def main(self):
#         logger.info(f"开始运行主程序: 数据集 '{self.dataset}', 模型配置名 '{self.model_name}'")
#         try:
#             train_loader, val_loader, test_loader = self.get_dataloader()
#         except (RuntimeError, ImportError, ValueError) as e:
#             logger.error(f"在 get_dataloader 期间发生错误: {e}")
#             logger.error("无法继续进行训练。请检查配置和数据。")
#             return 
    
#         trainer = None
#         model_save_path = os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
#         logger.info(f"模型参数将保存到: {model_save_path}")
#         os.makedirs(model_save_path, exist_ok=True) 
    
#         if self.dataset == "gossipcop":
#             if DOMAINTrainerGossipCop is None: 
#                 logger.error(f"DOMAINTrainerGossipCop 未加载，无法训练 {self.dataset}。")
#                 raise ImportError(f"DOMAINTrainerGossipCop 未加载，无法训练 {self.dataset}。")
#             if 'model_params' not in self.config or 'mlp' not in self.config['model_params']:
#                 logger.error("GossipCop trainer: 'model_params.mlp' 未在配置中找到。")
#                 raise ValueError("'model_params.mlp' (包含 'dims' 和 'dropout') 配置缺失。")
    
#             # MODIFICATION: Pass the distillation_weight to the Trainer.
#             trainer = DOMAINTrainerGossipCop(
#                 emb_dim=self.emb_dim,
#                 mlp_dims=self.config['model_params']['mlp']['dims'], 
#                 bert_path_or_name=self.bert_model_path,
#                 clip_path_or_name=self.clip_model_path,
#                 use_cuda=self.use_cuda,
#                 lr=self.lr,
#                 train_loader=train_loader,
#                 val_loader=val_loader, 
#                 test_loader=test_loader,
#                 dropout=self.config['model_params']['mlp']['dropout'],
#                 weight_decay=self.config.get('weight_decay', 5e-5),
#                 category_dict=self.category_dict,
#                 early_stop=self.early_stop,
#                 metric_key_for_early_stop=self.early_stop_metric_key,
#                 epoches=self.epoch, 
#                 save_param_dir=model_save_path,
#                 distillation_weight=self.config.get('distillation_weight', 0.1)  # Add the new parameter
#             )
#         # ... (rest of the main method for weibo/weibo21 remains the same)
#         elif self.dataset in ["weibo", "weibo21"]:
#             if DOMAINTrainerWeibo is None:
#                 logger.error(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
#                 raise ImportError(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
#             if 'model_params' not in self.config or 'mlp' not in self.config['model_params']:
#                 logger.error(f"{self.dataset} trainer: 'model_params.mlp' 未在配置中找到。")
#                 raise ValueError("'model_params.mlp' (包含 'dims' 和 'dropout') 配置缺失。")
#             trainer = DOMAINTrainerWeibo(emb_dim=self.emb_dim, mlp_dims=self.config['model_params']['mlp']['dims'], bert=self.bert_model_path, use_cuda=self.use_cuda, lr=self.lr, dropout=self.config['model_params']['mlp']['dropout'], train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, category_dict=self.category_dict, weight_decay=self.config.get('weight_decay', 5e-5), save_param_dir=model_save_path, early_stop=self.early_stop, epoches=self.epoch)
#         else:
#             logger.error(f"数据集 '{self.dataset}' 没有匹配的 Trainer 初始化逻辑。")
#             raise ValueError(f"数据集 {self.dataset} 没有匹配的 Trainer 初始化逻辑。")
    
#         if trainer:
#             logger.info(f"Trainer for '{self.dataset}' 初始化完成。开始训练...")
#             try:
#                 trainer.train()
#                 logger.info(f"训练过程已成功完成 for {self.dataset} - {self.model_name}.")
#             except Exception as train_e:
#                 logger.exception(f"训练过程中发生严重错误 for {self.dataset} - {self.model_name}: {train_e}")
#         else:
#             logger.error(f"Trainer for '{self.dataset}' 未能初始化。训练无法开始。")

# # ... (if __name__ == '__main__' block remains the same)
# if __name__ == '__main__':
#     logger.info("run.py executed directly (example). 通常此文件作为模块导入。")
#     logger.info("要运行，请从 main.py 调用此类并提供有效的配置。")

# run.py (unified)
# run.py (unified, patched)
import os
import logging
import traceback
import inspect
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor

# -----------------------
# Logger
# -----------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# -----------------------
# Optional imports (datasets)
# -----------------------
FakeNet_dataset = None
try:
    from FakeNet_dataset import FakeNet_dataset
    logger.info("FakeNet_dataset 导入成功。")
except Exception as e:
    logger.warning(f"FakeNet_dataset 未导入：{e}")

WeiboDataLoaderClass = None
try:
    from utils.clip_dataloader import bert_data as WeiboDataLoaderClass
    logger.info("utils.clip_dataloader.bert_data (Weibo) 导入成功。")
except Exception as e:
    logger.warning(f"Weibo dataloader 未导入：{e}")

Weibo21DataLoaderClass = None
try:
    from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass
    logger.info("utils.weibo21_clip_dataloader.bert_data (Weibo21) 导入成功。")
except Exception as e:
    logger.warning(f"Weibo21 dataloader 未导入：{e}")

# -----------------------
# Optional imports (trainers)
# -----------------------
DOMAINTrainerGossipCop = None
try:
    from model.domain_gossipcop import Trainer as DOMAINTrainerGossipCop
    logger.info("model.domain_gossipcop.Trainer 导入成功。")
except Exception as e:
    logger.warning(f"GossipCop Trainer 未导入：{e}")

# weibo / weibo21：根据数据集在 __init__ 中决定导入路径
DOMAINTrainerWeibo = None

# -----------------------
# global tokenizer/processor for gossipcop
# -----------------------
bert_tokenizer_gossipcop = None
clip_processor_gossipcop = None


# -----------------------
# Collate for GossipCop
# -----------------------
def collate_fn_gossipcop(batch):
    """允许 batch 中 None 过滤；张量字段自动 stack；其他字段以 list 形式保留。"""
    n0 = len(batch)
    batch = [x for x in batch if x is not None]
    if n0 > 0 and not batch:
        logger.warning("Collate(gossipcop): 所有样本为 None，返回 None。")
        return None
    if not batch:
        return None

    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [it[k] for it in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError as e:
                logger.error(f"collate '{k}' 堆叠失败：{e}")
                for i, v in enumerate(vals):
                    logger.error(f"  #{i} shape={getattr(v, 'shape', None)}")
                return None
        else:
            out[k] = vals
    return out


class Run:
    """
    统一 Runner
    需要的 config 关键字段（与合并后的 main.py 一致）：
      dataset: gossipcop/weibo21/weibo
      model_name, lr, batchsize, emb_dim, max_len, num_workers, early_stop, epoch, save_param_dir
      early_stop_metric
      gossipcop_data_dir / weibo_data_dir / weibo21_data_dir
      bert_model_path_gossipcop, clip_model_path_gossipcop
      bert_model_path_weibo, vocab_file
      weight_decay, model_params.mlp.dims & .dropout
      distillation_weight (gossipcop 可选)
      lambda_reasoning_align (weibo/weibo21 可选)
      gossipcop_reasoning_csv_path (gossipcop 需要)
    """
    def __init__(self, config):
        self.config = config
        self.use_cuda = config.get("use_cuda", torch.cuda.is_available())

        # 通用
        self.dataset = config["dataset"]
        self.model_name = config["model_name"]
        self.lr = config["lr"]
        self.batchsize = config["batchsize"]
        self.emb_dim = config["emb_dim"]
        self.max_len = config["max_len"]
        self.num_workers = config.get("num_workers", 0)
        self.early_stop = config["early_stop"]
        self.epoch = config["epoch"]
        self.save_param_dir = config["save_param_dir"]

        logger.info(f"Run init, dataset={self.dataset}, model_name={self.model_name}")

        # 分数据集初始化
        if self.dataset == "gossipcop":
            self._init_gossipcop()
        elif self.dataset == "weibo":
            self._init_weibo()
        elif self.dataset == "weibo21":
            self._init_weibo21()
        else:
            raise ValueError(f"未知数据集: {self.dataset}")

    # -------- gossipcop init --------
    def _init_gossipcop(self):
        self.root_path = self.config.get("gossipcop_data_dir")
        self.reasoning_csv_path = self.config.get("gossipcop_reasoning_csv_path")
        self.bert_model_path = self.config.get("bert_model_path_gossipcop")
        self.clip_model_path = self.config.get("clip_model_path_gossipcop")
        self.category_dict = {"gossip": 0}
        self.early_stop_metric_key = self.config.get("early_stop_metric", "F1")

        for key, val in {
            "gossipcop_data_dir": self.root_path,
            "gossipcop_reasoning_csv_path": self.reasoning_csv_path,
            "bert_model_path_gossipcop": self.bert_model_path,
            "clip_model_path_gossipcop": self.clip_model_path,
        }.items():
            if not val:
                raise ValueError(f"GossipCop 缺少配置: {key}")

        global bert_tokenizer_gossipcop, clip_processor_gossipcop
        try:
            bert_tokenizer_gossipcop = BertTokenizer.from_pretrained(self.bert_model_path)
        except Exception as e:
            logger.error(f"BERT tokenizer 加载失败（GossipCop）: {e}")
        try:
            clip_processor_gossipcop = CLIPProcessor.from_pretrained(self.clip_model_path)
        except Exception as e:
            logger.error(f"CLIP processor 加载失败（GossipCop）: {e}")

        if bert_tokenizer_gossipcop is None or clip_processor_gossipcop is None:
            raise RuntimeError("GossipCop 所需 tokenizer/processor 未能加载。")

    # -------- weibo init --------
    def _init_weibo(self):
        self.root_path = self.config.get("weibo_data_dir")
        if not self.root_path:
            raise ValueError("Weibo 缺少 weibo_data_dir")
        self.train_path = os.path.join(self.root_path, "train_origin.csv")
        self.val_path = os.path.join(self.root_path, "val_origin.csv")
        self.test_path = os.path.join(self.root_path, "test_origin.csv")

        self.category_dict = {"经济":0,"健康":1,"军事":2,"科学":3,"政治":4,"国际":5,"教育":6,"娱乐":7,"社会":8}
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")
        self.early_stop_metric_key = "metric"
        if not self.bert_model_path or not self.vocab_file:
            raise ValueError("Weibo 缺少 bert_model_path_weibo 或 vocab_file")

        # weibo -> Trainer 在 model/domain_weibo.py
        global DOMAINTrainerWeibo
        if DOMAINTrainerWeibo is None or DOMAINTrainerWeibo.__module__ != "model.domain_weibo":
            try:
                from model.domain_weibo import DOMAINTrainerWeibo as _TW
                DOMAINTrainerWeibo = _TW
                logger.info("已使用 model.domain_weibo.DOMAINTrainerWeibo")
            except Exception:
                logger.error("导入 model.domain_weibo.DOMAINTrainerWeibo 失败：\n" + traceback.format_exc())
                raise

    # -------- weibo21 init --------
    def _init_weibo21(self):
        self.root_path = self.config.get("weibo21_data_dir")
        if not self.root_path:
            raise ValueError("Weibo21 缺少 weibo21_data_dir")
        self.train_path = os.path.join(self.root_path, "train_datasets.xlsx")
        self.val_path = os.path.join(self.root_path, "val_datasets.xlsx")
        self.test_path = os.path.join(self.root_path, "test_datasets.xlsx")

        self.category_dict = {"科技":0,"军事":1,"教育考试":2,"灾难事故":3,"政治":4,"医药健康":5,"财经商业":6,"文体娱乐":7,"社会生活":8}
        self.bert_model_path = self.config.get("bert_model_path_weibo")
        self.vocab_file = self.config.get("vocab_file")
        self.early_stop_metric_key = "metric"
        if not self.bert_model_path or not self.vocab_file:
            raise ValueError("Weibo21 缺少 bert_model_path_weibo 或 vocab_file")

        # weibo21 -> Trainer 在 model/domain_weibo21.py  （重点要求）
        global DOMAINTrainerWeibo
        if DOMAINTrainerWeibo is None or DOMAINTrainerWeibo.__module__ != "model.domain_weibo21":
            try:
                from model.domain_weibo21 import DOMAINTrainerWeibo as _TW21
                DOMAINTrainerWeibo = _TW21
                logger.info("已使用 model.domain_weibo21.DOMAINTrainerWeibo（weibo21 专用）")
            except Exception:
                logger.error("导入 model.domain_weibo21.DOMAINTrainerWeibo 失败：\n" + traceback.format_exc())
                raise

    # -----------------------
    # Dataloaders
    # -----------------------
    def get_dataloader(self):
        logger.info(f"准备 dataloaders: {self.dataset}")
        if self.dataset == "gossipcop":
            if FakeNet_dataset is None:
                raise ImportError("缺少 FakeNet_dataset")
            img_size, clip_max_len = 224, 77

            train_ds = FakeNet_dataset(
                root_path=self.root_path,
                reasoning_csv_path=self.reasoning_csv_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=True,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len
            )
            val_ds = FakeNet_dataset(
                root_path=self.root_path,
                reasoning_csv_path=self.reasoning_csv_path,
                bert_tokenizer_instance=bert_tokenizer_gossipcop,
                clip_processor_instance=clip_processor_gossipcop,
                dataset_name="gossip",
                image_size=img_size,
                is_train=False,
                bert_max_len=self.max_len,
                clip_max_len=clip_max_len
            )
            train_loader = DataLoader(train_ds, batch_size=self.batchsize, shuffle=True,
                                      collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
                                      drop_last=True, pin_memory=self.use_cuda)
            val_loader = DataLoader(val_ds, batch_size=self.batchsize, shuffle=False,
                                    collate_fn=collate_fn_gossipcop, num_workers=self.num_workers,
                                    drop_last=False, pin_memory=self.use_cuda)
            test_loader = val_loader

        elif self.dataset == "weibo":
            if WeiboDataLoaderClass is None:
                raise ImportError("缺少 utils.clip_dataloader.bert_data")
            loader = WeiboDataLoaderClass(
                max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                category_dict=self.category_dict, num_workers=self.num_workers,
                clip_model_name="ViT-B-16", clip_download_root="./"
            )
            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False
            )

        elif self.dataset == "weibo21":
            if Weibo21DataLoaderClass is None:
                raise ImportError("缺少 utils.weibo21_clip_dataloader.bert_data")
            loader = Weibo21DataLoaderClass(
                max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                category_dict=self.category_dict, num_workers=self.num_workers
            )
            train_loader = loader.load_data(
                self.train_path,
                os.path.join(self.root_path, "train_loader.pkl"),
                os.path.join(self.root_path, "train_clip_loader.pkl"),
                True
            )
            val_loader = loader.load_data(
                self.val_path,
                os.path.join(self.root_path, "val_loader.pkl"),
                os.path.join(self.root_path, "val_clip_loader.pkl"),
                False
            )
            test_loader = loader.load_data(
                self.test_path,
                os.path.join(self.root_path, "test_loader.pkl"),
                os.path.join(self.root_path, "test_clip_loader.pkl"),
                False
            )
        else:
            raise ValueError(f"未实现 dataloader: {self.dataset}")

        if not all([train_loader, val_loader, test_loader]):
            missing = [n for n, l in [("train", train_loader), ("val", val_loader), ("test", test_loader)] if l is None]
            raise RuntimeError("Dataloader 初始化失败: " + ", ".join(missing))

        return train_loader, val_loader, test_loader

    # -----------------------
    # Main
    # -----------------------
    def main(self):
        try:
            train_loader, val_loader, test_loader = self.get_dataloader()
        except Exception:
            logger.error("get_dataloader 失败：\n" + traceback.format_exc())
            return

        save_dir = os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"模型参数将保存到：{save_dir}")

        trainer = None
        if self.dataset == "gossipcop":
            if DOMAINTrainerGossipCop is None:
                raise ImportError("DOMAINTrainerGossipCop 未导入")
            mp = self.config.get("model_params", {}).get("mlp", {})
            trainer = DOMAINTrainerGossipCop(
                emb_dim=self.emb_dim,
                mlp_dims=mp.get("dims", [384]),
                bert_path_or_name=self.bert_model_path,
                clip_path_or_name=self.clip_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dropout=mp.get("dropout", 0.2),
                weight_decay=self.config.get("weight_decay", 5e-5),
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                metric_key_for_early_stop=self.early_stop_metric_key,
                epoches=self.epoch,
                save_param_dir=save_dir,
                distillation_weight=self.config.get("distillation_weight", 0.1),
            )

        elif self.dataset in ("weibo", "weibo21"):
            if DOMAINTrainerWeibo is None:
                raise ImportError("DOMAINTrainerWeibo 未导入")
            mp = self.config.get("model_params", {}).get("mlp", {})

            # 公共参数
            weibo_kwargs = dict(
                emb_dim=self.emb_dim,
                mlp_dims=mp.get("dims", [384]),
                bert=self.bert_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=mp.get("dropout", 0.2),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=self.config.get("weight_decay", 5e-5),
                save_param_dir=save_dir,
                early_stop=self.early_stop,
                epoches=self.epoch,
            )

            # 只有当 __init__ 确实支持该参数时才传（避免 unexpected keyword）
            sig = inspect.signature(DOMAINTrainerWeibo.__init__)
            if "lambda_reasoning_align" in sig.parameters and self.dataset == "weibo21":
                weibo_kwargs["lambda_reasoning_align"] = self.config.get("lambda_reasoning_align", 0.1)

            trainer = DOMAINTrainerWeibo(**weibo_kwargs)

        else:
            raise ValueError(f"未实现 Trainer: {self.dataset}")

        logger.info("Trainer 初始化完成，开始训练…")
        try:
            trainer.train()
            logger.info(f"训练完成：{self.dataset} - {self.model_name}")
        except Exception:
            logger.error("训练过程中出现异常：\n" + traceback.format_exc())


# 仅被直接运行时的提示
if __name__ == "__main__":
    logger.info("本文件通常由 main.py 调用：Run(config).main()")


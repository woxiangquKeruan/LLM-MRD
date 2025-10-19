



# main.py
# 输入 python main.py --dataset gossipcop
# python main.py --dataset weibo21
# python main.py --dataset weibo
import os
import argparse
import logging
import random
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("runner")

# ---------------------------
# 1) 预设：三数据集推荐默认值
# ---------------------------
PRESETS = {
    "gossipcop": {
        "model_name": "domain_gossipcop",
        "model_py_path": "./model/domain_gossipcop.py",
        "lr": 3e-4,
        "batchsize": 24,
        "seed": 2024,
        "early_stop": 100,
        "early_stop_metric": "acc",
        "distillation_weight": 0.9,
        "lambda_reasoning_align": None,
        "data_dir_key": "gossipcop_data_dir",
    },
    "weibo21": {
        # 关键修改：weibo21 使用独立文件与名称
        "model_name": "domain_weibo21",
        "model_py_path": "./model/domain_weibo21.py",
        "lr": 5e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "distillation_weight": None,
        "lambda_reasoning_align": 0.1,
        "data_dir_key": "weibo21_data_dir",
    },
    "weibo": {
        "model_name": "domain_weibo",
        "model_py_path": "./model/domain_weibo.py",
        "lr": 2e-4,
        "batchsize": 64,
        "seed": 3074,
        "early_stop": 100,
        "early_stop_metric": "F1",
        "distillation_weight": None,
        "lambda_reasoning_align": 0.1,
        "data_dir_key": "weibo_data_dir",
    },
}

def pick(user_value, preset_value):
    return preset_value if user_value is None else user_value

# ---------------------------
# 2) 参数
# ---------------------------
parser = argparse.ArgumentParser(description="Unified runner for gossipcop/weibo21/weibo")

parser.add_argument("--dataset", choices=["gossipcop", "weibo21", "weibo"], required=True,
                    help="选择要运行的数据集。")

# 可覆写的通用训练超参（默认 None -> 用预设）
parser.add_argument("--model_name", default=None,
                    help="模型架构名：gossipcop=domain_gossipcop；weibo21=domain_weibo21；weibo=domain_weibo。")
parser.add_argument("--model_py_path", default=None,
                    help="模型 Python 文件路径（weibo21 默认 ./model/domain_weibo21.py）。")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--batchsize", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--early_stop", type=int, default=None)
parser.add_argument("--early_stop_metric", choices=["acc", "F1"], default=None)

# 训练 & 环境
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--max_len", type=int, default=197)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--gpu", default="0")
parser.add_argument("--bert_emb_dim", type=int, default=768)
parser.add_argument("--save_param_dir", default="./param_model")
parser.add_argument("--emb_type", default="bert")

# 预训练模型与数据路径
parser.add_argument("--bert_model_path_gossipcop", default="./pretrained_model/bert-base-uncased")
parser.add_argument("--clip_model_path_gossipcop", default="./pretrained_model/clip-vit-base-patch16")

parser.add_argument("--bert_model_path_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch")
parser.add_argument("--bert_vocab_file_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt")

parser.add_argument("--gossipcop_data_dir", default="./gossipcop/")
parser.add_argument("--gossipcop_reasoning_csv_path", default="./gossipcop_with_reasoning_encoded.csv")

parser.add_argument("--weibo_data_dir", default="./data/")
parser.add_argument("--weibo21_data_dir", default="./Weibo_21/")

# 仅在对应数据集生效的损失权重
parser.add_argument("--distillation_weight", type=float, default=None,
                    help="GossipCop 的蒸馏损失权重。")
parser.add_argument("--lambda_reasoning_align", type=float, default=None,
                    help="Weibo/Weibo21 的对齐损失权重。")

args = parser.parse_args()

# 先设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

try:
    from run import Run  # 你的训练入口类
except Exception as e:
    logger.error(f"无法导入 Run：{e}。请确认 run.py 存在且实现 Run(config).main()。")
    raise SystemExit(1)

# ---------------------------
# 3) 根据数据集应用预设 & 用户覆写
# ---------------------------
p = PRESETS[args.dataset]

current = {
    "dataset": args.dataset,
    "model_name": pick(args.model_name, p["model_name"]),
    "model_py_path": pick(args.model_py_path, p["model_py_path"]),  # <== 关键：把模型文件放进 config
    "lr": pick(args.lr, p["lr"]),
    "batchsize": pick(args.batchsize, p["batchsize"]),
    "seed": pick(args.seed, p["seed"]),
    "early_stop": pick(args.early_stop, p["early_stop"]),
    "early_stop_metric": pick(args.early_stop_metric, p["early_stop_metric"]),
    "distillation_weight": pick(args.distillation_weight, p["distillation_weight"]),
    "lambda_reasoning_align": pick(args.lambda_reasoning_align, p["lambda_reasoning_align"]),
}

# ---------------------------
# 4) 固定随机性
# ---------------------------
random.seed(current["seed"])
np.random.seed(current["seed"])
torch.manual_seed(current["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 5) 组装统一 config（Run 接口一致）
# ---------------------------
use_cuda = torch.cuda.is_available()
config = {
    "use_cuda": use_cuda,
    "dataset": current["dataset"],
    "model_name": current["model_name"],
    "model_py_path": current["model_py_path"],  # <== 传给 Run，由 Run 按需 import

    # 数据与模型路径
    "gossipcop_data_dir": args.gossipcop_data_dir,
    "weibo_data_dir": args.weibo_data_dir,
    "weibo21_data_dir": args.weibo21_data_dir,

    "bert_model_path_gossipcop": args.bert_model_path_gossipcop,
    "clip_model_path_gossipcop": args.clip_model_path_gossipcop,

    "bert_model_path_weibo": args.bert_model_path_weibo,
    "bert_vocab_file_weibo": args.bert_vocab_file_weibo,

    # 训练相关
    "batchsize": current["batchsize"],
    "max_len": args.max_len,
    "early_stop": current["early_stop"],
    "early_stop_metric": current["early_stop_metric"],
    "num_workers": args.num_workers,
    "emb_type": args.emb_type,
    "weight_decay": 5e-5,
    "model_params": {"mlp": {"dims": [384], "dropout": 0.2}},
    "emb_dim": args.bert_emb_dim,
    "lr": current["lr"],
    "epoch": args.epoch,
    "seed": current["seed"],
    "save_param_dir": args.save_param_dir,

    "distillation_weight": current["distillation_weight"],
    "lambda_reasoning_align": current["lambda_reasoning_align"],
    "gossipcop_reasoning_csv_path": args.gossipcop_reasoning_csv_path,
}

# weibo/weibo21 需要 vocab、bert
if args.dataset in {"weibo", "weibo21"}:
    config["vocab_file"] = args.bert_vocab_file_weibo
    config["bert"] = args.bert_model_path_weibo

# ---------------------------
# 6) 日志打印
# ---------------------------
logger.info("===== Final Config =====")
for k, v in config.items():
    logger.info(f"{k}: {v}")
logger.info("========================")

# ---------------------------
# 7) 跑起来
# ---------------------------
if __name__ == "__main__":
    if config["use_cuda"]:
        logger.info(f"CUDA 可用，使用 GPU {args.gpu}")
    else:
        logger.warning("CUDA 不可用，将使用 CPU")

    # 要求：Run 内部需支持通过 config["model_py_path"] 动态加载模型文件；
    # 若 Run 之前只用 model_name，这里同时把 model_name 改成 domain_weibo21，
    # 兼容两种方式。
    runner = Run(config=config)
    runner.main()
    logger.info("运行结束。")

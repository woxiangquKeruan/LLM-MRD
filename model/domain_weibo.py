#  baseline 代码    原版
# import os
# import tqdm
# import torch
# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
# from transformers import BertModel
# import torch.nn as nn
# # from positional_encodings.torch_encodings import PositionalEncoding1D
# import models_mae
# from utils.utils_weibo import data2gpu, Averager, metrics, Recorder, clipdata2gpu
# from utils.utils_weibo import metricsTrueFalse
# from .layers import *
# from .pivot import *
# from timm.models.vision_transformer import Block
# import cn_clip.clip as clip
# from cn_clip.clip import load_from_name, available_models
# class SimpleGate(nn.Module):
#     def __init__(self, dim=1):
#         super(SimpleGate, self).__init__()
#         self.dim = dim

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=self.dim)
#         return x1 * x2

# class AdaIN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def mu(self, x):
#         """ Takes a (n,c,h,w) tensor as input and returns the average across
#         it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
#         return torch.sum(x,(1))/(x.shape[1])

#     def sigma(self, x):
#         """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
#         across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
#         the permutations are required for broadcasting"""
#         return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

#     def forward(self, x, mu, sigma):
#         """ Takes a content embeding x and a style embeding y and changes
#         transforms the mean and standard deviation of the content embedding to
#         that of the style. [See eq. 8 of paper] Note the permutations are
#         required for broadcasting"""
#         # print(mu.shape) # 12
#         x_mean = self.mu(x)
#         x_std = self.sigma(x)
#         x_reduce_mean = x.permute([1, 0]) - x_mean
#         x_norm = x_reduce_mean/x_std
#         # print(x_mean.shape) # 768, 12
#         return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])




# class MultiDomainPLEFENDModel(torch.nn.Module):
#     def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
#         super(MultiDomainPLEFENDModel, self).__init__()
#         self.num_expert = 6
#         self.task_num = 2
#         #self.domain_num = 9
#         self.domain_num = self.task_num
#         self.gate_num = 3
#         self.num_share = 1
#         self.unified_dim, self.text_dim = emb_dim, 768
#         self.image_dim = 768
#         self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
#         feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
#         self.text_token_len = 197
#         self.image_token_len = 197

#         text_expert_list = []
#         for i in range(self.domain_num):
#             text_expert = []
#             for j in range(self.num_expert):
#                 text_expert.append(cnn_extractor(emb_dim, feature_kernel))

#             text_expert = nn.ModuleList(text_expert)
#             text_expert_list.append(text_expert)
#         self.text_experts = nn.ModuleList(text_expert_list)

#         image_expert_list = []
#         for i in range(self.domain_num):
#             image_expert = []
#             for j in range(self.num_expert):
#                 image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
#                 #image_expert.append(image_cnn_extractor())
#             image_expert = nn.ModuleList(image_expert)
#             image_expert_list.append(image_expert)
#         self.image_experts = nn.ModuleList(image_expert_list)

#         fusion_expert_list = []
#         for i in range(self.domain_num):
#             fusion_expert = []
#             for j in range(self.num_expert):
#                 expert = nn.Sequential(nn.Linear(320, 320),
#                                        nn.SiLU(),
#                                        #SimpleGate(),
#                                        #nn.BatchNorm1d(160),
#                                        nn.Linear(320, 320),
#                                        )
#                 fusion_expert.append(expert)
#             fusion_expert = nn.ModuleList(fusion_expert)
#             fusion_expert_list.append(fusion_expert)
#         self.fusion_experts = nn.ModuleList(fusion_expert_list)

#         final_expert_list = []
#         for i in range(self.domain_num):
#             final_expert = []
#             for j in range(self.num_expert):
#                 final_expert.append(Block(dim=320, num_heads=8))
#             final_expert = nn.ModuleList(final_expert)
#             final_expert_list.append(final_expert)
#         self.final_experts = nn.ModuleList(final_expert_list)

#         text_share_expert, image_share_expert, fusion_share_expert,final_share_expert = [], [], [],[]
#         for i in range(self.num_share):
#             text_share = []
#             image_share = []
#             fusion_share = []
#             final_share = []
#             for j in range(self.num_expert*2):
#                 text_share.append(cnn_extractor(emb_dim, feature_kernel))
#                 image_share.append(cnn_extractor(self.image_dim, feature_kernel))
#                 #image_share.append(image_cnn_extractor())
#                 expert = nn.Sequential(nn.Linear(320, 320),
#                                        nn.SiLU(),
#                                        #SimpleGate(),
#                                        #nn.BatchNorm1d(160),
#                                        nn.Linear(320, 320),
#                                        )
#                 fusion_share.append(expert)
#                 final_share.append(Block(dim=320, num_heads=8))
#             text_share = nn.ModuleList(text_share)
#             text_share_expert.append(text_share)
#             image_share = nn.ModuleList(image_share)
#             image_share_expert.append(image_share)
#             fusion_share = nn.ModuleList(fusion_share)
#             fusion_share_expert.append(fusion_share)
#             final_share = nn.ModuleList(final_share)
#             final_share_expert.append(final_share)
#         self.text_share_expert = nn.ModuleList(text_share_expert)
#         self.image_share_expert = nn.ModuleList(image_share_expert)
#         self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
#         self.final_share_expert = nn.ModuleList(final_share_expert)

#         image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0,final_gate_list = [], [], [], [],[]
#         for i in range(self.domain_num):
#             image_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
#                                        nn.SiLU(),
#                                        #SimpleGate(),
#                                        #nn.BatchNorm1d(int(self.unified_dim / 2)),
#                                        nn.Linear(self.unified_dim, self.num_expert * 3),
#                                        nn.Dropout(0.1),
#                                        nn.Softmax(dim=1)
#                                        )
#             text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
#                                       nn.SiLU(),
#                                       #SimpleGate(),
#                                       #nn.BatchNorm1d(int(self.unified_dim / 2)),
#                                       nn.Linear(self.unified_dim, self.num_expert * 3),
#                                       nn.Dropout(0.1),
#                                       nn.Softmax(dim=1)
#                                       )
#             fusion_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
#                                         nn.SiLU(),
#                                         #SimpleGate(),
#                                         #nn.BatchNorm1d(int(self.unified_dim / 2)),
#                                         nn.Linear(self.unified_dim, self.num_expert * 4),
#                                         nn.Dropout(0.1),
#                                         nn.Softmax(dim=1)
#                                         )
#             fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
#                                          nn.SiLU(),
#                                          #SimpleGate(),
#                                          #nn.BatchNorm1d(80),
#                                          nn.Linear(160, self.num_expert * 3),
#                                          nn.Dropout(0.1),
#                                          nn.Softmax(dim=1)
#                                          )
#             final_gate = nn.Sequential(nn.Linear(320, 320),
#                                         nn.SiLU(),
#                                         #SimpleGate(),
#                                         #nn.BatchNorm1d(int(self.unified_dim / 2)),
#                                         nn.Linear(320, 160),
#                                         nn.SiLU(),
#                                         nn.Linear(160, self.num_expert * 3),
#                                         nn.Dropout(0.1),
#                                         nn.Softmax(dim=1)
#                                          )
#             image_gate_list.append(image_gate)
#             text_gate_list.append(text_gate)
#             fusion_gate_list.append(fusion_gate)
#             fusion_gate_list0.append(fusion_gate0)
#             final_gate_list.append(final_gate)
#         self.image_gate_list = nn.ModuleList(image_gate_list)
#         self.text_gate_list = nn.ModuleList(text_gate_list)
#         self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
#         self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
#         self.final_gate_list = nn.ModuleList(final_gate_list)

#         #self.text_attention = TokenAttention(self.unified_dim)
#         self.text_attention = MaskAttention(self.unified_dim)
#         self.image_attention = TokenAttention(self.unified_dim)
#         self.fusion_attention = TokenAttention(self.unified_dim * 2)
#         self.final_attention = TokenAttention(320)

#         self.text_classifier = MLP(320, mlp_dims, dropout)
#         self.text_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
#         self.image_classifier = MLP(320, mlp_dims, dropout)
#         self.image_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
#         self.fusion_classifier = MLP(320, mlp_dims, dropout)
#         self.fusion_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)

#         # self.max_classifier = MLP(640, mlp_dims, dropout)
#         self.max_classifier = MLP(320 * 1, mlp_dims, dropout)

#         share_classifier_list = []

#         for i in range(self.domain_num):
#             share_classifier = MLP(320, mlp_dims, dropout)
#             share_classifier_list.append(share_classifier)
#         self.share_classifier_list = nn.ModuleList(share_classifier_list)

#         dom_classifier_list = []

#         for i in range(self.domain_num):
#             dom_classifier = MLP(320, mlp_dims, dropout)
#             dom_classifier_list.append(dom_classifier)
#         self.dom_classifier_list = nn.ModuleList(dom_classifier_list)



#         final_classifier_list = []

#         for i in range(self.domain_num):
#             final_classifier = MLP(320, mlp_dims, dropout)
#             final_classifier_list.append(final_classifier)
#         self.final_classifier_list = nn.ModuleList(final_classifier_list)

#         self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
#         self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
#         self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
#         self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
#         self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
#         self.att_mlp_img = MLP_fusion(320, 2, [174], 0.1)
#         self.att_mlp_mm = MLP_fusion(320, 2, [174], 0.1)







#         self.model_size = "base"
#         self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
#         self.image_model.cuda()
#         checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
#         self.image_model.load_state_dict(checkpoint['model'], strict=False)
#         for param in self.image_model.parameters():
#             param.requires_grad = False

#         #### mapping MLPs
#         self.mapping_IS_MLP_mu = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_IS_MLP_sigma = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim,1),
#         )
#         self.mapping_T_MLP_mu = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_T_MLP_sigma = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_IP_MLP_mu = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_IP_MLP_sigma = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_CC_MLP_mu = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.mapping_CC_MLP_sigma = nn.Sequential(
#             nn.Linear(1, self.unified_dim),
#             nn.SiLU(),
#             # nn.BatchNorm1d(self.unified_dim),
#             nn.Linear(self.unified_dim, 1),
#         )
#         self.adaIN = AdaIN()
#         self.irrelevant_tensor = []
#         for i in range(self.domain_num):
#             self.irrelevant_tensor.append(nn.Parameter(torch.ones((1, 320)), requires_grad=True))

#         self.ClipModel,_ = load_from_name("ViT-B-16", device="cuda", download_root='./')


#         #pivot:
#         feature_emb_size = 320
#         img_emb_size =320
#         feature_num = 4
#         self.feature_num = 4
#         text_emb_size = 320
#         #self.n_node = 64
#         self.feature_emb_size = 320
#         self.emb_size = 320
#         self.layers = 12
#         self.transformers = torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
#                                                                   attention_dropout=0,
#                                                                   initializer_range=0.02) for _ in
#                                                  range(self.layers)])
#         self.mlp_img = torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                                  range(feature_num)])

#         self.mlp_text = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                             range(feature_num)])
#         self.pivot_mlp_fusion = torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                              range(feature_num)])
#         self.transformers_list = torch.nn.ModuleList()
#         self.mlp_img_list = torch.nn.ModuleList()
#         self.mlp_text_list = torch.nn.ModuleList()
#         self.pivot_mlp_fusion_list = torch.nn.ModuleList()
#         for i in range(self.domain_num):
#             self.transformers_list.append(torch.nn.ModuleList([TransformerLayer(feature_emb_size, head_num=4, dropout=0.6,
#                                                                   attention_dropout=0,
#                                                                   initializer_range=0.02) for _ in
#                                                  range(self.layers)]))
#             self.mlp_img_list.append(torch.nn.ModuleList([MLP_trans(img_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                                  range(feature_num)]))
#             self.mlp_text_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                             range(feature_num)]))
#             self.pivot_mlp_fusion_list.append(torch.nn.ModuleList([MLP_trans(text_emb_size, feature_emb_size, dropout=0.6) for _ in
#                                              range(feature_num)]))


#         self.active = nn.SiLU()
#         self.dropout2 = nn.Dropout(0.2)
#         self.mlp_star_f1 = nn.Linear(self.feature_emb_size * 4, self.emb_size)
#         self.mlp_star_f2 = nn.Linear(self.emb_size, self.emb_size)
#         self.mlp_star_f1_list = torch.nn.ModuleList()
#         self.mlp_star_f2_list = torch.nn.ModuleList()
#         for i in range(self.domain_num):
#             self.mlp_star_f1_list.append(nn.Linear(self.feature_emb_size * 4, self.emb_size))
#             self.mlp_star_f2_list.append(nn.Linear(self.emb_size, self.emb_size))


#         self.fake_news_layernorm = LayerNorm(320 * 3, eps=1e-12)
#         self.domain_classification_layernorm = LayerNorm(320 * 1, eps=1e-12)
#         self.gate_trans = nn.Sequential(
#                 nn.Linear(320 * 1, 1 * 320, bias=False),
#                 nn.GELU(),
#                 nn.Linear(1 * 320, 320 * 1, bias=False),
#                 nn.GELU(),
#                 )

#         self.query_text = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 1, bias=False)
#         )
#         self.query_image = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 1, bias=False)
#         )
#         self.query_fusion = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 1, bias=False)
#         )


#         self.softmax = nn.Softmax(dim=-1)

#         self.gate_image_prefer = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 320),
#             nn.Sigmoid()
#         )

#         self.gate_text_prefer = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 320),
#             nn.Sigmoid()
#         )

#         self.gate_fusion_prefer = nn.Sequential(
#             nn.Linear(320, 320),
#             nn.GELU(),
#             nn.Linear(320, 320),
#             nn.Sigmoid()
#         )

#         self.tau = 0.5




#     def forward(self, **kwargs):
#         inputs = kwargs['content']
#         masks = kwargs['content_masks']
#         text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
#         image = kwargs['image']
#         image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
#         clip_image = kwargs['clip_image']
#         clip_text = kwargs['clip_text']
#         with torch.no_grad(): 
#             clip_image_feature = self.ClipModel.encode_image(clip_image)  # ([64, 512])
#             clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
#             clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
#             clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)


#         clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)  # torch.Size([64, 1024])
#         clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())  # torch.Size([64, 320])


#         text_atn_feature = self.text_attention(text_feature, masks)
#         image_atn_feature, _ = self.image_attention(image_feature)
#         fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
#         fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
#         fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)


#         text_gate_input = text_atn_feature  # ([64, 1536])
#         image_gate_input = image_atn_feature
#         fusion_gate_input = fusion_atn_feature

#         text_gate_out_list = []
#         for i in range(self.domain_num):
#             gate_out = self.text_gate_list[i](text_gate_input)
#             text_gate_out_list.append(gate_out)
#         self.text_gate_out_list = text_gate_out_list


#         image_gate_out_list = []
#         for i in range(self.domain_num):
#             gate_out = self.image_gate_list[i](image_gate_input)
#             image_gate_out_list.append(gate_out)
#         self.image_gate_out_list = image_gate_out_list

#         fusion_gate_out_list = []
#         for i in range(self.domain_num):
#             gate_out = self.fusion_gate_list[i](fusion_gate_input)
#             fusion_gate_out_list.append(gate_out)
#         self.fusion_gate_out_list = fusion_gate_out_list

#         # 文本模态
#         text_gate_expert_value = []
#         text_experts_feature = 0
#         text_gate_share_expert_value = []
#         for i in range(1):
#             gate_expert = 0
#             gate_share_expert = 0
#             for j in range(self.num_expert):
#                 tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
#                 gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
#             for j in range(self.num_expert * 2):
#                 tmp_expert = self.text_share_expert[0][j](text_feature)
#                 gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
#                 gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
#             text_experts_feature = gate_expert
#             text_gate_share_expert_value.append(gate_share_expert)

#         att = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
#         text_experts_feature0 = att[:, 0].view(-1, 1)*text_experts_feature
#         text_experts_feature1 = att[:, 1].view(-1, 1)*text_experts_feature
#         text_gate_expert_value.append(text_experts_feature0)
#         text_gate_expert_value.append(text_experts_feature1)

        
#         # 图像模态
#         image_gate_expert_value = []
#         image_experts_feature = 0
#         image_gate_share_expert_value = []
#         for i in range(1):
#             gate_expert = 0
#             gate_share_expert = 0
#             for j in range(self.num_expert):
#                 tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
#                 gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
#             for j in range(self.num_expert * 2):
#                 tmp_expert = self.image_share_expert[0][j](image_feature)
#                 gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
#                 gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
#             image_experts_feature = gate_expert
#             image_gate_share_expert_value.append(gate_share_expert)

#         att = F.softmax(self.att_mlp_img(image_experts_feature), dim=-1)
#         image_experts_feature0 = att[:, 0].view(-1, 1)*image_experts_feature
#         image_experts_feature1 = att[:, 1].view(-1, 1)*image_experts_feature
#         image_gate_expert_value.append(image_experts_feature0)
#         image_gate_expert_value.append(image_experts_feature1)


#         # 融合模态
#         text = text_gate_share_expert_value[0]
#         image = image_gate_share_expert_value[0]
#         fusion_share_feature = torch.cat((clip_fusion_feature, text, image), dim=-1)

#         fusion_share_feature = self.MLP_fusion(fusion_share_feature)


#         fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
#         fusion_gate_out_list0 = []
#         for k in range(self.domain_num):
#             gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
#             fusion_gate_out_list0.append(gate_out)
#         self.fusion_gate_out_list0 = fusion_gate_out_list0


#         # 融合模态
#         fusion_gate_expert_value0 = []
#         fusion_experts_feature = 0
#         fusion_gate_share_expert_value0 = []
#         for m in range(1):
#             share_gate_expert0 = 0
#             gate_spacial_expert = 0
#             gate_share_expert = 0
#             for n in range(self.num_expert):
#                 fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
#                 share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
#             for n in range(self.num_expert * 2):
#                 fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
#                 share_gate_expert0 += (
#                             fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
#                 gate_share_expert += (
#                             fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
#             #fusion_gate_expert_value0.append(share_gate_expert0)
#             fusion_gate_share_expert_value0.append(gate_share_expert)
#             fusion_experts_feature = fusion_tmp_expert0

#         att = F.softmax(self.att_mlp_mm(fusion_experts_feature), dim=-1)
#         fusion_experts_feature0 = att[:, 0].view(-1, 1)*fusion_experts_feature
#         fusion_experts_feature1 = att[:, 1].view(-1, 1)*fusion_experts_feature
#         fusion_gate_expert_value0.append(fusion_experts_feature0)
#         fusion_gate_expert_value0.append(fusion_experts_feature1)


#         # 每个模态的特征
#         text_features = text_gate_expert_value[0]
#         image_features = image_gate_expert_value[0]
#         fusion_features = fusion_gate_expert_value0[0]

#         # 多视角 
#         text_fake_news_logits = self.text_classifier(text_features).squeeze(1)
#         image_fake_news_logits = self.image_classifier(image_features).squeeze(1)
#         fusion_fake_news_logits = self.fusion_classifier(fusion_features).squeeze(1)

#         text_fake_news = torch.sigmoid(text_fake_news_logits)
#         image_fake_news = torch.sigmoid(image_fake_news_logits)
#         fusion_fake_news = torch.sigmoid(fusion_fake_news_logits)


#         # 多模态融合
#         all_modility = text_features + image_features + fusion_features


#         # 虚假新闻检测任务经过 sigmoid
#         fake_news_sigmoid = torch.sigmoid(self.max_classifier(all_modility).squeeze(1))

#         return fake_news_sigmoid, text_fake_news, image_fake_news, fusion_fake_news





# class Trainer():
#     def __init__(self,
#                  emb_dim,
#                  mlp_dims,
#                  bert,
#                  use_cuda,
#                  lr,
#                  dropout,
#                  train_loader,
#                  val_loader,
#                  test_loader,
#                  category_dict,
#                  weight_decay,
#                  save_param_dir,
#                  loss_weight=[1, 0.006, 0.009, 5e-5],
#                  early_stop=5,
#                  epoches=100
#                  ):
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.val_loader = val_loader
#         self.early_stop = early_stop
#         self.epoches = epoches
#         self.category_dict = category_dict
#         self.loss_weight = loss_weight
#         self.use_cuda = use_cuda

#         self.emb_dim = emb_dim
#         self.mlp_dims = mlp_dims
#         self.bert = bert
#         self.dropout = dropout
#         if not os.path.exists(save_param_dir):
#             self.save_param_dir = os.makedirs(save_param_dir)
#         else:
#             self.save_param_dir = save_param_dir

#     def train(self):
#         self.model = MultiDomainPLEFENDModel(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
#         if self.use_cuda:
#             self.model = self.model.cuda()
#         loss_fn = torch.nn.BCELoss()
#         optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
#         recorder = Recorder(self.early_stop)
#         for epoch in range(self.epoches):
#             self.model.train()
#             train_data_iter = tqdm.tqdm(self.train_loader)
#             avg_loss = Averager()
#             for step_n, batch in enumerate(train_data_iter):
#                 batch_data = clipdata2gpu(batch)
#                 label = batch_data['label']

#                 label0,text_fake_news, image_fake_news, fusion_fake_news = self.model(**batch_data)
#                 loss0 = loss_fn(label0,label.float())


#                 # 虚假新闻检测的辅助任务
#                 loss12 = loss_fn(text_fake_news,label.float())
#                 loss22 = loss_fn(image_fake_news, label.float())
#                 loss32 = loss_fn(fusion_fake_news, label.float())
#                 loss = loss0+(loss12+loss22+loss32)/3

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 if (scheduler is not None):
#                     scheduler.step()
#                 avg_loss.add(loss.item())
#             print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
#             #results0,results1,results2,results3 = self.test(self.val_loader)
#             results0 = self.test(self.test_loader)
#             mark = recorder.add(results0)
#             if mark == 'save':
#                 torch.save(self.model.state_dict(),
#                            os.path.join(self.save_param_dir, 'parameter_clip111.pkl'))
#             elif mark == 'esc':
#                 break
#             else:
#                 continue
#         self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_clip111.pkl')))
#         print("开始进行最后的测试: ")
#         results0 = self.test(self.test_loader)
#         print("最后的结果", results0)

#         return results0, os.path.join(self.save_param_dir, 'parameter_clip111.pkl')


#     def test(self, dataloader):
#         pred = []
#         label = []
#         category = []
#         self.model.eval()
#         data_iter = tqdm.tqdm(dataloader)
#         for step_n, batch in enumerate(data_iter):
#             with torch.no_grad():
#                 batch_data = clipdata2gpu(batch)
#                 batch_label = batch_data['label']
#                 batch_category = batch_data['category']
#                 batch_label_pred,_,_,_ = self.model(**batch_data)

#                 label.extend(batch_label.detach().cpu().numpy().tolist())
#                 pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
#                 category.extend(batch_category.detach().cpu().numpy().tolist())

#         metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
#         return metric_res


# import os
# import tqdm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F # Ensure F is imported for F.softmax
# from transformers import BertModel, AutoTokenizer, AutoModel 
# import models_mae # Domain specific
# from utils.utils_weibo import clipdata2gpu, Averager, metricsTrueFalse, Recorder 
# from .layers import MLP, MaskAttention, TokenAttention, cnn_extractor, LayerNorm, MLP_Mu, MLP_fusion, clip_fuion 
# from .pivot import TransformerLayer, MLP_trans 
# from timm.models.vision_transformer import Block 

# try:
#     import cn_clip.clip as clip
#     from cn_clip.clip import load_from_name
# except ImportError:
#     print("Warning: cn_clip library not found. CLIP functionalities will not work.")
#     clip = None
#     load_from_name = None

# class SimpleGate(nn.Module):
#     def __init__(self, dim=1):
#         super(SimpleGate, self).__init__()
#         self.dim = dim

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=self.dim)
#         return x1 * x2

# class AdaIN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def mu(self, x):
#         return torch.sum(x,(1))/(x.shape[1] + 1e-6) 

#     def sigma(self, x):
#         return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1] + 1e-6)) 

#     def forward(self, x, mu, sigma):
#         x_mean = self.mu(x)
#         x_std = self.sigma(x)
#         x_reduce_mean = x.permute([1, 0]) - x_mean
#         x_norm = x_reduce_mean/(x_std + 1e-6) 
#         return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


# class MultiDomainPLEFENDModel(torch.nn.Module):
#     # Modified __init__ to match domain_weibo.txt and add reasoning params
#     def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout, 
#                  reasoning_emb_dim=768, num_manipulation_classes=0):
#         super(MultiDomainPLEFENDModel, self).__init__()
#         self.num_expert = 6
#         self.task_num = 2
#         self.domain_num = self.task_num 
#         # self.gate_num = 3 # From original, but not directly used by name later
#         self.num_share = 1
#         self.unified_dim, self.text_dim = emb_dim, 768
#         self.image_dim = 768 # MAE ViT-Base output is 768
        
#         self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        
#         feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
#         # self.text_token_len = 197 # From original
#         # self.image_token_len = 197 # From original

#         # --- Text Experts (Specific) ---
#         text_expert_list = []
#         for i in range(self.domain_num):
#             text_expert = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert)]
#             text_expert_list.append(nn.ModuleList(text_expert))
#         self.text_experts = nn.ModuleList(text_expert_list)

#         # --- Image Experts (Specific) ---
#         image_expert_list = []
#         for i in range(self.domain_num):
#             image_expert = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)]
#             image_expert_list.append(nn.ModuleList(image_expert))
#         self.image_experts = nn.ModuleList(image_expert_list)
        
#         # --- Fusion Experts (Specific) ---
#         fusion_expert_list = []
#         for i in range(self.domain_num):
#             fusion_expert = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert)]
#             fusion_expert_list.append(nn.ModuleList(fusion_expert))
#         self.fusion_experts = nn.ModuleList(fusion_expert_list)

#         # --- Final Experts (Specific) - Restored from domain_weibo.txt ---
#         final_expert_list = []
#         for i in range(self.domain_num):
#             final_expert = [Block(dim=320, num_heads=8) for _ in range(self.num_expert)] # Assuming Block is timm.models.vision_transformer.Block
#             final_expert_list.append(nn.ModuleList(final_expert))
#         self.final_experts = nn.ModuleList(final_expert_list)

#         # --- Shared Experts ---
#         text_share_expert_outer, image_share_expert_outer, fusion_share_expert_outer, final_share_expert_outer = [], [], [], []
#         for _ in range(self.num_share): 
#             text_share = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert * 2)]
#             image_share = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert * 2)]
#             fusion_share = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert*2)]
#             final_share = [Block(dim=320, num_heads=8) for _ in range(self.num_expert*2)] # Restored

#             text_share_expert_outer.append(nn.ModuleList(text_share))
#             image_share_expert_outer.append(nn.ModuleList(image_share))
#             fusion_share_expert_outer.append(nn.ModuleList(fusion_share))
#             final_share_expert_outer.append(nn.ModuleList(final_share)) # Restored

#         self.text_share_expert = nn.ModuleList(text_share_expert_outer)
#         self.image_share_expert = nn.ModuleList(image_share_expert_outer)
#         self.fusion_share_expert = nn.ModuleList(fusion_share_expert_outer)
#         self.final_share_expert = nn.ModuleList(final_share_expert_outer) # Restored

#         # --- Gating Networks ---
#         gate_output_dim_specific_plus_shared = self.num_expert * 3 # num_specific + num_shared_components (2 * num_expert for text/image/fusion)
#         gate_output_dim_fusion_original = self.num_expert * 4 # As per original domain_weibo.txt for self.fusion_gate_list

#         image_gate_list, text_gate_list, fusion_gate_list_original, fusion_gate_list0, final_gate_list = [], [], [], [], []
#         for _ in range(self.domain_num):
#             text_gate_list.append(nn.Sequential(
#                 nn.Linear(self.text_dim, self.text_dim), nn.SiLU(),
#                 nn.Linear(self.text_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             image_gate_list.append(nn.Sequential(
#                 nn.Linear(self.image_dim, self.image_dim), nn.SiLU(),
#                 nn.Linear(self.image_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             # This fusion_gate is from the original domain_weibo.txt, input self.unified_dim (768), output num_expert*4
#             fusion_gate_list_original.append(nn.Sequential( 
#                 nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
#                 nn.Linear(self.unified_dim, gate_output_dim_fusion_original), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             # This fusion_gate0 is for the MLP_fusion output (320 dim), output num_expert*3
#             fusion_gate_list0.append(nn.Sequential(
#                 nn.Linear(320, 160), nn.SiLU(),
#                 nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             # Final gate from original domain_weibo.txt
#             final_gate_list.append(nn.Sequential(
#                 nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 160), nn.SiLU(),
#                 nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))

#         self.text_gate_list = nn.ModuleList(text_gate_list)
#         self.image_gate_list = nn.ModuleList(image_gate_list)
#         self.fusion_gate_list = nn.ModuleList(fusion_gate_list_original) # Restored original fusion gate
#         self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0) # The one used after MLP_fusion
#         self.final_gate_list = nn.ModuleList(final_gate_list) # Restored

#         # --- Attention Mechanisms ---
#         self.text_attention = MaskAttention(self.text_dim) 
#         self.image_attention = TokenAttention(self.image_dim)
#         self.fusion_attention = TokenAttention(self.text_dim + self.image_dim) 
#         self.final_attention = TokenAttention(320) # Restored

#         # --- Classifiers ---
#         feature_dim_after_experts = 320
#         self.text_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         self.image_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         self.fusion_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         # self.max_classifier in original was after combining features from T,I,F experts
#         # The "final_classifier_list" from domain_weibo.txt was used for the output of "final_experts"
#         # Let's keep self.max_classifier for the sum of T,I,F before "final_experts"
#         self.max_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout) 

#         # Final classifier list from original domain_weibo.txt (applied after final_experts)
#         final_classifier_list = []
#         for i in range(self.domain_num):
#             final_classifier_list.append(MLP(feature_dim_after_experts, mlp_dims, dropout))
#         self.final_classifier_list = nn.ModuleList(final_classifier_list)


#         # --- Fusion MLPs ---
#         self.MLP_fusion = MLP_fusion(320 * 3, 320, [348], 0.1) 
#         self.domain_fusion = MLP_fusion(320, 320, [348], 0.1) 
#         self.MLP_fusion0 = MLP_fusion(self.text_dim + self.image_dim, self.text_dim, [348], 0.1)
        
#         if clip is not None:
#              self.clip_fusion = clip_fuion(1024, 320, [348], 0.1) 
#         else:
#             self.clip_fusion = None

#         self.att_mlp_text = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
#         self.att_mlp_img = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
#         self.att_mlp_mm = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1) # For fusion features

#         # --- Image Encoder (MAE) ---
#         self.model_size = "base" 
#         try:
#             self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
#             # Loading checkpoint and moving to CUDA as in original domain_weibo.txt
#             checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
#             self.image_model.load_state_dict(checkpoint['model'], strict=False)
#             if torch.cuda.is_available(): # Check if CUDA is available before moving
#                 self.image_model.cuda() 
#             for param in self.image_model.parameters():
#                 param.requires_grad = False
#         except Exception as e:
#             print(f"Warning: Could not load MAE model 'mae_vit_{self.model_size}_patch16'. Error: {e}")
#             self.image_model = None

#         # --- CLIP Model ---
#         if clip is not None:
#             try:
#                 # Loading with device="cuda" as in original, and download_root='./'
#                 clip_device = "cuda" if torch.cuda.is_available() else "cpu"
#                 self.ClipModel, _ = load_from_name("ViT-B-16", device=clip_device, download_root='./') 
#             except Exception as e:
#                 print(f"Warning: Could not load CLIP model. Error: {e}")
#                 self.ClipModel = None
#         else:
#             self.ClipModel = None
            
#         # --- NEW MODULES FOR REASONING DISTILLATION ---
#         self.reasoning_emb_dim = reasoning_emb_dim
#         self.num_manipulation_classes = num_manipulation_classes

#         # Input to these will be the feature that goes into the final original classifier.
#         # In domain_weibo.txt, this is the output of final_attention (dim 320).
#         self.project_reasoning_text = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)
#         self.project_reasoning_image = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)
#         self.project_reasoning_cross = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)

#         if self.num_manipulation_classes > 0:
#             self.manipulation_classifier = MLP(feature_dim_after_experts, [self.num_manipulation_classes], dropout)
#         else:
#             self.manipulation_classifier = None
            
#     def get_expert_output(self, features, gate_outputs, specific_experts, shared_experts, is_final_expert=False):
#         batch_size = features.size(0)
#         num_specific_experts = len(specific_experts) 
#         num_shared_experts_total = len(shared_experts[0]) # shared_experts is a ModuleList of ModuleList
        
#         # Output dimension is 320 for all cnn_extractor and Block experts
#         expert_output_dim = 320 
#         expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device) 
#         shared_expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)

#         for i in range(num_specific_experts):
#             expert_out = specific_experts[i](features) 
#             gate_val = gate_outputs[:, i].unsqueeze(1) 
#             expert_outputs_sum += expert_out * gate_val

#         for i in range(num_shared_experts_total):
#             expert_out = shared_experts[0][i](features) 
#             gate_val = gate_outputs[:, num_specific_experts + i].unsqueeze(1)
#             expert_outputs_sum += expert_out * gate_val
#             if not is_final_expert: # For text, image, fusion, also accumulate shared part
#                  shared_expert_outputs_sum += expert_out * gate_val 
            
#         return expert_outputs_sum, shared_expert_outputs_sum


#     def forward(self, **kwargs):
#         inputs = kwargs['content'] 
#         masks = kwargs['content_masks'] 
#         image_raw = kwargs['image']  # Renamed from 'image' to 'image_raw' for clarity with MAE
        
#         text_feature_full = self.bert(inputs, attention_mask=masks)[0]
        
#         if self.image_model:
#             # Using forward_ying as per domain_weibo.txt (assuming it's the correct method for that MAE version)
#             try:
#                 image_feature_full = self.image_model.forward_ying(image_raw) 
#             except AttributeError: # Fallback if 'forward_ying' doesn't exist
#                 print("Warning: 'forward_ying' not found in MAE model, trying 'forward_features'.")
#                 image_feature_full = self.image_model.forward_features(image_raw)
#         else: 
#             image_feature_full = torch.zeros_like(text_feature_full, device=text_feature_full.device)

#         clip_image_input = kwargs.get('clip_image') 
#         clip_text_input = kwargs.get('clip_text')   
        
#         clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device) 
#         if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
#             with torch.no_grad():
#                 clip_image_feature = self.ClipModel.encode_image(clip_image_input)  
#                 clip_text_feature = self.ClipModel.encode_text(clip_text_input)    
#                 clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
#                 clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
            
#             clip_concat_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1).float() 
#             clip_fusion_feature = self.clip_fusion(clip_concat_feature)  

#         text_atn_feature = self.text_attention(text_feature_full, masks) 
#         image_atn_feature, _ = self.image_attention(image_feature_full)    
        
#         # Fusion attention input from original domain_weibo.txt
#         fusion_feature_for_atn = torch.cat((image_feature_full, text_feature_full), dim=-1) 
#         fusion_atn_feature, _ = self.fusion_attention(fusion_feature_for_atn)  
#         fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature) # Projects to 768 dim

#         domain_idx = kwargs.get('domain_idx', 0) 
#         domain_idx = domain_idx % self.domain_num 

#         # Gate outputs
#         text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)    
#         image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)  
#         # Original fusion_gate_list takes fusion_atn_feature (768 dim)
#         # fusion_gate_out_original = self.fusion_gate_list[domain_idx](fusion_atn_feature) 
        
#         # --- Text Experts ---
#         text_experts_output, text_shared_output = self.get_expert_output(
#             text_feature_full, text_gate_out, 
#             self.text_experts[domain_idx], self.text_share_expert
#         ) 
#         text_att_split = F.softmax(self.att_mlp_text(text_experts_output), dim=-1)
#         text_final_feature = text_att_split[:, 0].unsqueeze(1) * text_experts_output 

#         # --- Image Experts ---
#         image_experts_output, image_shared_output = self.get_expert_output(
#             image_feature_full, image_gate_out,
#             self.image_experts[domain_idx], self.image_share_expert
#         ) 
#         image_att_split = F.softmax(self.att_mlp_img(image_experts_output), dim=-1)
#         image_final_feature = image_att_split[:, 0].unsqueeze(1) * image_experts_output 
        
#         # --- Fusion Experts (using MLP_fusion output as base) ---
#         concat_for_MLP_fusion = torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1)
#         fusion_share_base_feature = self.MLP_fusion(concat_for_MLP_fusion) 

#         fusion_gate_input_for_gate0 = self.domain_fusion(fusion_share_base_feature) 
#         fusion_gate_out0 = self.fusion_gate_list0[domain_idx](fusion_gate_input_for_gate0) 

#         fusion_experts_output, _ = self.get_expert_output( 
#             fusion_share_base_feature, fusion_gate_out0, # Input is fusion_share_base_feature
#             self.fusion_experts[domain_idx], self.fusion_share_expert, is_final_expert=True # No separate shared output needed here
#         ) 
#         fusion_att_split = F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)
#         fusion_final_feature = fusion_att_split[:, 0].unsqueeze(1) * fusion_experts_output 

#         # --- Predictions for auxiliary tasks (original) ---
#         text_fake_news_logits = self.text_classifier(text_final_feature).squeeze(1)
#         image_fake_news_logits = self.image_classifier(image_final_feature).squeeze(1)
#         fusion_fake_news_logits = self.fusion_classifier(fusion_final_feature).squeeze(1)

#         text_fake_news_prob = torch.sigmoid(text_fake_news_logits)
#         image_fake_news_prob = torch.sigmoid(image_fake_news_logits)
#         fusion_fake_news_prob = torch.sigmoid(fusion_fake_news_logits)

#         # --- Combined feature before final experts (as in original domain_weibo.txt) ---
#         # This 'all_modility' is used by self.max_classifier in original, which then might feed into final_experts
#         # Or, if final_experts path is separate, this is one main path.
#         # The original script's `all_modility` feeds self.max_classifier to get `fake_news_sigmoid`
#         # There isn't a separate "final_expert" path in the returns of forward in domain_weibo.txt.
#         # It seems `max_classifier` IS the final classifier in the original return structure.
#         # The `final_experts`, `final_gate_list`, `final_attention`, `final_classifier_list` were defined but not clearly used
#         # in the `forward` path of the provided `domain_weibo.txt` to produce the returned `fake_news_sigmoid`.
#         # For now, I will stick to the simpler combination that leads to the 4 returned values,
#         # and use its result for reasoning heads.
#         all_modality_combined = text_final_feature + image_final_feature + fusion_final_feature 
        
#         final_fake_news_logits = self.max_classifier(all_modality_combined).squeeze(1)
#         final_fake_news_prob = torch.sigmoid(final_fake_news_logits) # This is label0 in original Trainer

#         # --- Reasoning Module Predictions ---
#         # Use 'all_modality_combined' (dim 320) as input, as it's the penultimate combined feature.
#         pred_reasoning_text_emb = self.project_reasoning_text(all_modality_combined)
#         pred_reasoning_image_emb = self.project_reasoning_image(all_modality_combined)
#         pred_reasoning_cross_emb = self.project_reasoning_cross(all_modality_combined)

#         manipulation_pred_logits = None
#         if self.manipulation_classifier:
#             manipulation_pred_logits = self.manipulation_classifier(all_modality_combined)

#         return (final_fake_news_prob, # label0
#                 text_fake_news_prob, 
#                 image_fake_news_prob, 
#                 fusion_fake_news_prob,
#                 pred_reasoning_text_emb, 
#                 pred_reasoning_image_emb, 
#                 pred_reasoning_cross_emb,
#                 manipulation_pred_logits)


# class DOMAINTrainerWeibo():
#     # Modified __init__ to match domain_weibo.txt Trainer and add reasoning params
#     def __init__(self,
#                  emb_dim, mlp_dims, bert, use_cuda, lr, dropout, # bert instead of bert_path
#                  train_loader, val_loader, test_loader, category_dict,
#                  weight_decay, save_param_dir,
#                  reasoning_emb_dim=768, 
#                  num_manipulation_classes=0, 
#                  lambda_reasoning_align=0.1, 
#                  lambda_manipulation_predict=0.1, 
#                  # loss_weight=[1, 0.006, 0.009, 5e-5], # This was in original, can be added if used
#                  early_stop=5, epoches=100 # epoches instead of epochs
#                  ):
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.val_loader = val_loader 
#         self.early_stop = early_stop
#         self.epoches = epoches # Changed from epochs
#         self.category_dict = category_dict 
#         self.use_cuda = use_cuda

#         self.emb_dim = emb_dim
#         self.mlp_dims = mlp_dims
#         self.bert = bert # Changed from bert_path
#         self.dropout = dropout
#         self.save_param_dir = save_param_dir
#         if not os.path.exists(self.save_param_dir):
#             os.makedirs(self.save_param_dir, exist_ok=True)

#         self.reasoning_emb_dim = reasoning_emb_dim
#         self.num_manipulation_classes = num_manipulation_classes
#         self.lambda_reasoning_align = lambda_reasoning_align
#         self.lambda_manipulation_predict = lambda_manipulation_predict
        
#         self.model = MultiDomainPLEFENDModel(
#             emb_dim=self.emb_dim, 
#             mlp_dims=self.mlp_dims, 
#             bert=self.bert, # Pass bert
#             out_channels=320, # out_channels is in original model's __init__ signature
#             dropout=self.dropout,
#             reasoning_emb_dim=self.reasoning_emb_dim,
#             num_manipulation_classes=self.num_manipulation_classes
#         )

#         if self.use_cuda:
#             self.model = self.model.cuda()
#             # MAE and CLIP are moved to CUDA inside MultiDomainPLEFENDModel's __init__
#             # if torch.cuda.is_available() in that class.

#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss() 
#         if self.num_manipulation_classes > 0:
#             self.ce_loss = nn.CrossEntropyLoss() 

#         self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        
#         self.model_save_filename = 'parameter_clip111.pkl'


#     def train(self):
#         recorder = Recorder(self.early_stop) 

#         for epoch in range(self.epoches): # Changed from self.epochs
#             self.model.train()
#             train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epoches}")
#             avg_loss_epoch = Averager() 

#             for batch_idx, batch in enumerate(train_data_iter):
#                 # Original domain_weibo.txt used a generic data2gpu, assuming clipdata2gpu is the intended one
#                 batch_data = clipdata2gpu(batch, self.use_cuda) 
#                 labels_main_task = batch_data['label'].float()
                
#                 # Unpack all returned values from the model
#                 (final_pred_prob, aux_text_prob, aux_image_prob, aux_fusion_prob,
#                  pred_r_text_emb, pred_r_image_emb, pred_r_cross_emb,
#                  manip_pred_logits) = self.model(**batch_data)

#                 # Main task loss (as in original domain_weibo.txt, final_pred_prob is label0)
#                 loss0 = self.bce_loss(final_pred_prob, labels_main_task)

#                 # Auxiliary losses for individual modality predictions (as in original)
#                 loss12 = self.bce_loss(aux_text_prob, labels_main_task)
#                 loss22 = self.bce_loss(aux_image_prob, labels_main_task)
#                 loss32 = self.bce_loss(aux_fusion_prob, labels_main_task)
                
#                 loss_detection_tasks = loss0 + (loss12 + loss22 + loss32) / 3.0

#                 # Reasoning alignment loss 
#                 teacher_r_text_emb = batch_data.get('teacher_reasoning_text_emb')
#                 teacher_r_image_emb = batch_data.get('teacher_reasoning_image_emb')
#                 teacher_r_cross_emb = batch_data.get('teacher_reasoning_cross_emb')
                
#                 loss_align = torch.tensor(0.0, device=labels_main_task.device)
#                 if teacher_r_text_emb is not None and \
#                    teacher_r_image_emb is not None and \
#                    teacher_r_cross_emb is not None:
#                     loss_align_r_text = self.mse_loss(pred_r_text_emb, teacher_r_text_emb)
#                     loss_align_r_image = self.mse_loss(pred_r_image_emb, teacher_r_image_emb)
#                     loss_align_r_cross = self.mse_loss(pred_r_cross_emb, teacher_r_cross_emb)
#                     loss_align = (loss_align_r_text + loss_align_r_image + loss_align_r_cross) / 3.0
                
#                 # Manipulation pattern prediction loss
#                 loss_manip = torch.tensor(0.0, device=labels_main_task.device)
#                 if self.num_manipulation_classes > 0 and manip_pred_logits is not None:
#                     manip_labels = batch_data.get('manipulation_labels')
#                     if manip_labels is not None:
#                         loss_manip = self.ce_loss(manip_pred_logits, manip_labels.long()) 

#                 # Total combined loss
#                 total_loss = loss_detection_tasks + \
#                              self.lambda_reasoning_align * loss_align + \
#                              self.lambda_manipulation_predict * loss_manip
                
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()
                
#                 if self.scheduler: # Original had (scheduler is not None)
#                     self.scheduler.step()
                
#                 avg_loss_epoch.add(total_loss.item()) # Original used avg_loss
#                 train_data_iter.set_postfix(loss=avg_loss_epoch.item())

#             # Original print format: print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
#             print(f'Training Epoch {epoch+1}; Loss {avg_loss_epoch.item():.4f}; ')


#             val_results = self.test(self.test_loader) # Original used self.test_loader for validation
#             # print(f"Validation Results Epoch {epoch+1}: {val_results}") # This line can be added for more info

#             mark = recorder.add(val_results) 
#             if mark == 'save':
#                 param_path = os.path.join(self.save_param_dir, self.model_save_filename)
#                 torch.save(self.model.state_dict(), param_path)
#                 # print(f"Epoch {epoch+1}: Model saved to {param_path}") # Original didn't have this print
#             elif mark == 'esc':
#                 # print(f"Early stopping at epoch {epoch+1}.") # Original didn't have this
#                 break
#             # else: # Original had continue
#             #     continue 
        
#         final_model_path = os.path.join(self.save_param_dir, self.model_save_filename)
#         if os.path.exists(final_model_path): # Ensure model is loaded for final test
#             self.model.load_state_dict(torch.load(final_model_path))
#             # print(f"Loaded model from {final_model_path} for final testing.") # Original didn't have this
#         else:
#             print(f"Warning: Model file {final_model_path} not found for final testing. Using current model state.")

#         print("开始进行最后的测试: ") 
#         final_test_results = self.test(self.test_loader)
#         print("最后的结果", final_test_results)

#         return final_test_results, final_model_path


#     def test(self, dataloader):
#         self.model.eval()
#         all_preds = [] # Original: pred = []
#         all_labels = [] # Original: label = []
#         all_categories = [] # Original: category = []

#         data_iter_test = tqdm.tqdm(dataloader) # Original: data_iter = tqdm.tqdm(dataloader)
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(data_iter_test): # Original: step_n, batch
#                 # Original domain_weibo.txt used a generic data2gpu
#                 batch_data = clipdata2gpu(batch, self.use_cuda) 
#                 labels_main_task = batch_data['label'] # Original: batch_label
                
#                 # Model now returns more items, only the first is the main prediction prob
#                 final_pred_prob, _, _, _, _, _, _, _ = self.model(**batch_data) # Original: batch_label_pred,_,_,_

#                 all_preds.extend(final_pred_prob.cpu().numpy().tolist())
#                 all_labels.extend(labels_main_task.cpu().numpy().tolist())
#                 if 'category' in batch_data: # Original: batch_category = batch_data['category']
#                      all_categories.extend(batch_data['category'].cpu().numpy().tolist())

#         # Metrics calculation should remain the same using metricsTrueFalse
#         results = {}
#         try:
#             categories_to_pass = all_categories if all_categories else []
#             results = metricsTrueFalse(all_labels, all_preds, categories_to_pass, self.category_dict)
#         except Exception as e:
#             print(f"Error in metrics calculation: {e}. Returning basic accuracy.")
#             correct_predictions = sum(1 for p, l in zip(all_preds, all_labels) if (p > 0.5) == l)
#             accuracy = correct_predictions / len(all_labels) if len(all_labels) > 0 else 0.0
#             results = {
#                 "accuracy": accuracy, "f1": 0.0, "Macro_F1":0.0, "Macro_Acc":0.0, 
#                 "Macro_Pre":0.0, "Macro_Rec":0.0, "Fake_Acc":0.0, "Fake_F1":0.0, 
#                 "Fake_Pre":0.0, "Fake_Rec":0.0, "Real_Acc":0.0, "Real_F1":0.0, 
#                 "Real_Pre":0.0, "Real_Rec":0.0,
#             } 
#             print(f"Basic Accuracy: {accuracy:.4f}")
#         return results



# import os    weibo蒸馏损失
# import tqdm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel, AutoTokenizer, AutoModel 
# import models_mae
# from utils.utils_weibo import clipdata2gpu, Averager, metricsTrueFalse, Recorder 
# from .layers import MLP, MaskAttention, TokenAttention, cnn_extractor, LayerNorm, MLP_Mu, MLP_fusion, clip_fuion 
# from .pivot import TransformerLayer, MLP_trans 
# from timm.models.vision_transformer import Block 

# try:
#     import cn_clip.clip as clip
#     from cn_clip.clip import load_from_name
# except ImportError:
#     print("Warning: cn_clip library not found. CLIP functionalities will not work.")
#     clip = None
#     load_from_name = None

# class SimpleGate(nn.Module):
#     def __init__(self, dim=1):
#         super(SimpleGate, self).__init__()
#         self.dim = dim

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=self.dim)
#         return x1 * x2

# class AdaIN(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def mu(self, x):
#         return torch.sum(x,(1))/(x.shape[1] + 1e-6) 

#     def sigma(self, x):
#         return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1] + 1e-6)) 

#     def forward(self, x, mu, sigma):
#         x_mean = self.mu(x)
#         x_std = self.sigma(x)
#         x_reduce_mean = x.permute([1, 0]) - x_mean
#         x_norm = x_reduce_mean/(x_std + 1e-6) 
#         return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])


# class MultiDomainPLEFENDModel(torch.nn.Module):
#     def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout, 
#                  reasoning_emb_dim=768, num_manipulation_classes=0):
#         super(MultiDomainPLEFENDModel, self).__init__()
#         self.num_expert = 6
#         self.task_num = 2
#         self.domain_num = self.task_num 
#         self.num_share = 1
#         self.unified_dim, self.text_dim = emb_dim, 768
#         self.image_dim = 768
        
#         self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        
#         feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

#         # --- Text Experts (Specific) ---
#         text_expert_list = []
#         for i in range(self.domain_num):
#             text_expert = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert)]
#             text_expert_list.append(nn.ModuleList(text_expert))
#         self.text_experts = nn.ModuleList(text_expert_list)

#         # --- Image Experts (Specific) ---
#         image_expert_list = []
#         for i in range(self.domain_num):
#             image_expert = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)]
#             image_expert_list.append(nn.ModuleList(image_expert))
#         self.image_experts = nn.ModuleList(image_expert_list)
        
#         # --- Fusion Experts (Specific) ---
#         fusion_expert_list = []
#         for i in range(self.domain_num):
#             fusion_expert = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert)]
#             fusion_expert_list.append(nn.ModuleList(fusion_expert))
#         self.fusion_experts = nn.ModuleList(fusion_expert_list)

#         # --- Final Experts (Specific) ---
#         final_expert_list = []
#         for i in range(self.domain_num):
#             final_expert = [Block(dim=320, num_heads=8) for _ in range(self.num_expert)]
#             final_expert_list.append(nn.ModuleList(final_expert))
#         self.final_experts = nn.ModuleList(final_expert_list)

#         # --- Shared Experts ---
#         text_share_expert_outer, image_share_expert_outer, fusion_share_expert_outer, final_share_expert_outer = [], [], [], []
#         for _ in range(self.num_share): 
#             text_share = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert * 2)]
#             image_share = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert * 2)]
#             fusion_share = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert*2)]
#             final_share = [Block(dim=320, num_heads=8) for _ in range(self.num_expert*2)]

#             text_share_expert_outer.append(nn.ModuleList(text_share))
#             image_share_expert_outer.append(nn.ModuleList(image_share))
#             fusion_share_expert_outer.append(nn.ModuleList(fusion_share))
#             final_share_expert_outer.append(nn.ModuleList(final_share))

#         self.text_share_expert = nn.ModuleList(text_share_expert_outer)
#         self.image_share_expert = nn.ModuleList(image_share_expert_outer)
#         self.fusion_share_expert = nn.ModuleList(fusion_share_expert_outer)
#         self.final_share_expert = nn.ModuleList(final_share_expert_outer)

#         # --- Gating Networks ---
#         gate_output_dim_specific_plus_shared = self.num_expert * 3
#         gate_output_dim_fusion_original = self.num_expert * 4

#         image_gate_list, text_gate_list, fusion_gate_list_original, fusion_gate_list0, final_gate_list = [], [], [], [], []
#         for _ in range(self.domain_num):
#             text_gate_list.append(nn.Sequential(
#                 nn.Linear(self.text_dim, self.text_dim), nn.SiLU(),
#                 nn.Linear(self.text_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             image_gate_list.append(nn.Sequential(
#                 nn.Linear(self.image_dim, self.image_dim), nn.SiLU(),
#                 nn.Linear(self.image_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             fusion_gate_list_original.append(nn.Sequential( 
#                 nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
#                 nn.Linear(self.unified_dim, gate_output_dim_fusion_original), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             fusion_gate_list0.append(nn.Sequential(
#                 nn.Linear(320, 160), nn.SiLU(),
#                 nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))
#             final_gate_list.append(nn.Sequential(
#                 nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 160), nn.SiLU(),
#                 nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
#             ))

#         self.text_gate_list = nn.ModuleList(text_gate_list)
#         self.image_gate_list = nn.ModuleList(image_gate_list)
#         self.fusion_gate_list = nn.ModuleList(fusion_gate_list_original)
#         self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
#         self.final_gate_list = nn.ModuleList(final_gate_list)

#         # --- Attention Mechanisms ---
#         self.text_attention = MaskAttention(self.text_dim) 
#         self.image_attention = TokenAttention(self.image_dim)
#         self.fusion_attention = TokenAttention(self.text_dim + self.image_dim) 
#         self.final_attention = TokenAttention(320)

#         # --- Classifiers ---
#         feature_dim_after_experts = 320
#         self.text_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         self.image_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         self.fusion_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
#         self.max_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout) 

#         final_classifier_list = []
#         for i in range(self.domain_num):
#             final_classifier_list.append(MLP(feature_dim_after_experts, mlp_dims, dropout))
#         self.final_classifier_list = nn.ModuleList(final_classifier_list)


#         # --- Fusion MLPs ---
#         self.MLP_fusion = MLP_fusion(320 * 3, 320, [348], 0.1) 
#         self.domain_fusion = MLP_fusion(320, 320, [348], 0.1) 
#         self.MLP_fusion0 = MLP_fusion(self.text_dim + self.image_dim, self.text_dim, [348], 0.1)
        
#         if clip is not None:
#              self.clip_fusion = clip_fuion(1024, 320, [348], 0.1) 
#         else:
#             self.clip_fusion = None

#         self.att_mlp_text = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
#         self.att_mlp_img = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
#         self.att_mlp_mm = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)

#         # --- Image Encoder (MAE) ---
#         self.model_size = "base" 
#         try:
#             self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
#             checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
#             self.image_model.load_state_dict(checkpoint['model'], strict=False)
#             if torch.cuda.is_available():
#                 self.image_model.cuda() 
#             for param in self.image_model.parameters():
#                 param.requires_grad = False
#         except Exception as e:
#             print(f"Warning: Could not load MAE model 'mae_vit_{self.model_size}_patch16'. Error: {e}")
#             self.image_model = None

#         # --- CLIP Model ---
#         if clip is not None:
#             try:
#                 clip_device = "cuda" if torch.cuda.is_available() else "cpu"
#                 self.ClipModel, _ = load_from_name("ViT-B-16", device=clip_device, download_root='./') 
#             except Exception as e:
#                 print(f"Warning: Could not load CLIP model. Error: {e}")
#                 self.ClipModel = None
#         else:
#             self.ClipModel = None
            
#         # --- 推理嵌入投影层（学生模型）---
#         self.reasoning_emb_dim = reasoning_emb_dim
#         self.num_manipulation_classes = num_manipulation_classes

#         # 学生模型的文本/图像/图文交叉推理嵌入输出
#         self.project_reasoning_text = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)  # 文本推理嵌入
#         self.project_reasoning_image = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)  # 图像推理嵌入
#         self.project_reasoning_cross = MLP(feature_dim_after_experts, [self.reasoning_emb_dim], dropout)  # 图文交叉推理嵌入

#         if self.num_manipulation_classes > 0:
#             self.manipulation_classifier = MLP(feature_dim_after_experts, [self.num_manipulation_classes], dropout)
#         else:
#             self.manipulation_classifier = None
            
#     def get_expert_output(self, features, gate_outputs, specific_experts, shared_experts, is_final_expert=False):
#         batch_size = features.size(0)
#         num_specific_experts = len(specific_experts) 
#         num_shared_experts_total = len(shared_experts[0]) 
        
#         expert_output_dim = 320 
#         expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device) 
#         shared_expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)

#         for i in range(num_specific_experts):
#             expert_out = specific_experts[i](features) 
#             gate_val = gate_outputs[:, i].unsqueeze(1) 
#             expert_outputs_sum += expert_out * gate_val

#         for i in range(num_shared_experts_total):
#             expert_out = shared_experts[0][i](features) 
#             gate_val = gate_outputs[:, num_specific_experts + i].unsqueeze(1)
#             expert_outputs_sum += expert_out * gate_val
#             if not is_final_expert:
#                  shared_expert_outputs_sum += expert_out * gate_val 
            
#         return expert_outputs_sum, shared_expert_outputs_sum


#     def forward(self, **kwargs):
#         inputs = kwargs['content'] 
#         masks = kwargs['content_masks'] 
#         image_raw = kwargs['image']  
        
#         text_feature_full = self.bert(inputs, attention_mask=masks)[0]
        
#         if self.image_model:
#             try:
#                 image_feature_full = self.image_model.forward_ying(image_raw) 
#             except AttributeError:
#                 print("Warning: 'forward_ying' not found in MAE model, trying 'forward_features'.")
#                 image_feature_full = self.image_model.forward_features(image_raw)
#         else: 
#             image_feature_full = torch.zeros_like(text_feature_full, device=text_feature_full.device)

#         clip_image_input = kwargs.get('clip_image') 
#         clip_text_input = kwargs.get('clip_text')   
        
#         clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device) 
#         if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
#             with torch.no_grad():
#                 clip_image_feature = self.ClipModel.encode_image(clip_image_input)  
#                 clip_text_feature = self.ClipModel.encode_text(clip_text_input)    
#                 clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
#                 clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
            
#             clip_concat_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1).float() 
#             clip_fusion_feature = self.clip_fusion(clip_concat_feature)  

#         text_atn_feature = self.text_attention(text_feature_full, masks) 
#         image_atn_feature, _ = self.image_attention(image_feature_full)    
        
#         fusion_feature_for_atn = torch.cat((image_feature_full, text_feature_full), dim=-1) 
#         fusion_atn_feature, _ = self.fusion_attention(fusion_feature_for_atn)  
#         fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)

#         domain_idx = kwargs.get('domain_idx', 0) 
#         domain_idx = domain_idx % self.domain_num 

#         # Gate outputs
#         text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)    
#         image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)  

#         # --- Text Experts ---
#         text_experts_output, text_shared_output = self.get_expert_output(
#             text_feature_full, text_gate_out, 
#             self.text_experts[domain_idx], self.text_share_expert
#         ) 
#         text_att_split = F.softmax(self.att_mlp_text(text_experts_output), dim=-1)
#         text_final_feature = text_att_split[:, 0].unsqueeze(1) * text_experts_output 

#         # --- Image Experts ---
#         image_experts_output, image_shared_output = self.get_expert_output(
#             image_feature_full, image_gate_out,
#             self.image_experts[domain_idx], self.image_share_expert
#         ) 
#         image_att_split = F.softmax(self.att_mlp_img(image_experts_output), dim=-1)
#         image_final_feature = image_att_split[:, 0].unsqueeze(1) * image_experts_output 
        
#         # --- Fusion Experts ---
#         concat_for_MLP_fusion = torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1)
#         fusion_share_base_feature = self.MLP_fusion(concat_for_MLP_fusion) 

#         fusion_gate_input_for_gate0 = self.domain_fusion(fusion_share_base_feature) 
#         fusion_gate_out0 = self.fusion_gate_list0[domain_idx](fusion_gate_input_for_gate0) 

#         fusion_experts_output, _ = self.get_expert_output( 
#             fusion_share_base_feature, fusion_gate_out0,
#             self.fusion_experts[domain_idx], self.fusion_share_expert, is_final_expert=True
#         ) 
#         fusion_att_split = F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)
#         fusion_final_feature = fusion_att_split[:, 0].unsqueeze(1) * fusion_experts_output 

#         # --- 分类预测 ---
#         text_fake_news_logits = self.text_classifier(text_final_feature).squeeze(1)
#         image_fake_news_logits = self.image_classifier(image_final_feature).squeeze(1)
#         fusion_fake_news_logits = self.fusion_classifier(fusion_final_feature).squeeze(1)

#         text_fake_news_prob = torch.sigmoid(text_fake_news_logits)
#         image_fake_news_prob = torch.sigmoid(image_fake_news_logits)
#         fusion_fake_news_prob = torch.sigmoid(fusion_fake_news_logits)

#         # --- 最终分类特征 ---
#         all_modality_combined = text_final_feature + image_final_feature + fusion_final_feature 
#         final_fake_news_logits = self.max_classifier(all_modality_combined).squeeze(1)
#         final_fake_news_prob = torch.sigmoid(final_fake_news_logits)

#         # --- 学生模型推理嵌入（用于蒸馏）---
#         pred_reasoning_text_emb = self.project_reasoning_text(all_modality_combined)  # 文本推理嵌入
#         pred_reasoning_image_emb = self.project_reasoning_image(all_modality_combined)  # 图像推理嵌入
#         pred_reasoning_cross_emb = self.project_reasoning_cross(all_modality_combined)  # 图文交叉推理嵌入

#         manipulation_pred_logits = None
#         if self.manipulation_classifier:
#             manipulation_pred_logits = self.manipulation_classifier(all_modality_combined)

#         return (final_fake_news_prob, 
#                 text_fake_news_prob, 
#                 image_fake_news_prob, 
#                 fusion_fake_news_prob,
#                 pred_reasoning_text_emb,  # 学生文本推理嵌入
#                 pred_reasoning_image_emb,  # 学生图像推理嵌入
#                 pred_reasoning_cross_emb,  # 学生图文交叉推理嵌入
#                 manipulation_pred_logits)


# class DOMAINTrainerWeibo():
#     def __init__(self,
#                  emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
#                  train_loader, val_loader, test_loader, category_dict,
#                  weight_decay, save_param_dir,
#                  reasoning_emb_dim=768, 
#                  num_manipulation_classes=0, 
#                  lambda_reasoning_align=0.1,  # 原有推理对齐损失权重（若保留）
#                  lambda_manipulation_predict=0.1,
#                  lambda_distill=0.5,  # 蒸馏损失权重（新增）
#                  early_stop=5, epoches=100
#                  ):
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.val_loader = val_loader 
#         self.early_stop = early_stop
#         self.epoches = epoches
#         self.category_dict = category_dict 
#         self.use_cuda = use_cuda

#         self.emb_dim = emb_dim
#         self.mlp_dims = mlp_dims
#         self.bert = bert
#         self.dropout = dropout
#         self.save_param_dir = save_param_dir
#         if not os.path.exists(self.save_param_dir):
#             os.makedirs(self.save_param_dir, exist_ok=True)

#         self.reasoning_emb_dim = reasoning_emb_dim
#         self.num_manipulation_classes = num_manipulation_classes
#         self.lambda_reasoning_align = lambda_reasoning_align
#         self.lambda_manipulation_predict = lambda_manipulation_predict
#         self.lambda_distill = lambda_distill  # 蒸馏损失权重（新增）
        
#         self.model = MultiDomainPLEFENDModel(
#             emb_dim=self.emb_dim, 
#             mlp_dims=self.mlp_dims, 
#             bert=self.bert,
#             out_channels=320,
#             dropout=self.dropout,
#             reasoning_emb_dim=self.reasoning_emb_dim,
#             num_manipulation_classes=self.num_manipulation_classes
#         )

#         if self.use_cuda:
#             self.model = self.model.cuda()

#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()  # 用于蒸馏损失（学生与教师嵌入的均方误差）
#         if self.num_manipulation_classes > 0:
#             self.ce_loss = nn.CrossEntropyLoss() 

#         self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        
#         self.model_save_filename = 'parameter_clip111.pkl'


#     def train(self):
#         recorder = Recorder(self.early_stop) 

#         for epoch in range(self.epoches):
#             self.model.train()
#             train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epoches}")
#             avg_loss_epoch = Averager() 

#             for batch_idx, batch in enumerate(train_data_iter):
#                 batch_data = clipdata2gpu(batch, self.use_cuda) 
#                 labels_main_task = batch_data['label'].float()
                
#                 # 模型输出（包含学生推理嵌入）
#                 (final_pred_prob, aux_text_prob, aux_image_prob, aux_fusion_prob,
#                  pred_r_text_emb, pred_r_image_emb, pred_r_cross_emb,
#                  manip_pred_logits) = self.model(** batch_data)

#                 # --- 1. 分类损失（原有检测任务损失）---
#                 loss0 = self.bce_loss(final_pred_prob, labels_main_task)
#                 loss12 = self.bce_loss(aux_text_prob, labels_main_task)
#                 loss22 = self.bce_loss(aux_image_prob, labels_main_task)
#                 loss32 = self.bce_loss(aux_fusion_prob, labels_main_task)
#                 loss_detection_tasks = loss0 + (loss12 + loss22 + loss32) / 3.0

#                 # --- 2. 蒸馏损失（学生嵌入与教师嵌入的MSE）---
#                 # 从batch中获取教师推理嵌入（来自CSV）
#                 teacher_text_emb = batch_data['teacher_text_emb']  # 教师文本推理嵌入
#                 teacher_image_emb = batch_data['teacher_image_emb']  # 教师图像推理嵌入
#                 teacher_cross_emb = batch_data['teacher_cross_emb']  # 教师图文交叉推理嵌入
                
#                 # 确保教师嵌入维度与学生一致（转为张量并移动到设备）
#                 teacher_text_emb = teacher_text_emb.to(pred_r_text_emb.device)
#                 teacher_image_emb = teacher_image_emb.to(pred_r_image_emb.device)
#                 teacher_cross_emb = teacher_cross_emb.to(pred_r_cross_emb.device)
                
#                 # 计算各模态蒸馏损失
#                 loss_distill_text = self.mse_loss(pred_r_text_emb.squeeze(1), teacher_text_emb)  # 文本蒸馏损失
#                 loss_distill_image = self.mse_loss(pred_r_image_emb.squeeze(1), teacher_image_emb)  # 图像蒸馏损失
#                 loss_distill_cross = self.mse_loss(pred_r_cross_emb.squeeze(1), teacher_cross_emb)  # 图文交叉蒸馏损失
#                 loss_distill = (loss_distill_text + loss_distill_image + loss_distill_cross) / 3.0  # 平均蒸馏损失

#                 # --- 3. 其他原有损失（若保留）---
#                 loss_align = torch.tensor(0.0, device=labels_main_task.device)
#                 teacher_r_text_emb = batch_data.get('teacher_reasoning_text_emb')
#                 teacher_r_image_emb = batch_data.get('teacher_reasoning_image_emb')
#                 teacher_r_cross_emb = batch_data.get('teacher_reasoning_cross_emb')
#                 if teacher_r_text_emb is not None and teacher_r_image_emb is not None and teacher_r_cross_emb is not None:
#                     loss_align_r_text = self.mse_loss(pred_r_text_emb, teacher_r_text_emb)
#                     loss_align_r_image = self.mse_loss(pred_r_image_emb, teacher_r_image_emb)
#                     loss_align_r_cross = self.mse_loss(pred_r_cross_emb, teacher_r_cross_emb)
#                     loss_align = (loss_align_r_text + loss_align_r_image + loss_align_r_cross) / 3.0
                
#                 loss_manip = torch.tensor(0.0, device=labels_main_task.device)
#                 if self.num_manipulation_classes > 0 and manip_pred_logits is not None:
#                     manip_labels = batch_data.get('manipulation_labels')
#                     if manip_labels is not None:
#                         loss_manip = self.ce_loss(manip_pred_logits, manip_labels.long()) 

#                 # --- 4. 总损失 = 分类损失 + 蒸馏损失（+ 其他损失）---
#                 total_loss = loss_detection_tasks + \
#                              self.lambda_distill * loss_distill +  # 新增蒸馏损失
#                              self.lambda_reasoning_align * loss_align + \
#                              self.lambda_manipulation_predict * loss_manip
                
#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()
                
#                 if self.scheduler:
#                     self.scheduler.step()
                
#                 avg_loss_epoch.add(total_loss.item())
#                 train_data_iter.set_postfix(loss=avg_loss_epoch.item())

#             print(f'Training Epoch {epoch+1}; Loss {avg_loss_epoch.item():.4f}; ')

#             val_results = self.test(self.test_loader)
#             mark = recorder.add(val_results) 
#             if mark == 'save':
#                 param_path = os.path.join(self.save_param_dir, self.model_save_filename)
#                 torch.save(self.model.state_dict(), param_path)
#             elif mark == 'esc':
#                 break
        
#         final_model_path = os.path.join(self.save_param_dir, self.model_save_filename)
#         if os.path.exists(final_model_path):
#             self.model.load_state_dict(torch.load(final_model_path))
#         else:
#             print(f"Warning: Model file {final_model_path} not found for final testing.")

#         print("开始进行最后的测试: ") 
#         final_test_results = self.test(self.test_loader)
#         print("最后的结果", final_test_results)

#         return final_test_results, final_model_path


#     def test(self, dataloader):
#         self.model.eval()
#         all_preds = []
#         all_labels = []
#         all_categories = []

#         data_iter_test = tqdm.tqdm(dataloader)
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(data_iter_test):
#                 batch_data = clipdata2gpu(batch, self.use_cuda) 
#                 labels_main_task = batch_data['label']
                
#                 final_pred_prob, _, _, _, _, _, _, _ = self.model(**batch_data)

#                 all_preds.extend(final_pred_prob.cpu().numpy().tolist())
#                 all_labels.extend(labels_main_task.cpu().numpy().tolist())
#                 if 'category' in batch_data:
#                      all_categories.extend(batch_data['category'].cpu().numpy().tolist())

#         results = {}
#         try:
#             categories_to_pass = all_categories if all_categories else []
#             results = metricsTrueFalse(all_labels, all_preds, categories_to_pass, self.category_dict)
#         except Exception as e:
#             print(f"Error in metrics calculation: {e}. Returning basic accuracy.")
#             correct_predictions = sum(1 for p, l in zip(all_preds, all_labels) if (p > 0.5) == l)
#             accuracy = correct_predictions / len(all_labels) if len(all_labels) > 0 else 0.0
#             results = {
#                 "accuracy": accuracy, "f1": 0.0, "Macro_F1":0.0, "Macro_Acc":0.0, 
#                 "Macro_Pre":0.0, "Macro_Rec":0.0, "Fake_Acc":0.0, "Fake_F1":0.0, 
#                 "Fake_Pre":0.0, "Fake_Rec":0.0, "Real_Acc":0.0, "Real_F1":0.0, 
#                 "Real_Pre":0.0, "Real_Rec":0.0,
#             } 
#             print(f"Basic Accuracy: {accuracy:.4f}")
#         return results
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import models_mae
from utils.utils_weibo import clipdata2gpu, Averager, metricsTrueFalse, Recorder
from .layers import MLP, MaskAttention, TokenAttention, cnn_extractor, LayerNorm, MLP_fusion, clip_fuion
from timm.models.vision_transformer import Block

try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
except ImportError:
    print("Warning: cn_clip library not found. CLIP functionalities will not work.")
    clip = None
    load_from_name = None


# =========================
# Helper: 统一解包模型输出
# =========================
def _unpack_outputs(outs):
    """
    将模型 forward 的输出统一成 11 元组：
      (final_prob, text_prob, image_prob, fusion_prob,
       text_fusion_prob, image_fusion_prob,
       final_logits, text_logits, image_logits,
       fusion_logits, aux_dict 或 manip_logits)

    兼容三种返回格式：
    1) 11 项 tuple/list（完整版接口）：
       (final_prob, text_prob, image_prob, fusion_prob,
        text_fusion_prob, image_fusion_prob,
        final_logits, text_logits, image_logits,
        fusion_logits, aux_dict)

    2) 5 项 tuple/list（轻量版接口）：
       (final_prob, text_prob, image_prob, fusion_prob, final_logits_or_manip_logits)
       —— 为保持兼容，这里把第 5 项当作“manip_logits/aux”的位置返还在第 11 位，
          第 7~10 位的各类 logits 设为 None

    3) dict：按以下键名提取，不存在的键补 None/{}：
       final_prob, text_prob, image_prob, fusion_prob,
       text_fusion_prob, image_fusion_prob,
       final_logits, text_logits, image_logits, fusion_logits, aux_dict(或 manip_logits)
    """
    # 字典格式：按键提取
    if isinstance(outs, dict):
        final_prob         = outs.get("final_prob")
        text_prob          = outs.get("text_prob")
        image_prob         = outs.get("image_prob")
        fusion_prob        = outs.get("fusion_prob")
        text_fusion_prob   = outs.get("text_fusion_prob")
        image_fusion_prob  = outs.get("image_fusion_prob")
        final_logits       = outs.get("final_logits")
        text_logits        = outs.get("text_logits")
        image_logits       = outs.get("image_logits")
        fusion_logits      = outs.get("fusion_logits")
        aux_dict_or_manip  = outs.get("aux_dict", outs.get("manip_logits", {}))
        if aux_dict_or_manip is None:
            aux_dict_or_manip = {}
        return (final_prob, text_prob, image_prob, fusion_prob,
                text_fusion_prob, image_fusion_prob,
                final_logits, text_logits, image_logits, fusion_logits,
                aux_dict_or_manip)

    # 非 tuple/list：包装
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)

    # 11 项：原样
    if len(outs) == 11:
        return tuple(outs)

    # 5 项：补齐剩余位
    if len(outs) == 5:
        final_prob, text_prob, image_prob, fusion_prob, last5 = outs
        # 这里把 7~10（各 logits）设为 None，最后一位放“manip_logits/aux”
        return (final_prob, text_prob, image_prob, fusion_prob,
                None, None,              # text_fusion_prob, image_fusion_prob
                None, None, None, None,  # final_logits, text_logits, image_logits, fusion_logits
                last5)                   # manip_logits or aux

    raise ValueError(f"Unexpected number of outputs from model: {len(outs)}")


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout,
                 reasoning_emb_dim=768, num_manipulation_classes=0):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768

        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        # --- Text Experts (Specific) ---
        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert)]
            text_expert_list.append(nn.ModuleList(text_expert))
        self.text_experts = nn.ModuleList(text_expert_list)

        # --- Image Experts (Specific) ---
        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)]
            image_expert_list.append(nn.ModuleList(image_expert))
        self.image_experts = nn.ModuleList(image_expert_list)

        # --- Fusion Experts (Specific) ---
        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert)]
            fusion_expert_list.append(nn.ModuleList(fusion_expert))
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        # --- Shared Experts ---
        text_share_expert_outer, image_share_expert_outer, fusion_share_expert_outer = [], [], []
        for _ in range(self.num_share):
            text_share = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert * 2)]
            image_share = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert * 2)]
            fusion_share = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert*2)]
            text_share_expert_outer.append(nn.ModuleList(text_share))
            image_share_expert_outer.append(nn.ModuleList(image_share))
            fusion_share_expert_outer.append(nn.ModuleList(fusion_share))
        self.text_share_expert = nn.ModuleList(text_share_expert_outer)
        self.image_share_expert = nn.ModuleList(image_share_expert_outer)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert_outer)

        # --- Gating Networks (旧版结构) ---
        gate_output_dim_specific_plus_shared = self.num_expert * 3
        image_gate_list, text_gate_list, fusion_gate_list0 = [], [], []
        for _ in range(self.domain_num):
            text_gate_list.append(nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim), nn.SiLU(),
                nn.Linear(self.text_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            image_gate_list.append(nn.Sequential(
                nn.Linear(self.image_dim, self.image_dim), nn.SiLU(),
                nn.Linear(self.image_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            fusion_gate_list0.append(nn.Sequential(
                nn.Linear(320, 160), nn.SiLU(),
                nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)

        # --- Attention Mechanisms ---
        self.text_attention = MaskAttention(self.text_dim)
        self.image_attention = TokenAttention(self.image_dim)

        # --- Classifiers ---
        feature_dim_after_experts = 320
        self.text_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.image_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.fusion_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.max_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)

        # --- Fusion MLPs ---
        self.MLP_fusion = MLP_fusion(320 * 3, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        if clip is not None:
            self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        else:
            self.clip_fusion = None
        self.att_mlp_text = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)

        # --- Image Encoder (MAE) ---
        self.model_size = "base"
        try:
            self.image_model = models_mae.__dict__[f"mae_vit_{self.model_size}_patch16"](norm_pix_loss=False)
            checkpoint = torch.load(f'./mae_pretrain_vit_{self.model_size}.pth', map_location='cpu')
            self.image_model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                self.image_model.cuda()
            for param in self.image_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load MAE model. Error: {e}")
            self.image_model = None

        # --- CLIP Model ---
        if clip is not None:
            try:
                clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.ClipModel, _ = load_from_name("ViT-B-16", device=clip_device, download_root='./')
            except Exception as e:
                print(f"Warning: Could not load CLIP model. Error: {e}")
                self.ClipModel = None
        else:
            self.ClipModel = None

        # 可选：篡改类型分类头
        self.num_manipulation_classes = num_manipulation_classes
        if self.num_manipulation_classes > 0:
            self.manipulation_classifier = MLP(feature_dim_after_experts, [self.num_manipulation_classes], dropout)
        else:
            self.manipulation_classifier = None

    def get_expert_output(self, features, gate_outputs, specific_experts, shared_experts):
        batch_size = features.size(0)
        num_specific_experts = len(specific_experts)
        num_shared_experts_total = len(shared_experts[0])
        expert_output_dim = 320

        expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)
        shared_expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)

        for i in range(num_specific_experts):
            expert_out = specific_experts[i](features)
            gate_val = gate_outputs[:, i].unsqueeze(1)
            expert_outputs_sum += expert_out * gate_val

        for i in range(num_shared_experts_total):
            expert_out = shared_experts[0][i](features)
            gate_val = gate_outputs[:, num_specific_experts + i].unsqueeze(1)
            expert_outputs_sum += expert_out * gate_val
            shared_expert_outputs_sum += expert_out * gate_val

        return expert_outputs_sum, shared_expert_outputs_sum

    def forward(self, **kwargs):
        """
        轻量版 forward，返回 5 项：
          (final_prob, text_prob, image_prob, fusion_prob, manipulation_pred_logits)
        若后续你扩展为 11 项完整版，本文件的 trainer 也能自动适配。
        """
        inputs, masks, image_raw = kwargs['content'], kwargs['content_masks'], kwargs['image']

        # Text encoder
        text_feature_full = self.bert(inputs, attention_mask=masks)[0]

        # Image encoder
        if self.image_model:
            try:
                image_feature_full = self.image_model.forward_ying(image_raw)
            except AttributeError:
                image_feature_full = self.image_model.forward_features(image_raw)
        else:
            image_feature_full = torch.zeros_like(text_feature_full)

        # CLIP (optional)
        clip_image_input, clip_text_input = kwargs.get('clip_image'), kwargs.get('clip_text')
        clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device)
        if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
            with torch.no_grad():
                img_feat = self.ClipModel.encode_image(clip_image_input)
                txt_feat = self.ClipModel.encode_text(clip_text_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            clip_concat_feature = torch.cat((img_feat, txt_feat), dim=-1).float()
            clip_fusion_feature = self.clip_fusion(clip_concat_feature)

        # Attention
        text_atn_feature = self.text_attention(text_feature_full, masks)
        image_atn_feature, _ = self.image_attention(image_feature_full)

        # Domain gating
        domain_idx = kwargs.get('domain_idx', 0) % self.domain_num
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        # Experts
        text_experts_output, text_shared_output = self.get_expert_output(
            text_feature_full, text_gate_out, self.text_experts[domain_idx], self.text_share_expert
        )
        text_final_feature = text_experts_output * F.softmax(self.att_mlp_text(text_experts_output), dim=-1)[:, 0].unsqueeze(1)

        image_experts_output, image_shared_output = self.get_expert_output(
            image_feature_full, image_gate_out, self.image_experts[domain_idx], self.image_share_expert
        )
        image_final_feature = image_experts_output * F.softmax(self.att_mlp_img(image_experts_output), dim=-1)[:, 0].unsqueeze(1)

        # Fusion
        fusion_base = self.MLP_fusion(torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1))
        fusion_gate_out = self.fusion_gate_list0[domain_idx](self.domain_fusion(fusion_base))
        fusion_experts_output, _ = self.get_expert_output(
            fusion_base, fusion_gate_out, self.fusion_experts[domain_idx], self.fusion_share_expert
        )
        fusion_final_feature = fusion_experts_output * F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)[:, 0].unsqueeze(1)

        # Heads
        text_logits = self.text_classifier(text_final_feature).squeeze(1)
        image_logits = self.image_classifier(image_final_feature).squeeze(1)
        fusion_logits = self.fusion_classifier(fusion_final_feature).squeeze(1)

        text_prob = torch.sigmoid(text_logits)
        image_prob = torch.sigmoid(image_logits)
        fusion_prob = torch.sigmoid(fusion_logits)

        all_modality_combined = text_final_feature + image_final_feature + fusion_final_feature
        final_logits = self.max_classifier(all_modality_combined).squeeze(1)
        final_prob = torch.sigmoid(final_logits)

        manipulation_pred_logits = self.manipulation_classifier(all_modality_combined) if self.manipulation_classifier else None

        # 轻量返回 5 项
        return (final_prob, text_prob, image_prob, fusion_prob, manipulation_pred_logits)

    def extract_features(self, **kwargs):
        inputs, masks, image_raw = kwargs['content'], kwargs['content_masks'], kwargs['image']
        text_feature_full = self.bert(inputs, attention_mask=masks)[0]
        if self.image_model:
            try:
                image_feature_full = self.image_model.forward_ying(image_raw)
            except AttributeError:
                image_feature_full = self.image_model.forward_features(image_raw)
        else:
            image_feature_full = torch.zeros_like(text_feature_full)
        clip_image_input, clip_text_input = kwargs.get('clip_image'), kwargs.get('clip_text')
        clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device)
        if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
            with torch.no_grad():
                img_feat = self.ClipModel.encode_image(clip_image_input)
                txt_feat = self.ClipModel.encode_text(clip_text_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            clip_concat_feature = torch.cat((img_feat, txt_feat), dim=-1).float()
            clip_fusion_feature = self.clip_fusion(clip_concat_feature)
        text_atn_feature = self.text_attention(text_feature_full, masks)
        image_atn_feature, _ = self.image_attention(image_feature_full)
        domain_idx = kwargs.get('domain_idx', 0) % self.domain_num
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)
        text_experts_output, text_shared_output = self.get_expert_output(text_feature_full, text_gate_out, self.text_experts[domain_idx], self.text_share_expert)
        text_final_feature = text_experts_output * F.softmax(self.att_mlp_text(text_experts_output), dim=-1)[:, 0].unsqueeze(1)
        image_experts_output, image_shared_output = self.get_expert_output(image_feature_full, image_gate_out, self.image_experts[domain_idx], self.image_share_expert)
        image_final_feature = image_experts_output * F.softmax(self.att_mlp_img(image_experts_output), dim=-1)[:, 0].unsqueeze(1)
        fusion_base = self.MLP_fusion(torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1))
        fusion_gate_out = self.fusion_gate_list0[domain_idx](self.domain_fusion(fusion_base))
        fusion_experts_output, _ = self.get_expert_output(fusion_base, fusion_gate_out, self.fusion_experts[domain_idx], self.fusion_share_expert)
        fusion_final_feature = fusion_experts_output * F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)[:, 0].unsqueeze(1)
        all_modality_combined = text_final_feature + image_final_feature + fusion_final_feature
        return all_modality_combined


class DOMAINTrainerWeibo():
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict,
                 weight_decay, save_param_dir, reasoning_emb_dim=768,
                 num_manipulation_classes=0, lambda_reasoning_align=0.1,
                 lambda_manipulation_predict=0, early_stop=100, epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.early_stop, self.epoches = early_stop, epoches
        self.category_dict, self.use_cuda = category_dict, use_cuda
        self.emb_dim, self.mlp_dims, self.bert, self.dropout = emb_dim, mlp_dims, bert, dropout
        self.save_param_dir = save_param_dir
        if not os.path.exists(self.save_param_dir):
            os.makedirs(self.save_param_dir, exist_ok=True)

        self.lambda_reasoning_align = lambda_reasoning_align
        self.lambda_manipulation_predict = lambda_manipulation_predict

        self.model = MultiDomainPLEFENDModel(
            emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
            out_channels=320, dropout=self.dropout, reasoning_emb_dim=reasoning_emb_dim,
            num_manipulation_classes=num_manipulation_classes)
        if self.use_cuda:
            self.model = self.model.cuda()

        # 使用 BCE on prob（与你原逻辑一致）
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        if num_manipulation_classes > 0:
            self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        self.model_save_filename = 'parameter_calibration_distill.pkl'

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for batch in train_iter:
                batch_data = clipdata2gpu(batch, self.use_cuda)
                labels = batch_data['label'].float()

                # ======== 前向 + 柔性解包（兼容 5/11）========
                outs = self.model(**batch_data)
                (final_prob, text_prob, image_prob, fusion_prob,
                 text_fusion_prob, image_fusion_prob,
                 final_logits, text_logits, image_logits,
                 fusion_logits, aux_or_manip) = _unpack_outputs(outs)

                # === 1) 分类损失 ===
                # 主分支（final）存在时优先；若 final_prob None，可回退到 fusion_prob
                loss_detection = torch.tensor(0.0, device=labels.device)
                if final_prob is not None:
                    loss_main = self.bce_loss(final_prob, labels)
                    loss_detection = loss_detection + loss_main
                elif fusion_prob is not None:
                    loss_detection = loss_detection + self.bce_loss(fusion_prob, labels)

                # 辅助 3 分支（有就算）
                aux_terms, aux_count = 0.0, 0
                for p in (text_prob, image_prob, fusion_prob):
                    if p is not None:
                        aux_terms = aux_terms + self.bce_loss(p, labels)
                        aux_count += 1
                if aux_count > 0:
                    loss_detection = loss_detection + (aux_terms / aux_count)

                # === 2) 蒸馏损失（存在再算） ===
                # 注意：你的 forward 轻量版没有 F_text/F_image/F_cross & P_delta_*；
                # 这里默认 aux_or_manip 是 manip_logits（第 11 位）。若你将来把 11 项完整版接入，
                # 可把这些 teacher/student 张量放入 aux_dict 中，由此处取出计算。
                loss_distill = torch.tensor(0.0, device=labels.device)
                teacher_E_text = batch_data.get('teacher_reasoning_text_emb')
                teacher_E_image = batch_data.get('teacher_reasoning_image_emb')
                teacher_E_cross = batch_data.get('teacher_reasoning_cross_emb')

                # 从 aux 字典中取学生侧特征与预测（当且仅当 forward 返回 11 项且 aux_dict 内含以下键）
                F_text = F_image = F_cross = None
                P_delta_text = P_delta_image = P_delta_cross = None

                if isinstance(aux_or_manip, dict):
                    F_text        = aux_or_manip.get("F_text")
                    F_image       = aux_or_manip.get("F_image")
                    F_cross       = aux_or_manip.get("F_cross")
                    P_delta_text  = aux_or_manip.get("P_delta_text")
                    P_delta_image = aux_or_manip.get("P_delta_image")
                    P_delta_cross = aux_or_manip.get("P_delta_cross")

                has_teacher = all(e is not None for e in [teacher_E_text, teacher_E_image, teacher_E_cross])
                has_student = all(s is not None for s in [F_text, F_image, F_cross, P_delta_text, P_delta_image, P_delta_cross])
                if has_teacher and has_student:
                    Delta_text  = teacher_E_text  - F_text
                    Delta_image = teacher_E_image - F_image
                    Delta_cross = teacher_E_cross - F_cross
                    loss_distill = (self.mse_loss(P_delta_text, Delta_text) +
                                    self.mse_loss(P_delta_image, Delta_image) +
                                    self.mse_loss(P_delta_cross, Delta_cross)) / 3.0

                # === 3) 篡改类型多分类损失（可选，存在再算） ===
                loss_manip = torch.tensor(0.0, device=labels.device)
                if self.lambda_manipulation_predict > 0:
                    manip_logits = None
                    if isinstance(aux_or_manip, dict):
                        manip_logits = aux_or_manip.get("manip_logits")
                    else:
                        # 轻量 5 项返回里第 5 项被放在 aux_or_manip，可能就是 manip_logits
                        manip_logits = aux_or_manip

                    manip_labels = batch_data.get('manipulation_labels')
                    if manip_logits is not None and manip_labels is not None:
                        loss_manip = self.ce_loss(manip_logits, manip_labels.long())

                # === 总损失 ===
                total_loss = loss_detection \
                             + self.lambda_reasoning_align * loss_distill \
                             + self.lambda_manipulation_predict * loss_manip

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                avg_loss.add(total_loss.item())
                train_iter.set_postfix(loss=avg_loss.item())

            print(f'Epoch {epoch+1} Loss: {avg_loss.item():.4f}')

            # 验证
            val_results = self.test(self.val_loader)

            # 早停与保存
            mark = Recorder(self.early_stop).add(val_results)  # 如果你的 Recorder 是要复用一个实例，请移到循环外（此处示例）
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, self.model_save_filename))
            elif mark == 'esc':
                break

        # 测试（加载最优）
        best_path = os.path.join(self.save_param_dir, self.model_save_filename)
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location='cpu'))
            if self.use_cuda:
                self.model.cuda()
        test_results = self.test(self.test_loader)
        print(f"Final Test Results: {test_results}")
        return test_results, best_path

    def test(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_categories = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch_data = clipdata2gpu(batch, self.use_cuda)
                outs = self.model(**batch_data)
                (final_prob, text_prob, image_prob, fusion_prob,
                 text_fusion_prob, image_fusion_prob,
                 final_logits, text_logits, image_logits,
                 fusion_logits, aux_or_manip) = _unpack_outputs(outs)

                # 优先使用 final_prob；不存在则退化到 fusion_prob/text_prob/image_prob
                pred = None
                for p in (final_prob, fusion_prob, text_prob, image_prob):
                    if p is not None:
                        pred = p
                        break
                if pred is None:
                    # 极端情况下防御
                    raise RuntimeError("No probability output found in model forward.")

                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(batch_data['label'].detach().cpu().numpy())
                if 'category' in batch_data:
                    all_categories.extend(batch_data['category'].detach().cpu().numpy())

        return metricsTrueFalse(all_labels, all_preds, all_categories, self.category_dict)

import os

# event_type = "pulmonary"
# event_type = "yidu"
event_type = "DuIE"

#################################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"

#################################################################################

# NER
train_file_path_NER = proj_path + "/NER/data/%s.train" % event_type
test_file_path_NER = proj_path + "/NER/data/%s.test" % event_type
val_file_path_NER = proj_path + "/NER/data/%s.validate" % event_type

# Classification
train_file_path_Classification = proj_path + "/Classification/data/train_sentiment.txt"
val_file_path_Classification = proj_path + "/Classification/data/val_sentiment.txt"
test_file_path_Classification = proj_path + "/Classification/data/train_sentiment.txt"

# Similarity
train_file_path_Similarity = proj_path + "/Similarity/data/train.tsv"
val_file_path_Similarity = proj_path + "/Similarity/data/dev.tsv"
test_file_path_Similarity = proj_path + "/Similarity/data/test.tsv"

# Segmentation
train_file_path_Segmentation = proj_path + "/Segmentation/data/pku_training.utf8"

# KnowledgeExtraction
train_file_path_KE = proj_path + "/KnowledgeExtraction/data/train_data.json"
val_file_path_KE = proj_path + "/KnowledgeExtraction/data/dev_data.json"
schemas_KE = proj_path + "/KnowledgeExtraction/data/all_50_schemas"

#################################################################################

# Model Config
# MODEL_TYPE = 'electra'
MODEL_TYPE = 'albert'

# BASE_MODEL_DIR = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4"
# BASE_CONFIG_NAME = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4/small_ex_discriminator_config.json"
# BASE_CKPT_NAME = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4/electra_small_ex"

# BASE_MODEL_DIR = proj_path + "/electra_180g_base"
# BASE_CONFIG_NAME = proj_path + "/electra_180g_base/base_discriminator_config.json"
# BASE_CKPT_NAME = proj_path + "/electra_180g_base/electra_180g_base.ckpt"

BASE_MODEL_DIR = proj_path + "/albert_tiny_google_zh"
BASE_CONFIG_NAME = proj_path + "/albert_tiny_google_zh/albert_config.json"
BASE_CKPT_NAME = proj_path + "/albert_tiny_google_zh/albert_model.ckpt"

# BASE_MODEL_DIR = proj_path + "/albert_base_google_zh"
# BASE_CONFIG_NAME = proj_path + "/albert_base_google_zh/albert_config.json"
# BASE_CKPT_NAME = proj_path + "/albert_base_google_zh/albert_model.ckpt"

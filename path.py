import os

event_type = "pulmonary"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"

train_file_path = proj_path + "/data/%s.train" % event_type
test_file_path = proj_path + "/data/%s.test" % event_type

all_path = proj_path + '/data/all.txt'

MODEL_TYPE = 'electra'

BASE_MODEL_DIR = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4"
BASE_CONFIG_NAME = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4/small_ex_discriminator_config.json"
BASE_CKPT_NAME = proj_path + "/chinese_electra_small_ex_L-24_H-256_A-4/electra_small_ex"

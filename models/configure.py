import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from models.Utils import *
import random

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
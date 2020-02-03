

from game import *
from mcts import *
from network import *
import numpy as np 
import tensorflow as tf 

# 加载模型
model = network()
model.load_weights("model-3cnn-3000")

# 进行人机对战，n_playout是模拟次数，越大结果越好
mcts = MCTS(n_playout=2000)
mcts.human_play(model=model, use_model=True)


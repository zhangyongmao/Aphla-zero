'''
网络部分
'''

import numpy as np 
import random
import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from game import Board
from mcts import TreeNode, MCTS

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class network(keras.Model):
    def __init__(self, l2_param = 1e-4):
        super(network, self).__init__(self)
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu", padding='same',kernel_regularizer=tf.keras.regularizers.l2(l2_param))
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding='same',kernel_regularizer=tf.keras.regularizers.l2(l2_param))
        self.conv3 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',kernel_regularizer=tf.keras.regularizers.l2(l2_param))
        
        # 预测走法网络部分
        self.conv4 = tf.keras.layers.Conv2D(filters=4,kernel_size=(1,1),activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l2_param))
        self.flatten = tf.keras.layers.Flatten()
        self.dnn = tf.keras.layers.Dense(16, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(l2_param))

        # 预测最终胜率部分
        self.conv5 = tf.keras.layers.Conv2D(filters=2,kernel_size=(1,1),activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l2_param))
        self.flatten2 = layers.Flatten()
        self.dnn2 = layers.Dense(16, kernel_regularizer=keras.regularizers.l2(l2_param))
        self.dnn3 = layers.Dense(1, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_param))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 预测走法部分
        p = self.conv4(x)
        p = self.flatten(p)
        p = self.dnn(p)

        # 预测最终胜率部分
        v = self.conv5(x)
        v = self.flatten2(v)
        v = self.dnn2(v)
        v = self.dnn3(v)

        return p, v



class DataLoader(object):
    ''' 数据加载处理，收集数据并处理成 batch 形式
    '''
    def __init__(self, max_data = 1024):
        self.states = []
        self.Qs = []
        self.win = []
        self.player = []
        self.last_move = []

        # 处理好的数据
        self.max_data = max_data
        self.input_data = [None for i in range(self.max_data)]
        self.output_q = [None for i in range(self.max_data)]
        self.output_v = [None for i in range(self.max_data)]
        self.count = 0
        self.play_count = 0
        self.flag_full = False
        

    def self_play(self, model, use_model = True):
        self.play_count += 1
        print("第 ", self.play_count, " 局自我练习 ！")
        
        # 下棋并记录结果
        mcts = MCTS(use_model, n_playout = 400)
        self.states, self.Qs, self.win, self.player, self.last_move = mcts.self_play(model)
        
        # 处理数据
        for i in range(len(self.states)):
            s_ = self.states[i]
            q_ = self.Qs[i]
            w_ = self.win[i]
            player = self.player[i]
            s_ = s_ * player   # 1 代表当前棋手的棋
            last_move_ = self.last_move[i]
            
            for r in [1,2,3,4]:
                # 旋转增加数据                
                s = np.rot90(s_, r)
                last_move = np.rot90(last_move_, r)
                q = np.rot90(q_, r)
                w = w_

                # 添加数据， 转置状态矩阵增强数据
                # self.input_data[self.count] = np.stack([s, last_move, player * np.ones([8, 8])], axis=2).reshape([8,8,3])
                self.input_data[self.count] = np.stack([np.where(s==1,1,0), np.where(s==-1,1,0), last_move, player * np.ones([4, 4],dtype="float32")], axis=0).transpose((1,2,0)).reshape([4,4,4])
                self.output_q[self.count] = q.reshape([16])
                self.output_v[self.count] = w

                self.count += 1
                if(self.count == self.max_data):
                    self.count = 0
                    self.flag_full = True

                self.input_data[self.count] = np.stack([np.where(s.T==1,1,0), np.where(s.T==-1,1,0), last_move.T, player * np.ones([4, 4],dtype="float32")], axis=0).transpose((1,2,0)).reshape([4,4,4])
                self.output_q[self.count] = q.T.reshape([16])
                self.output_v[self.count] = w

                self.count += 1
                if(self.count == self.max_data):
                    self.count = 0
                    self.flag_full = True

    def get_data(self, batch_size=128):
        '''返回训练batch数据'''
        if(self.flag_full):
            data = np.stack(random.sample(self.input_data, batch_size), axis=0)
            label_q = np.stack(random.sample(self.output_q, batch_size), axis=0).reshape([batch_size, -1])
            label_v = np.stack(random.sample(self.output_v, batch_size), axis=0).reshape([-1,1])
            return data, label_q, label_v, True
        else:
            if(self.count > batch_size):
                data = np.stack(random.sample(self.input_data[:self.count], batch_size), axis=0)
                label_q = np.stack(random.sample(self.output_q[:self.count], batch_size), axis=0).reshape([batch_size, -1])
                label_v = np.stack(random.sample(self.output_v[:self.count], batch_size), axis=0).reshape([-1,1])
                return data, label_q, label_v, True
            return None,None, None, False


model = network()
model.load_weights("model-4-4-3-1000")
mcts = MCTS(n_playout=2000)
mcts.human_play(model=model, use_model=True)
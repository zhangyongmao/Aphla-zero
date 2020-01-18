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
    def __init__(self):
        super(network, self).__init__(self)
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        
        # 预测走法网络部分
        self.conv4 = tf.keras.layers.Conv2D(filters=4,kernel_size=(1,1),kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.flatten = tf.keras.layers.Flatten()
        self.dnn = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        # 预测最终胜率部分
        self.conv5 = tf.keras.layers.Conv2D(filters=2,kernel_size=(1,1),kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.flatten2 = layers.Flatten()
        self.dnn2 = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))
        self.dnn3 = layers.Dense(1, activation="tanh", kernel_regularizer=keras.regularizers.l2(1e-4))

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
    def __init__(self):
        self.states = []
        self.Qs = []
        self.win = []
        self.player = []
        self.last_move = []

        # 处理好的数据
        self.max_data = 10000
        self.input_data = [None for i in range(self.max_data)]
        self.output_q = [None for i in range(self.max_data)]
        self.output_v = [None for i in range(self.max_data)]
        self.count = 0
        self.play_count = 0
        self.flag_full = False
        

    def self_play(self, model):
        self.play_count += 1
        print("第 ", self.play_count, " 局自我练习 ！")
        
        # 下棋并记录结果
        mcts = MCTS()
        self.states, self.Qs, self.win, self.player, self.last_move = mcts.self_play(model)
        
        # 处理数据
        for i in range(len(self.states)):
            s = self.states[i]
            q = self.Qs[i]
            w = self.win[i]
            player = self.player[i]
            s = s * player
            last_move = self.last_move[i]
            
            # 添加数据， 转置状态矩阵增强数据
            # self.input_data[self.count] = np.stack([s, last_move, player * np.ones([8, 8])], axis=2).reshape([8,8,3])
            self.input_data[self.count] = np.stack([s, last_move, player * np.ones([8, 8],dtype="float32")], axis=0).transpose((1,2,0)).reshape([8,8,3])
            self.output_q[self.count] = q.reshape([64])
            self.output_v[self.count] = w

            self.count += 1
            if(self.count == self.max_data):
                self.count = 0
                self.flag_full = True

            self.input_data[self.count] = np.stack([s.T, last_move.T, player * np.ones([8, 8],dtype="float32")], axis=0).transpose((1,2,0)).reshape([8,8,3])
            self.output_q[self.count] = q.T.reshape([64])
            self.output_v[self.count] = w

            self.count += 1
            if(self.count == self.max_data):
                self.count = 0
                self.flag_full = True

    def get_data(self, batch_size=16):
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

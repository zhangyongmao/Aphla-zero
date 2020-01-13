# 
'''
蒙罗卡罗树部分
'''

import numpy as np 
from game import Board
import copy

class TreeNode(object):
    def __init__(self, parent=None, board = None, p = None):
        self.parent =  parent
        self.child = {}  # 键为动作，值为子节点
        self.Q = 0  # 胜率
        self.U = 0  # 探索因子
        self.n = 0  # 访问次数
        if(p == None and self.parent != None):
            self.P = 1.0 / len(self.parent.board.moveable)
        else:
            if(self.parent == None):
                self.P = 1
            else:
                self.P = p

        # self.current_player = 1  # 当前玩家
        if(board == None):
            self.board = Board()  # 当前棋盘
        else:
            self.board = board

    def expand(self, q_network = np.ones([8,8]) * 0.02):
        '''
        拓展子节点
        '''
        for move in self.board.moveable:
            h,w = self.board.move_to_hw(move)
            q = q_network[h,w]
            self.child[move] = TreeNode(parent=self, board = self.board.move(move), p = q)
            
            
    def get_best_move(self):
        '''
        找到当前局面的最好走法
        '''
        return max(self.child.items(), key=lambda node: node[1].get_value())

    def get_value(self, p = 5):
        '''
        计算平均胜率和探索因子的和 作为选择子节点的标准
        '''
        # if(self.parent != None):
        #     self.P = 1.0 / len(self.parent.board.moveable)
        # self.U = self.P * p * np.sqrt(np.log(self.parent.n) / (1 + self.n))
        self.U = p * self.P * np.sqrt(self.parent.n) / (1 + self.n)
        return self.Q + self.U

    def update(self, winner):
        '''
        更新
        '''
        value = -winner * self.board.current_player
        self.n += 1.0
        self.Q += 1.0 * (value - self.Q) / self.n
        if(self.parent != None):
            self.parent.update(winner)


class MCTS(object):
    '''进行蒙罗卡罗搜索的部分'''

    def __init__(self, play_fn = None, n_playout = 400):
        self.play_fn = play_fn
        self.n_playout = n_playout
        self.root = TreeNode()

    def self_play(self, model = None):
        '''自我对局到最后结束，并收集数据'''
        self.model = model
        while(True):
            if(self.root.board.have_winer()[0]):
                break
            self.one_step()

        # 收集数据
        winner = self.root.board.have_winer()[1]
        states = []
        Qs = []
        winners = []
        player = []
        last_moves = []
        node = self.root
        while(True):
            # 放在开始，最后结束的叶节点不计
            node = node.parent
            states.append(node.board.state)
            player.append(node.board.current_player)
            
            tmp_move = np.zeros([node.board.h, node.board.w])
            h_, w_ = node.board.move_to_hw(node.board.last_move)
            tmp_move[h_,w_] = 1
            last_moves.append(tmp_move)

            Q_ = np.zeros([node.board.h, node.board.w])
            for move in node.board.moveable:
                q = node.child[move].Q
                Q_[node.board.move_to_hw(move)[0], node.board.move_to_hw(move)[1]] = q
            Q_ = softmax(Q_)
            Qs.append(Q_)
            # 当前玩家与最终赢家
            winners.append(winner)

            if(node.parent == None):
                break

        return (states, Qs, winners, player, last_moves)

    def human_play(self):
        '''与人对战，测试'''
        '''自我对局到最后结束，并收集数据'''
        self.root.board.show()
        while(True):
            if(self.root.board.have_winer()[0]):
                break
            print(self.root.board.current_player)
            self.one_step()
            self.root.expand()

            self.root.board.show()
            
            i = input()
            p = i.split(" ")
            h = int(p[0])
            w = int(p[1])
            if(self.root.child == {}):
                self.root.expand()
            self.root = self.root.child[self.root.board.hw_to_move(h,w)]
            self.root.board.show()
            
        print("end play !! ")

    def one_step(self):
        '''计算根节点向下的胜率，并下一步棋'''
        for i in range(self.n_playout):
            self.playout()
        best_move, node = self.root.get_best_move()
        for m in self.root.board.moveable:
            if(m != best_move):
                self.root.child[m].child = {}
                self.root.child[m].state = None
                self.root.child[m].moveable = None
        self.root = node


    def playout(self):
        '''
        一次模拟过程: 
        移动到叶子节点，模拟下棋到结束，根据结果更新
        '''
        node = self.root
        while(True):
            if(node.child == {}):
                break
            move, node = node.get_best_move()
        
        # 模拟下棋到游戏结束
        # 返回 leaf_value : 如果赢家与当前节点的玩家一样为1，不一样为-1 
        winner = self.simulate(copy.deepcopy(node.board))
        node.update(winner)

        end, winner = node.board.have_winer()
        if(not end):
            # 算出概率 Q 来拓展
            s = node.board.state.astype("float32")
            player = node.board.current_player
            s = s * player
            last_move = np.zeros([node.board.h, node.board.w],dtype = "float32")
            h_, w_ = node.board.move_to_hw(node.board.last_move)
            last_move[h_,w_] = 1
            
            # 添加数据， 转置状态矩阵增强数据
            # q, v = self.model(np.stack([s, last_move, player * np.ones([8, 8],dtype="float32")], axis=2).reshape([1,8,8,3]))
            q, v = self.model(np.stack([s, last_move, player * np.ones([8, 8],dtype="float32")], axis=0).transpose((1,2,0)).reshape([1,8,8,3]))
            # 拓展子节点，记录访问的先验概率
            node.expand(q.numpy().reshape([node.board.h, node.board.w]))
        
    def simulate(self, board):
        '''模拟一次对局到结束'''
        while(True):
            move = board.moveable[np.random.randint(0, len(board.moveable), dtype='int')]
            board.move_no_copy(move)
            if(board.have_winer()[0] or len(board.moveable) <= 2):
                break
        return board.have_winer()[1]

        
    
def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))    


# mcts = MCTS()
# mcts.human_play()


# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# d = np.array([[0,0],[0,0]])
# c = np.stack([a,b,d], axis=2)
# print(c.shape)
# print(c)

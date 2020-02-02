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

        if(board == None):
            self.board = Board()  # 当前棋盘
        else:
            self.board = board

    def expand(self, q_network = np.ones([4,4]) * 0.1):
        '''
        拓展子节点
        '''
        for move in self.board.moveable:
            h,w = self.board.move_to_hw(move)
            q = q_network[h,w]
            self.child[move] = TreeNode(parent=self, board = self.board.move(move), p = q)
            
            
    def get_best_move(self):
        '''
        找到mcts树节点对应的局面的最好走法，用于400次模拟时选择下一步节点
        '''
        return max(self.child.items(), key=lambda node: node[1].get_value())

    def get_one_step_move(self, training = True):
        '''
        用来进行n次模拟后选出当前局面最终要走哪一步
        '''
        move_visit = [(move, child_node.n) 
                    for move, child_node in self.child.items()]
        move, visit = zip(* move_visit)
        move_prob = softmax(1.0 * np.log(np.array(visit) + 1e-10))
        if(training):
            result_move = np.random.choice(move, 
                        p= 0.8 * move_prob + 0.2 * np.random.dirichlet(0.3*np.ones(len(visit))))
        else:
            result_move = np.random.choice(move, p= move_prob)

        # 计算 Q 矩阵作为预测的走法概率label
        move_Q = np.zeros([self.board.h, self.board.w])
        for i in range(len(move)):
            m = move[i]
            h,w = self.board.move_to_hw(m)
            move_Q[h, w] = move_prob[i]
        # 返回最终走法, 对应的节点, 每个位置对应的走法预测概率
        return result_move, self.child[result_move], move_Q

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

    def __init__(self, use_network = False, n_playout = 400):
        self.use_network = use_network  # 是否使用网络来模拟一次下棋的胜负结果，False的话用随机下棋来模拟结果，True用network预测结果
        self.n_playout = n_playout
        self.root = TreeNode()
        self.training = True

    def self_play(self, model = None):
        '''自我对局到最后结束，用来收集数据训练网络'''
        states = []
        Qs = []
        winners = []
        player = []
        last_moves = []
        self.model = model
        self.training = True

        # 进行一局模拟对战到结束
        while(True):
            # 有赢家
            if(self.root.board.have_winer()[0]):
                break
            
            # 走一步棋，返回下一个节点，最好的走法，8*8的价值矩阵（用来作为网络的走法概率输出）
            next_node, best_move, move_prob = self.one_step()
            
            # 记录这一步的训练数据
            states.append(self.root.board.state)  # 记录状态
            Qs.append(move_prob)                  # 记录价值
            player.append(self.root.board.current_player) # 记录当前玩家
            
            # 记录上一步棋的位置
            tmp_move = np.zeros([self.root.board.h, self.root.board.w])
            if(self.root.board.last_move != -1):
                h_, w_ = self.root.board.move_to_hw(self.root.board.last_move)
                tmp_move[h_, w_] = 1
            last_moves.append(tmp_move)

            # 树根节点向下移，丢弃之上的部分，之上部分的数据都已经保存好了
            self.root = next_node
            self.root.parent = None 
        
        # 收集每一步对应的胜率
        winner = self.root.board.have_winer()[1]
        for current_player in player:
            if(current_player == winner):
                winners.append(1)
            else:
                if(winner == 0):
                    winners.append(0)
                else:
                    winners.append(-1)

        return (states, Qs, winners, player, last_moves)


    def one_step(self):
        '''计算根节点向下的胜率，并下一步棋'''
        for i in range(self.n_playout):
            self.playout()
        best_move, node, move_prob = self.root.get_one_step_move(self.training)
        next_node = node

        return next_node, best_move, move_prob


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
        end, winner = node.board.have_winer()

        if(not end):
            if(not self.use_network):  
                # 用随机模拟来预测结果
                winner = self.simulate_random(copy.deepcopy(node.board))
                node.expand()
                node.update(winner)

            else:
                # 用网络来预测结果
                # 算出概率 Q 来拓展
                s = node.board.state.astype("float32")
                player = node.board.current_player
                s = s * player
                last_move = np.zeros([node.board.h, node.board.w],dtype = "float32")
                if(node.board.last_move != -1):
                    h_, w_ = node.board.move_to_hw(node.board.last_move)
                    last_move[h_,w_] = 1
                
                # 通过网络得到走法概率和胜率
                q, v = self.model(np.stack([np.where(s==1,1,0), np.where(s==-1,1,0), last_move, player * np.ones([4, 4],dtype="float32")], axis=0).transpose((1,2,0)).reshape([1,4,4,4]).astype("float32"))
                
                # 拓展子节点，记录访问的先验概率
                node.expand(q.numpy().reshape([node.board.h, node.board.w]))

                # 根据网络输出的胜率来更新 蒙罗卡罗树节点，v表示最后当前玩家是否会获胜，乘以当前玩家后就是最终胜利的真实玩家（1 或 -1）
                node.update(v.numpy()[0][0] * player)
            
        else:
            # mcts 树增长到游戏结束的节点, 更新后一局自我对战结束
            node.update(winner)


    def human_play(self, model = None, use_model = False, n_playout = 2000):
        '''与人对战，测试'''
        '''自我对局到最后结束，并收集数据'''
        self.model = model
        self.use_network = use_model
        self.n_playout = n_playout
        self.training = False
        self.root.board.show()
        while(True):
            if(self.root.board.have_winer()[0]):
                self.root.board.show()
                break
            print(self.root.board.current_player)
            self.root, _, _ = self.one_step()
            self.root.expand()
            if(self.root.board.have_winer()[0]):
                self.root.board.show()
                break

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
        
    def simulate_random(self, board):
        '''随机模拟一次对局到结束，用于纯蒙罗卡罗搜索，aphla-zero中不用'''
        while(True):
            move = board.moveable[np.random.randint(0, len(board.moveable), dtype='int')]
            board.move_no_copy(move)
            if(board.have_winer()[0] or len(board.moveable) <= 1):
                break
        return board.have_winer()[1]

        
    
def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
    
     
# mcts = MCTS()
# mcts.human_play()



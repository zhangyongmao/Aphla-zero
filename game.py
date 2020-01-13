# 
'''
五子棋游戏部分，包括棋盘结构，判断输赢，状态
'''

import numpy as np
import copy 

class Board(object):
    '''棋盘结构，'''

    def __init__(self, h=8, w=8 ):
        self.h = h      # 行数
        self.w = w      # 列数
        self.n_line = 5 # 连续的棋子数算赢
        self.player = [1, -1]   # 玩家，默认1先手
        self.current_player = 1 # 当前棋手，默认1先手
        self.state = np.zeros([self.h, self.w])  # 棋盘矩阵
        self.moveable = [i for i in range(self.h * self.w)] # 可以走的棋的位置
        self.last_move = -1 # 最后下的棋的编号

    def move_to_hw(self, move):
        '''
        输入棋盘位置编号，返回棋盘上行号和列号
        '''
        return (move // self.w, move % self.w)

    def hw_to_move(self, h, w):
        '''
        返回hw转化为的位置编号
        '''
        if(h >= self.h or h < 0 or w >= self.w or w < 0):
            print('error hw位置不合法') 
        return h * self.w + w
    
    def get_state_player(self, player):
        '''
        返回该玩家下的棋的矩阵
        '''
        if(player not in self.player):
            print("error 无此玩家")
        return self.state[self.state == player]

    def move(self, num):
        '''
        下一个棋子，更换当前棋手
        '''
        if(num not in self.moveable):
            print("error 当前位置不可以下")

        # 复制当前局面为新的局面
        next_board = copy.deepcopy(self)

        next_board.moveable.remove(num)
        next_board.state[self.move_to_hw(num)[0], self.move_to_hw(num)[1]] = self.current_player
        next_board.current_player = -1 * self.current_player
        next_board.last_move = num
        return next_board

    def move_no_copy(self, num):
        '''
        下一个棋子，更换当前棋手,不生成新局面，用于模拟使用
        '''
        if(num not in self.moveable):
            print("error 当前位置不可以下")

        self.moveable.remove(num)
        self.state[self.move_to_hw(num)[0], self.move_to_hw(num)[1]] = self.current_player
        self.current_player = -1 * self.current_player
        self.last_move = num

    def have_winer(self):
        '''
        判断是否有赢家
        ''' 
        if(self.last_move == -1):
            return False, 0
        last_move = self.last_move
        dir = [ [1,0], [0,1], [1,1], [1,-1] ]
        h,w = self.move_to_hw(last_move)
        if(self.state[h,w] == 0):
            print("error 计算hw出错")

        for d in dir:
            count = 1
            for i in range(1, self.n_line):
                h_, w_ = h + d[0] * i , w + d[1] * i
                if(h_ >= 0 and h_ < self.h and w_ >= 0 and w_ < self.w and self.state[h_, w_] == self.state[h,w] ):
                    count += 1
                else:
                    break
            for i in range(-1, -self.n_line, -1):
                h_, w_ = h + d[0] * i , w + d[1] * i
                if(h_ >= 0 and h_ < self.h and w_ >= 0 and w_ < self.w and self.state[h_, w_] == self.state[h,w] ):
                    count += 1
                else:
                    break
            if(count >= self.n_line): 
                # print("胜利 ： player：", self.current_player * -1)
                return True, self.current_player * -1

            # 平局
            if(len(self.moveable) == 0):
                return True, 0
        return False, 0
    
    def show(self):
        '''方便人机对战'''
        print("当前玩家：",self.current_player)
        print('*    ',end="")
        for i in range(8):
            print(i,'  ',end="")
        print("")
        for i in range(8):
            print(i,'   ',end="")
            for j in range(8):
                if(self.state[i,j] == 0.):
                    print('_   ',end="")
                else:
                    if(self.state[i,j] == 1.):
                        print('1   ',end="")
                    else:
                        print('0   ',end="")
            print("")
            print("")

# board = Board()
# while(True):
#     print("\r ", board.state)
#     i = input()
#     p = i.split(" ")
#     h = int(p[0])
#     w = int(p[1])
#     board = board.move(board.hw_to_move(h,w))
#     if(board.have_winer()[0]):
#         break

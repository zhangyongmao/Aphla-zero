# 训练部分

from game import *
from mcts import *
from network import *
import numpy as np 
import tensorflow as tf 


model = network()
optim = keras.optimizers.Adam(learning_rate=2e-3)
dataloader = DataLoader()

# 继续训练
model.load_weights("model")

for epcho in range(10):
    # 进行一次自我对战，收集处理数据
    dataloader.self_play(model, use_model=True)
    
    for repeat in range(1000):
        data, label_q, label_v, flag_get = dataloader.get_data(128)
        if(flag_get):
            # 进行训练
            x = data.astype("float32")
            q = label_q.astype("float32")
            w = label_v.astype("float32")
            with tf.GradientTape() as tape:
                Q_pred, v_pred = model(x)
                loss1 = tf.reduce_mean(keras.losses.categorical_crossentropy(y_true=q,y_pred=Q_pred)) 
                loss2 = tf.reduce_mean(keras.losses.mean_squared_error(y_true=w, y_pred=v_pred))
                loss = loss1 + loss2
                print("loss = ",loss)
                print(" 走法预测loss= ",loss1.numpy()," 最终胜率预测loss = ", loss2.numpy())

            gards = tape.gradient(loss, model.variables)
            optim.apply_gradients(grads_and_vars=zip(gards,model.variables))

    if(epcho % 200 == 0 and epcho > 0):
        model.save_weights("model-"+str(epcho))

model.save_weights("model")


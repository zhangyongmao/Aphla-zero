# 训练部分

from game import *
from mcts import *
from network import *
import numpy as np 
import tensorflow as tf 


model = network()
optim = keras.optimizers.Adam(learning_rate=1e-3)
dataloader = DataLoader()

# 继续训练
model.load_weights("model-100")

for epcho in range(1):
    # 进行一次自我对战，收集处理数据
    dataloader.self_play(model)
    data, label_q, label_v = dataloader.get_data()
    
    if(data != []):
        for repeat in range(5):
            # 进行训练
            x = data.astype("float32")
            q = label_q.astype("float32")
            w = label_v

            with tf.GradientTape() as tape:
                Q_pred, v_pred = model(x)

                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_true=q,y_pred=Q_pred) + keras.losses.mean_squared_error(y_true=w, y_pred=v_pred))
                print("loss = ",loss)

            gards = tape.gradient(loss, model.variables)
            optim.apply_gradients(grads_and_vars=zip(gards,model.variables))

print("end")

model.save_weights("model")


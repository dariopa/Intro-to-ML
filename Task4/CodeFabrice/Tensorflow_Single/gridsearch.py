import Train1
import Train2
import Train3
import math

epoch_list = [300, 400, 500]
param_list = [200, 320, 440]
batch_size_list = [64, 128, 256]

iters = len(epoch_list)*len(param_list)*len(batch_size_list)
i = 0
avg_accuracy_best = 0
model1 = Train3.Train_3()

while i < iters:
    epochs = epoch_list[i%len(epoch_list)]
    params = param_list[math.floor((i/len(param_list))%len(epoch_list))]
    batch_size = batch_size_list[math.floor((i/(len(param_list)*len(batch_size_list)))%len(epoch_list))]
    print(epochs)
    print(params)
    print(batch_size)
    # Train
    avg_accuracy = model1.fit(epochs, batch_size, params)
    if  avg_accuracy > avg_accuracy_best:
        epoch_best = epochs
        param_best = params
        batch_size_best = batch_size
        avg_accuracy_best = avg_accuracy
    i += 1

print('epoch = ', repr(epoch_best), '| param = ', repr(param_best),  '| batch_size = ', repr(batch_size_best), '| accuracy = ', repr(avg_accuracy_best))

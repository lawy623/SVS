import os
import pylab
import sys

log_file = sys.argv[1]
train_iter = []
train_loss = []
test_iter = []
test_acc = []
test_loss = []

class CaffeVersion:
    OLD, NEW = range(0, 2)
CAFFE_VERSION = CaffeVersion.NEW

# prefix_loss = (CAFFE_VERSION is CaffeVersion.OLD) and 'loss = ' or 'loss = '
prefix_loss = (CAFFE_VERSION is CaffeVersion.OLD) and 'flow_loss0 = ' or 'flow_loss0 = '

prefix_acc  = 'accuracy_texture = '
prefix_iter = 'Iteration '
# prefix_scr0 = 'Test score #0: '
# prefix_scr1 = 'Test score #1: '

prefix_train_scr0 = 'Train net output #0: flow_loss0 = '
prefix_scr0 = 'Test net output #0: flow_loss0 = '
prefix_scr1 = 'Test net output #1: '

plateau = 4

with open(log_file) as f:
    for line in f:
        if line.find(prefix_iter) >= 0:
            st = line.find(prefix_iter) + len(prefix_iter)
            ed = line.find(',')
            iter_num = int(line[st:ed])
            if line.find('Testing net') >= 0:
                test_iter.append(iter_num)
            else:
                if line.find('lr = ') < 0:
                    train_iter.append(iter_num)

        if CAFFE_VERSION is CaffeVersion.OLD:
            if line.find(prefix_train_scr0) >= 0:
                # st = line.find(prefix_loss) + len(prefix_loss)
                st = line.find(prefix_train_scr0) + len(prefix_train_scr0)
                train_loss.append(float(line[st:]))
                continue
            if line.find(prefix_scr0) >= 0:
                st = line.find(prefix_scr0) + len(prefix_scr0)
                test_acc.append(float(line[st:]))
                continue
            if line.find(prefix_scr1) >= 0:
                st = line.find(prefix_scr1) + len(prefix_scr1)
                test_loss.append(float(line[st:]))
                continue
        else:
            if line.find(prefix_train_scr0) >= 0:
                st = line.find(prefix_train_scr0) + len(prefix_train_scr0)
                ed = line.find('(*')
                loss = float(line[st:ed])
                if loss > plateau:
                    loss = plateau
                if line.find('flow_loss0 =') >= 0:
                    train_loss.append(loss)
                if line.find('Test net') >= 0:
                    test_loss.append(loss)
                continue
            if line.find(prefix_scr0) >= 0:
                st = line.find(prefix_scr0) + len(prefix_scr0)
                ed = line.find('(*')
                loss = float(line[st:ed])
                if line.find('flow_loss0 =') >= 0:
                    test_loss.append(loss)
                continue
            if line.find(prefix_acc) >= 0 and line.find('Test net') >= 0:
                st = line.find(prefix_acc) + len(prefix_acc)
                test_acc.append(float(line[st:]))

                continue

if len(train_iter) - len(train_loss) == 1:
    train_iter = train_iter[:-1]
if len(train_iter) - len(train_loss) == 2:
    train_iter = train_iter[:-2]
if len(test_iter) - len(test_loss) == 1:
    test_iter = test_iter[:-1]

print len(train_iter), len(train_loss)
print len(test_iter), len(test_loss)

if len(train_iter) > 100:
    pylab.plot(train_iter[4:len(train_iter)], train_loss[4:len(train_loss)])
    pylab.plot(test_iter[1:len(test_iter)], test_loss[1:len(test_iter)], 'r')
else:
    pylab.plot(train_iter[4:len(train_iter)], train_loss[4:len(train_loss)],'*-')
    pylab.plot(test_iter[1:len(test_iter)], test_loss[1:len(test_iter)], 'rs-')

pylab.plot(train_iter, train_loss, 'b')
pylab.plot(test_iter, test_loss, 'r')
pylab.legend(['training', 'testing'])
pylab.xlabel('iter'), pylab.ylabel('loss')
# pylab.figure()
# pylab.plot(test_iter, test_acc)
# pylab.xlabel('iter'), pylab.ylabel('test accuracy')
pylab.show()

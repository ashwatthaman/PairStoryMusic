
import codecs,sys,json,csv
import numpy as np
# import cupy

from chainer import serializers,optimizers
from chainer import functions as F
import chainer

from evaluation import evalDiscrete
import os,math
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")

# @profile由来のエラー除去
try:
    profile
except NameError:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

@profile
def train_dev_test(args,model,x_train,y_train,x_dev,y_dev,x_test,y_test):
    @profile
    def iter_train_dev_test(model,optimizer,args,x_train,y_train,x_dev,y_dev,x_test,y_test,iter_unit,es_now,eval_dev_max,early_stop=3):
        ep_unit = len(x_train) // iter_unit
        if ep_unit>0:
            x_train_arr=[x_train[ri*iter_unit:(ri+1)*iter_unit] for ri in range(ep_unit)]
            y_train_arr=[y_train[ri*iter_unit:(ri+1)*iter_unit] for ri in range(ep_unit)]
        else:
            x_train_arr=[x_train]
            y_train_arr=[y_train]
        for iter_i,(x_tr,y_tr) in enumerate(zip(x_train_arr,y_train_arr)):
            with chainer.using_config('train', True):
                loss_sum=trainSub(model,optimizer,x_tr, y_tr,train=True)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                eval_dev = devSub(args, model,x_dev, y_dev)
            if eval_dev_max is None or eval_dev > eval_dev_max:
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    eval_test = testSub(args,model,x_test,y_test)
                print("\n   epoch{}_{}:loss_sum:{},eval_dev:{},eval_test:{}".format(model.epoch_now,iter_i, round(float(loss_sum), 4),round(float(eval_dev), 4),round(float(eval_test), 4)))
                eval_dev_max = eval_dev
                print('update_model:{}'.format(model.epoch_now))
                serializers.save_npz(args.model_name.format(model.epoch_now)+'.npz', model)
                es_now = 0
            else:
                print("\n   epoch{}_{}:loss_sum:{},eval_dev:{}".format(model.epoch_now, iter_i,round(float(loss_sum), 4),round(float(eval_dev), 4)))
                es_now += 1
            if es_now > early_stop:
                print('earlyStopped')
                break
            # break
        return eval_dev_max,es_now

    def trainSub(model,optimizer,x_train,y_train,train):
        model.setData(x_train,y_train)
        loss_sum = 0
        for tupl in model.getBatchGen():
            # try:
            loss = model(tupl)
            # except cupy.cuda.memory.OutOfMemoryError:
            #     print('outMemory')
            #     loss = model(tupl[:len(tupl)//2])
            assert not math.isnan(loss.data)
            loss_sum += loss.data
            assert not math.isnan(loss_sum)
            if train:
                model.cleargrads()
                loss.backward()
                optimizer.update()
        return loss_sum#"""

    def devSub(args, model,x_dev,y_dev):
        model.setData(x_dev, y_dev)
        if args.outtype=="reg":
            eval_dev = 100.0 / trainSub(model.copy(),None, train=False)
        elif args.outtype == 'cls':
            return test(args, model, evalfunc=evalDiscrete)
        else:  # args.outtype=='summarize':
            return test(args, model, evalfunc=evalRouge)
        return eval_dev

    def testSub(args, model,  x_test, y_test):
        model.setData(x_test, y_test)
        if args.outtype=="reg":
            return test(args, model, evalfunc=evalRank)
        elif args.outtype=='cls':
            return test(args, model, evalfunc=evalDiscrete)
        else: #args.outtype=='summarize':
            return test(args, model, evalfunc=evalRouge)

    first_e=model.loadModel(args)
    early_stop=3;es_now=0

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    # with chainer.using_config('train', False), chainer.no_backprop_mode():
    #     eval_dev_max = devSub(args, model, x_dev, y_dev)
    #     eval_test = testSub(args, model, x_test, y_test)
    #     print("\n   epoch{}_{}:eval_dev:{},eval_test:{}".format(model.epoch_now, -1,round(float(eval_dev_max), 4),round(float(eval_test), 4)))
    eval_dev_max=0.0;eval_test=0.0
    for e_i in range(model.epoch_now, args.epoch):
        model.setEpochNow(e_i)
        eval_dev_max,es_now=iter_train_dev_test(model,optimizer,args,x_train,y_train,x_dev,y_dev,x_test,y_test,iter_unit=3*64000,es_now=es_now,eval_dev_max=eval_dev_max,early_stop=early_stop)
        if es_now > early_stop:
            print('earlyStopped')
            break

def testCommon(args,model,epoch=None):
    model.loadModel(args,load_epoch=epoch)
    args.model_name_base=args.model_name_base+'_test'
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        if args.outtype=="reg":
            return test(args,model,evalRank)
        elif args.outtype=='cls':
            return test(args,model,evalDiscrete)
        else:
            return test(args,model,evalRouge)

def test(args,model,evalfunc,write=True):
    filename = args.model_name_base.format(model.epoch_now)
    if write:
        # fw = codecs.open(model.pwd + "./{}/output/result_{}.csv".format(args.dataname,filename), "w",encoding="utf-8")
        csvw = csv.writer(codecs.open(model.pwd + "./{}/output/result_{}.csv".format(args.dataname,filename), "w",encoding="utf-8"))
        logw = codecs.open(model.pwd + "./{}/log/log_{}.json".format(args.dataname,filename), "w", encoding="utf-8")
        log_hash = {}
        log_hash["args"] = args.__dict__
        csvw.writerow(model.write_output_col())
    all_t=[];all_y=[]
    for tupl in model.getBatchGen():
        line_list,t_list,y = model.write_output_each(tupl)
        if write:[csvw.writerow(line) for line in line_list]
        all_y+=y;all_t+=t_list
    eval_val,eval_hash=evalfunc(all_t, all_y)
    if write:
        log_hash["eval"] = eval_hash
        logw.write(json.dumps(log_hash, default=lambda x: x.__class__.__name__))
        logw.close()
        # fw.close()
    return eval_val#,eval_hash


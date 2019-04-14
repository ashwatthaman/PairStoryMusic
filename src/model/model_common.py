import numpy as np
from chainer import cuda
if cuda.cudnn_enabled:
    import cupy as xp
else:
    global xp
    xp=np

from chainer import Chain
import os,sys
from chainer import serializers
import util.generators as gens
import random,codecs
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
#@profile由来のエラー除去
try:
    profile
except NameError:
    def profile(func):
        def inner(*args,**kwargs):
            return func(*args,**kwargs)
        return inner

class NNChainer(Chain):

    def __init__(self,**args):
        super(NNChainer, self).__init__(**args)

    def make_xp_array(self,list1,type='int'):
        if type=='int':
            return xp.array(list1, dtype=self.xp.int32)
        elif type=='float':
            return xp.array(list1, dtype=self.xp.float32)

    def setArgs(self,args):
        self.max_len = args.max_len
        if args.gpu>=0:
            import cupy as xp

        self.pwd = CDIR + '../'+args.model_dir+'/'
        self.wfile = CDIR + "../{}/log/log_{}.txt".format(args.model_dir,args.dataname)

        self.n_vocab = args.n_vocab
        self.n_embed = args.n_embed
        self.n_hidden= args.n_hidden
        self.n_layers = 1
        self.setBatchSize(args.batchsize)
        self.setMaxEpoch(args.epoch)
        self.setEpochNow(0)

    def setDevice(self,args):
        global xp
        if args.gpu>=0:
            import cupy as cp
            xp = cp
            try:
                xp.cuda.Device(args.gpu).use()
            except cp.cuda.runtime.CUDARuntimeError:
                xp.cuda.Device().use()
            try:
                self.to_gpu(args.gpu)
            except xp.cuda.runtime.CUDARuntimeError:
                self.to_gpu()
        else:
            xp = np


    def setEpochNow(self, epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self, epoch):
        self.epoch = epoch

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setData(self,tt_list,cat_list):
        self.tt_list =tt_list[:]
        self.cat_list=cat_list[:]

    def loadModel(self,args,load_epoch=None):
        first_e = 0
        model_name = ""
        max_epoch = args.epoch if load_epoch is None else load_epoch
        for e in range(max_epoch):
            model_name_tmp = args.model_name.format(e)+'.npz'
            # print('model_tmp',model_name_tmp)
            if os.path.exists(model_name_tmp):
                model_name = model_name_tmp
                self.setEpochNow(e+1)
        if os.path.exists(model_name):
            serializers.load_npz(model_name, self)
            print("loaded_{}".format(model_name))
            first_e = self.epoch_now
        else:
            print("loadW2V")
            if os.path.exists(args.premodel):
                self.loadW(args.premodel)
            else:
                print("wordvec model doesnt exist.")
        return first_e


    def getBatchGen(self,test_flag=True):
        ind_arr = list(range(len(self.tt_list)))
        if not test_flag:
            # print("shuffled")
            random.shuffle(ind_arr)

        tt_now  = (self.tt_list[ind] for ind in ind_arr)
        cat_now = (self.cat_list[ind] for ind in ind_arr)
        tt_gen  = gens.batch(tt_now, self.batch_size)
        cat_gen = gens.batch(cat_now, self.batch_size)
        for tt,cat in zip(tt_gen,cat_gen):
            yield (tt,cat)

    def get_input_and_label(self,tupl,shuffle=False):
        def shuffle2N(x):
            ind_arr = [random.randint(1,len(x)-1) for _ in range(int(len(x)/2))]
            for ind in ind_arr:
                x[ind-1],x[ind] = x[ind],x[ind-1]
            return x
        xs = [t_e[0] for t_e in tupl[0]];
        xs = [x[:self.max_len] for x in xs]
        t  = [[1] + t_e for t_e in tupl[1]]  # 1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        t  = [t_e[:self.max_len] for t_e in t]  # 1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        if shuffle:
            xs = [shuffle2N(x) for x in xs]
        return xs,t#,categ

    def __call__(self,y,tupl_1):
        pass




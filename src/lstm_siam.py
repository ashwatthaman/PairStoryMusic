from chainer import links as L
from chainer import functions as F

import os,sys,codecs
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from data_loader import loadSiamTest
from model_common import NNChainer
from util.vocabulary import Vocabulary
import random
random.seed(623)
import numpy as np
xp = np

try:
    import cupy as xp
except ImportError:
    import numpy as xp
except ModuleNotFoundError:
    import numpy as xp

class LSTMSiam(NNChainer):
    def __init__(self,args):
        self.setArgs(args)
        super(LSTMSiam, self).__init__(
                embed = L.EmbedID(self.n_vocab,self.n_embed),
                lstm = L.NStepBiLSTM(self.n_layers,self.n_embed,self.out_size,dropout=self.drop_ratio),
                bgm2h = L.Linear(self.bgm_h,2*self.out_size),
                bgm2h2 = L.Linear(2*self.out_size,2*self.out_size),
                h2w = L.Linear(2*self.out_size,2),
        )
        self.setDevice(args)

    def setArgs(self,args):
        if args.train:
            input_list,label_list,vocab = loadSiam([],args)
            self.input_list = input_list
            self.label_list = label_list
        if not args.train:
            vocab = Vocabulary.load("{}/vocab_h2_30000.bin".format(args.dataname))
            self.testbgmfeature = loadSiamTest()

        self.pwd = CDIR
        self.bgm_h = args.bgm_h
        self.vocab = vocab
        super().setArgs(args)

    def setVocab(self,args):
        pass
        # self.vocab = self.vocab

    def extractText(self,tupl_x):
        return tupl_x[0]

    def extractBGM(self,tupl_x):
        return tupl_x[1]

    def extractBGMName(self,tupl_x):
        return tupl_x[2]

    def extractVisCols(self):
        return "投稿,予測float,予測,正解,BGMName\n"

    # def extractVisAttr(self,tupl_x,t_list,y_list):
    def extractVisAttr(self,tupl_x,t_list):
        y = self.predict(tupl_x)
        y_list = y.data.argmax(1).tolist()
        y = F.softmax(y,axis=1)
        y_float_list = xp.split(y.data,axis=1,indices_or_sections=2)
        y_float_list = [y[0] for y in y_float_list[1]]
        x_list=self.extractText(tupl_x)
        bgmn_list = self.extractBGMName(tupl_x)
        line_list = []
        for x, y_fl,y, t,bgmn in zip(x_list,y_float_list, y_list, t_list,bgmn_list):
            txt = "\n".join([" ".join([self.vocab.itos(id) for id in x_e]) for x_e in x])
            line_str = "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n".format(txt,y_fl, y, t,bgmn)
            line_list.append(line_str)
        return line_list,t_list,y_list


    def getTrDvTe(self,args,test_ratio=0.1):
        kl = int(1./test_ratio)

        xs_list,y_list = self.input_list,self.label_list

        assert len(y_list)==len(xs_list)
        ind_arr = list(range(len(y_list)))
        random.shuffle(ind_arr)
        xs_list = [xs_list[ind] for ind in ind_arr]
        y_list = [y_list[ind] for ind in ind_arr]

        test_inds = [ind for ii,ind in enumerate(ind_arr) if ii%kl==args.cv]
        dev_inds  = [ind for ii,ind in enumerate(ind_arr) if ii%kl==(args.cv+1)%kl]
        tr_inds   = [ind for ii,ind in enumerate(ind_arr) if ii%kl!=args.cv and ii%kl!=(args.cv+1)%kl]

        assert len(set(tr_inds).intersection(set(dev_inds)))  ==0
        assert len(set(tr_inds).intersection(set(test_inds))) ==0
        assert len(set(dev_inds).intersection(set(test_inds)))==0

        xs_tr = [xs_list[tri] for tri in tr_inds]
        xs_dv = [xs_list[dei] for dei in dev_inds]
        xs_te = [xs_list[tei] for tei in test_inds]

        y_tr = [y_list[tri] for tri in tr_inds]
        y_dv = [y_list[dei] for dei in dev_inds]
        y_te = [y_list[tei] for tei in test_inds]

        print("tr:{},dv:{},te:{}".format(len(xs_tr),len(xs_dv),len(xs_te)))
        return xs_tr,y_tr,xs_dv,y_dv,xs_te,y_te


    def __call__(self,tupl):
        tupl_x = tupl[0];t =  tupl[1]
        t_all = xp.array(t,dtype=xp.int32)
        # print("call_tall:{}".format(Counter(t_all.tolist()).most_common()))
        ys_w = self.predict(tupl_x)
        loss = F.softmax_cross_entropy(ys_w, t_all,ignore_label=-1)  # /len(t_all)
        return loss


    def encode(self,xs):
        section_hi = [len(x) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs_conc = [x_e for x in xs for x_e in x]
        xs_emb = self.makeEmbedBatch(xs_conc)
        hx,_,_ = self.lstm(None,None,xs_emb)
        hx_b,hx_f = F.split_axis(hx,axis=0,indices_or_sections=2)
        hx = F.concat((hx_b,hx_f),axis=2)
        ys_txt = F.split_axis(hx,axis=1,indices_or_sections=sections)
        y_sum = [F.sum(y_txt,axis=1) for y_txt in ys_txt]
        y_txt = F.concat(y_sum,axis=0)
        return y_txt


    def predict(self,tupl_x):
        x_txt = self.extractText(tupl_x)
        x_bgm = self.extractBGM(tupl_x)

        y_txt = self.encode(x_txt)
        x_bgm = xp.array(x_bgm,dtype=xp.float32)
        y_bgm= self.bgm2h(x_bgm)
        y_bgm= self.bgm2h2(F.tanh(y_bgm))
        ys_w = self.h2w(F.tanh(y_txt*y_bgm))
        return ys_w

    def test(self,tupl_x):
        x_txt = self.extractText(tupl_x)

        # for ti,txt_e in enumerate(x_txt):
        #     print(txt_e)

        len_test = len(self.testbgmfeature)

        x_bgmf = [vec for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_bgmn = [key for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_txt = [t_e[:]  for t_e in x_txt for _ in range(len_test)]
        t_list = [0]*len(x_txt)
        tupl_x = (x_txt,x_bgmf,x_bgmn)

        line_list,t_list,y_list=self.extractVisAttr(tupl_x,t_list)
        return line_list

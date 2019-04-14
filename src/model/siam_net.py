
import os,sys,codecs
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from data_loader import loadSiamTest,loadVocab,loadTest
from model.model_common import NNChainer
from model.embedding import WordEmbeddings
import random
random.seed(623)
import numpy as np
xp = np
from chainer import links as L
from chainer import functions as F

try:
    import cupy as xp
except ImportError:
    import numpy as xp
except ModuleNotFoundError:
    import numpy as xp


class SiamNet(NNChainer):
    def __init__(self,args):
        self.setArgs(args)
        super(SiamNet, self).__init__(
                embed = WordEmbeddings(self.vocab,self.n_embed), #L.EmbedID(self.n_vocab,self.n_embed),
                lstm = L.NStepBiLSTM(self.n_layers,self.n_embed,self.n_hidden,dropout=0.5),#self.drop_ratio),
                bgm2h = L.Linear(self.bgm_h,self.n_hidden//2),
                bgm2h2 = L.Linear(self.n_hidden//2,2*self.n_hidden),
                h2w = L.Linear(2*self.n_hidden,2),
        )
        self.setDevice(args)

    def setArgs(self,args):
        parasol_softs = ['majicara','delivara','haruno', 'kanoren', 'koiimo', 'qsplash', 'sakura', 'yumekoi']
        inre_softs = ['bokukimi', 'chusingura', 'chusingura_fd', 'miburo']
        uguisu_softs = ["kaminoue", "suisoginka"]
        alcot_softs = ["onigokko", "onigokko_fd", "fair_child", "natsupochi", "daitouryou", "daitouryou_fd",
                       "realimouto"]
        honeycomb_softs = ["1_2summer","kicking_horse"]
        akabee_softs = ["okibaganai", "yayaokibaganai","sono_yokogao","lavender","konboku"]
        title_list = parasol_softs + inre_softs + uguisu_softs + alcot_softs + honeycomb_softs + akabee_softs
        print('title_len',len(title_list))
        if args.train:
            self.input_train, self.label_train, self.input_dev, self.label_dev, self.input_test, self.label_test, self.vocab = loadSiam17and1(title_list,args,args.cv)
        else:
            self.vocab = loadVocab(args)
            self.testbgmfeature = loadSiamTest()

        self.pwd = CDIR
        self.bgm_h = args.bgm_h
        super().setArgs(args)

    def setVocab(self,args):
        pass
        # self.vocab = self.vocab

    def extractText(self,tupl_x):
        # return [tupl[0] for tupl in tupl_x]
        return tupl_x[0]

    def extractBGM(self,tupl_x):
        # return [tupl[1] for tupl in tupl_x]
        return tupl_x[1]

    def extractBGMName(self,tupl_x):
        return tupl_x[2]

    def write_output_col(self):
        return "投稿,予測float,予測,正解,BGMName".split(',')

    def write_output_each(self,tupl):
        tupl_x = tupl[0];t_list =  tupl[1]
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
            # line_str = "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n".format(txt,y_fl, y, t,bgmn)
            line_str = [txt,y_fl, y, t,bgmn]
            line_list.append(line_str)
        return line_list,t_list,y_list


    def getTrDvTe(self,args):
        return self.input_train, self.label_train, self.input_dev, self.label_dev, self.input_test, self.label_test#loadSiam17and1(title_list,args,args.cv)

    def __call__(self,tupl):
        tupl_x = tupl[0];t =  tupl[1]
        t_all = xp.array(t,dtype=xp.int32)
        ys_w = self.predict(tupl_x,predict=False)
        loss = F.softmax_cross_entropy(ys_w, t_all,ignore_label=-1)  # /len(t_all)
        return loss

    # compute text feature
    def encode(self,xs):
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs_conc = [x_e for x in xs for x_e in x]
        xs_emb = self.embed.embedBatch(xs_conc)
        hx,_,_ = self.lstm(None,None,xs_emb)

        hx_b,hx_f = F.split_axis(hx,axis=0,indices_or_sections=2)
        hx = F.concat((hx_b,hx_f),axis=2)
        hx = F.sum(hx,axis=0)

        ys_txt = F.split_axis(hx,axis=0,indices_or_sections=sections)
        ys_txt = F.pad_sequence(ys_txt)

        y_sum = F.sum(ys_txt,axis=1)
        return y_sum


    def predict(self,tupl_x,predict=True):
        x_txt = self.extractText(tupl_x)
        y_txt = self.encode(x_txt)
        x_bgm = self.extractBGM(tupl_x)
        x_bgm = xp.array(x_bgm,dtype=xp.float32)
        y_bgm= self.bgm2h(F.tanh(x_bgm))
        y_bgm= self.bgm2h2(F.tanh(y_bgm))
        if not predict:y_bgm =F.dropout(y_bgm)
        ys_w = self.h2w(F.tanh(y_txt*y_bgm))
        return ys_w

    def test(self,tupl_x):
        x_txt = self.extractText(tupl_x)
        len_test = len(self.testbgmfeature)
        x_bgmf = [vec for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_bgmn = [key for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_txt = [t_e[:]  for t_e in x_txt for _ in range(len_test)]
        t_list = [0]*len(x_txt)
        tupl_x = (x_txt,x_bgmf,x_bgmn)

        line_list,t_list,y_list=self.write_output_each([tupl_x,t_list])
        return line_list

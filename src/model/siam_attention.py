
import os,sys,codecs
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from data_loader import loadSiamTest#,loadSiam17and1
from model.model_common import NNChainer
from model.embedding import WordEmbeddings
from model.attention import Attention
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

class AttSiam(NNChainer):
    def __init__(self,args,vocab):
        self.setArgs(args)
        self.setVocab(vocab)
        super(AttSiam, self).__init__(
                embed  = WordEmbeddings(vocab,self.n_embed,self.make_xp_array), #L.EmbedID(self.n_vocab,self.n_embed),
                lstm   = L.NStepBiLSTM(self.n_layers,self.n_embed,self.n_hidden,dropout=0.5),#self.drop_ratio),
                att    = Attention(2*self.n_hidden),
                bgm2h  = L.Linear(self.bgm_h,self.n_hidden),
                bgm2h2 = L.Linear(self.n_hidden,2*self.n_hidden),
                h2w    = L.Linear(2*self.n_hidden,2),
        )
        self.setDevice(args)

    def setArgs(self,args):
        if not args.train:
            self.testbgmfeature = loadSiamTest()
        self.pwd = CDIR
        self.bgm_h = args.bgm_h
        super().setArgs(args)

    def setVocab(self,vocab):
        self.vocab = vocab

    def extractText(self,tupl_x):
        # return [tupl[0] for tupl in tupl_x]
        return tupl_x[0]

    def extractBGM(self,tupl_x):
        # return [tupl[1] for tupl in tupl_x]
        return tupl_x[1]

    def extractBGMName(self,tupl_x):
        return tupl_x[2]

    def extractId(self,tupl_x):
        return tupl_x[3]

    def write_output_col(self):
        return "id,att_scores,文（１０文単位）,予測float,予測,正解,BGMName,att_score,att_sentence".split(',')

    def write_output_each(self,tupl):
        tupl_x = tupl[0];t_list =  tupl[1]
        y,att_w = self.predict(tupl_x)
        y_list = y.data.argmax(1).tolist()
        y = F.softmax(y,axis=1)
        # y_float_list = xp.split(y.data,axis=1,indices_or_sections=2)
        y_float_list = [y.data for y in F.split_axis(y,axis=1,indices_or_sections=2)]
        y_float_list = [y[0] for y in y_float_list[1]]
        x_list=self.extractText(tupl_x)
        bgmn_list = self.extractBGMName(tupl_x)
        id_list = self.extractId(tupl_x)
        line_list = []
        sgn = lambda x:round(float(x),4)
        for id_e,x, y_fl,y, t,bgmn,att_w_e in zip(id_list,x_list,y_float_list, y_list, t_list,bgmn_list,att_w):
            att_w_e = F.flatten(att_w_e).data[:len(x)]
            att_ind = int(att_w_e.argmax(axis=0))
            att_scr = att_w_e.max(axis=0)
            att_scores = '\n'.join(
                ["{}".format(sgn(att_w_e[ai])) for ai in range(len(x))])
            most_post = self.vocab.idlist_to_sentence(x[att_ind]).replace('"', '')

            txt = "\n".join([self.vocab.idlist_to_sentence(x_e) for x_e in x])
            line_str = [id_e,att_scores,txt,y_fl, y, t,bgmn,att_scr,most_post]
            line_list.append(line_str)
        return line_list,t_list,y_list

    def __call__(self,tupl):
        tupl_x = tupl[0];t =  tupl[1]
        t_all = self.make_xp_array(t)

        ys_w,att_w = self.predict(tupl_x,predict=False)
        loss = F.softmax_cross_entropy(ys_w, t_all,ignore_label=-1)  # /len(t_all)
        return loss

    def encode_text(self,xs):
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs_conc = [x_e for x in xs for x_e in x]
        xs_emb = self.embed.embedBatch(xs_conc)
        hx,_,_ = self.lstm(None,None,xs_emb)
        hx_b,hx_f = F.split_axis(hx,axis=0,indices_or_sections=2)
        hx = F.concat((hx_b,hx_f),axis=2)
        hx = F.sum(hx,axis=0)
        ys_txt = F.split_axis(hx,axis=0,indices_or_sections=sections)
        return ys_txt

    def encode_bgm(self,x_bgm,predict):
        x_bgm = self.make_xp_array(x_bgm,type='float')
        y_bgm = self.bgm2h(F.tanh(x_bgm))
        # if not predict:y_bgm =F.dropout(y_bgm)
        y_bgm = self.bgm2h2(F.tanh(y_bgm))
        if not predict: y_bgm = F.dropout(y_bgm)
        return y_bgm

    def predict(self,tupl_x,predict=True):
        y_bgm = self.encode_bgm(self.extractBGM(tupl_x),predict)
        ys_txt = self.encode_text(self.extractText(tupl_x))
        ys_txt = F.pad_sequence(ys_txt)
        y_txt,att_w = self.att(ys_txt,y_bgm)
        ys_w = self.h2w(F.tanh(y_txt*y_bgm))
        return ys_w,att_w

    def test(self,tupl_x):
        x_txt = self.extractText(tupl_x)
        id_list = self.extractId(tupl_x)
        len_test = len(self.testbgmfeature)
        x_bgmf = [vec for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_bgmn = [key for _ in range(len(x_txt)) for key,vec in self.testbgmfeature.items()]
        x_txt = [t_e[:]  for t_e in x_txt for _ in range(len_test)]
        id_list = [id_ for id_ in id_list for _ in range(len_test)]
        t_list = [0]*len(x_txt)
        tupl_x = (x_txt,x_bgmf,x_bgmn,id_list)
        # tupl_x = [(x_t,x_bt,x_bn) for x_t,x_bt,x_bn in zip(x_txt,x_bgmf,x_bgmn)]
        line_list, t_list, y_list = self.write_output_each([tupl_x, t_list])
        return line_list

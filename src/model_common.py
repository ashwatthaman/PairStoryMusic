from chainer import Variable
from chainer import cuda
from chainer import functions as F
from chainer import links as L

import numpy as np
if cuda.cudnn_enabled:
    import cupy as xp
else:
    global xp
    xp=np

from chainer import Chain
import os,sys
from chainer import serializers
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from util.vocabulary import Vocabulary,vocabularize,labelize

import chainer

import util.generators as gens
import random
from gensim.models import word2vec

class NNChainer(Chain):

    def __init__(self,**args):
        super(NNChainer, self).__init__(**args)

    # def setParams(self,args):
    def setArgs(self,args):
        self.n_vocab = args.n_vocab
        self.n_embed = args.embed
        self.n_layers = args.layer
        self.out_size = args.hidden
        self.drop_ratio = args.dropout
        if args.gpu>=0:
            import cupy as xp

        self.setBatchSize(args.batchsize)
        self.setVocab(args)
        self.setMaxEpoch(args.epoch)
        self.setEpochNow(0)

    def setAttr(self,attr_list,attr_visi,txt_attr):
        self.attr_list = attr_list
        self.visible_attr_list = attr_visi
        self.txt_attr = txt_attr


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


    def setText(self,text_arr):
        self.text_arr = text_arr

    def setVocab(self, args):
        vocab_name = "./{}/vocab_{}.bin".format(args.dataname, args.dataname)
        set_vocab = set()

        sent_new_arr,vocab = vocabularize(self.text_arr,vocab_size=args.n_vocab,normalize=False)
        self.setText(sent_new_arr)
        n_vocab = len(set_vocab) + 3
        print("n_vocab:{}".format(n_vocab))
        print("arg_vocab:{}".format(args.n_vocab))
        # src_vocab = Vocabulary.new(self.text_arr, args.n_vocab)
        vocab.save(vocab_name)
        self.vocab = vocab
        return vocab

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
            model_name_tmp = args.model_name.format(e)
            # print("model_tmp",model_name_tmp)
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
                print("wordvec model doesnt exists.")
        return first_e

    def makeEmbedBatch(self,xs):
        xs = [xp.asarray(x,dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs


    def getBatchGen(self,args):
        ind_arr = list(range(len(self.tt_list)))
        random.shuffle(ind_arr)
        tt_now = (self.tt_list[ind] for ind in ind_arr)
        cat_now = (self.cat_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.batchsize)
        cat_gen = gens.batch(cat_now, args.batchsize)
        for tt,cat in zip(tt_gen,cat_gen):
            yield (tt,cat)

    def loadW(self, premodel_name):
        src_vocab = self.vocab
        src_w2ind,src_ind2w = {},{}
        src_size = len(self.vocab)
        for vi in range(src_size):
            src_ind2w[vi] = src_vocab.itos(vi)
            src_w2ind[src_ind2w[vi]] = vi
        self.embed.W.data = xp.array(transferWordVector(src_w2ind, src_ind2w, premodel_name), dtype=xp.float32)

    def setClassWeights(self,cat_list):
        count_hash = getPosNegWeightHash(cat_list)
        self.weights = xp.array([count_hash[ci] for ci in range(self.class_n)], dtype=xp.float32)



def transferWordVector(w2ind_post, ind2w_post, premodel_name):
    premodel = word2vec.Word2Vec.load(premodel_name).wv
    vocab = premodel.vocab
    word = ""
    error_count=0
    print("ind2len:" + str(len(ind2w_post)))
    for ind in range(len(ind2w_post)):
        try:
            vocab[ind2w_post[ind]]
            word = ind2w_post[ind]
        except:
            error_count += 1
    sims = premodel.most_similar(word,topn=5)
    if '<unk>' not in vocab:
        unk_ind = vocab[word]
    else:
        unk_ind = vocab["<unk>"]
    print("unk_ind:"+str(unk_ind))
    print("errcount:"+str(error_count))
    W = [premodel.syn0norm[vocab.get(ind2w_post[ind],unk_ind).index].tolist() for ind in range(len(ind2w_post))]
    return W

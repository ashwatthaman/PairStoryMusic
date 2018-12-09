import numpy as np
from gensim.models import word2vec

import numpy as np
from chainer import cuda
if cuda.cudnn_enabled:
    import cupy as xp
else:
    global xp
    xp=np
from chainer import Chain
from chainer import links as L
from chainer import functions as F
#@profile由来のエラー除去
try:
    profile
except NameError:
    def profile(func):
        def inner(*args,**kwargs):
            return func(*args,**kwargs)
        return inner

class WordEmbeddings(Chain):
    # def __init__(self,n_vocab,n_embed):
    # self.setArgs(n_vocab,n_embed)
    def __init__(self,vocab,n_embed):
        self.setArgs(vocab,n_embed)
        super(WordEmbeddings,self).__init__(
            embed=L.EmbedID(self.n_vocab, self.n_embed),
        )

    def setArgs(self,vocab,n_embed):
        self.vocab   = vocab
        self.n_vocab = len(vocab)
        self.n_embed = n_embed

    @profile
    def __call__(self,xs,reverse=False):
        return self.embed(xs)

    def embedBatch(self,xs,reverse=False):
        if reverse:xs = [x[::-1] for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs = xp.array([x_e for x in xs for x_e in x],dtype=xp.int32)
        # print('xs_type',xs.__class__.__name__)
        embed_list = F.split_axis(self.embed(xs), sections, axis=0)
        return embed_list

    #gensimで作ったword2vecのベクトルをEmbed層に転移します。
    def loadW(self,premodel_name):
        vocab = self.vocab
        w2ind = {}
        ind2w = {}
        vocab_size = len(self.vocab)
        for vi in range(vocab_size):
            ind2w[vi] = vocab.itos(vi)
            w2ind[ind2w[vi]] = vi
        self.embed.W.data = xp.array(transferWordVector(w2ind, ind2w, premodel_name), dtype=xp.float32)


def transferWordVector(w2ind_post, ind2w_post, premodel_name):
    premodel = word2vec.Word2Vec.load(premodel_name).wv
    vocab = premodel.vocab
    word = ""#"the"
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
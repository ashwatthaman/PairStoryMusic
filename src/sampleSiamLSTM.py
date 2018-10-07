import sys,os,codecs
sys.path.append("../")
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from lstm_siam import LSTMSiam
from data_loader import loadTest
# from traintest import trainDev
import argparse

class Args():
    def __init__(self,train=True,cv=1,mecab=True):
        self.cv = cv
        self.mecab=mecab
        self.train=train
        self.outtype="cls"
        self.epoch = 30
        if mecab:
            self.n_vocab = 30000
        else:
            self.n_vocab = 16550
        self.embed = 128#2#128
        self.bgm_h = 54
        self.categ_size=475#221
        #points directory to transfer gensim w2v model
        self.premodel=""

        self.hidden= 128
        self.n_latent = 60#0
        self.layer = 1
        self.batchsize=5#64#5#32#6#0
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        self.n_class = 2
        if train:
            self.gpu = -1#0
        else:
            self.gpu = -1
        self.gradclip = 5
        self.model_name_base = "LSTMSiam_e{}_h{}_cv{}".format(self.embed,self.hidden,self.cv)
        if mecab:self.model_name_base+='_mecab'
        self.dataname = self.model_name_base
        if not os.path.exists(self.dataname):os.mkdir(self.dataname)
        if not os.path.exists("{}/model".format(self.dataname)):os.mkdir("{}/model".format(self.dataname))
        self.model_dir = CDIR + "./{}/model/".format(self.dataname)
        self.model_name_base+= "_e{}"
        self.model_name = self.model_dir + self.model_name_base+'.npz'  # +"_man"

def sampleTrain():
    args = Args(True)
    model = LSTMSiam(args)
    x_tr, y_tr, x_dv, y_dv, x_te, y_te = model.getTrDvTe(args)
    model.vocab.save("./vocab_h2_{}.bin".format(args.n_vocab))
    trainDev(args, model, x_tr, y_tr, x_dv, y_dv, x_te, y_te)


def sampleTest(txtfile):
    args = Args(False)
    model = LSTMSiam(args)
    model.loadModel(args)

    xs_list = loadTest(txtfile,vocab=model.vocab)
    y_list = [0]*len(xs_list)
    model.setData(xs_list,y_list)
    fw = codecs.open(CDIR+"./test_bgm/test_result.csv","w")
    # model.batchsize = 3
    # batchsize = 3
    for bi,batch in enumerate(model.getBatchGen(args)):
        line_list=model.test(batch[0])
        [fw.write(line) for line in line_list]
        if bi==30:break
    fw.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--train",
                        help="train mode",
                        action="store_true")
    args = parser.parse_args()
    # if args.train:
    # sampleTrain()
    sampleTest("./test_bgm/meros.txt")

import sys,os,codecs,csv
sys.path.append("../")
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from data_loader import loadTest
from model.siam_net import SiamNet

from traintest import train_dev_test
import argparse

class Args():
    def __init__(self,model_name,train=False,cv=6,mecab=True):

        self.cv = cv
        self.mecab=mecab
        self.train=train
        self.outtype="cls"
        self.epoch = 30
        if mecab:
            self.n_vocab = 30000
        else:
            self.n_vocab = 16550
        self.n_embed = 64#2#128
        self.bgm_h = 54
        #points directory to transfer gensim w2v model
        self.premodel=""

        self.n_hidden= 64#
        self.max_len = 200
        self.layer = 1
        self.batchsize=256
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        self.n_class = 2
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 5

        self.model_dir = model_name
        self.dataname = model_name + "_e{}_h{}_cv{}".format(self.n_embed, self.n_hidden,self.cv)
        self.model_name_base = self.dataname + '_e{}'
        if mecab:self.model_name_base+='_mecab'

        if not os.path.exists(model_name):           os.mkdir(model_name)
        if not os.path.exists(model_name+'/'+self.dataname):           os.mkdir(model_name+'/'+self.dataname)
        if not os.path.exists(model_name+'/'+self.dataname + '/output'): os.mkdir(model_name+'/'+self.dataname + '/output')
        if not os.path.exists(model_name+'/'+self.dataname + '/model'): os.mkdir(model_name+'/'+self.dataname + '/model')
        if not os.path.exists(model_name+'/'+self.dataname + '/log'):  os.mkdir(model_name+'/'+self.dataname + '/log')
        self.model_name = "./{}/{}/model/".format(model_name,self.dataname) + self.model_name_base  # + ".npz"


def sampleTrain(cv):
    args = Args('SiamMax',True,cv)
    model = SiamNet(args)
    x_tr, y_tr, x_dv, y_dv, x_te, y_te = model.getTrDvTe(args)
    model.vocab.save("./{}/{}/vocab_{}.bin".format(args.model_dir,args.dataname,args.n_vocab))
    train_dev_test(args, model, x_tr, y_tr, x_dv, y_dv, x_te, y_te)


def sampleTest(cv,txtfile):
    args = Args('SiamMax',False,cv)
    model = SiamNet(args)
    model.loadModel(args)

    xs_list = loadTest(txtfile,vocab=model.vocab)
    y_list = [0]*len(xs_list)
    model.setData(xs_list,y_list)
    fw = csv.writer(codecs.open(CDIR+"./test_bgm/test_result.csv","w"))
    for bi,batch in enumerate(model.getBatchGen()):
        line_list=model.test(batch[0])
        # [fw.write(line) for line in line_list]
        [fw.writerow(line) for line in line_list]
        if bi==30:break
    # fw.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i","--input",
                        help="input file",
                        default="./test_bgm/meros.txt"
                        )
    args = parser.parse_args()

    sampleTest(6,args.input)

import sys,os,codecs
sys.path.append("../")
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
# from LSTMSiam import LSTMSiam
from model.NStepLSTMSiam import NStepLSTMSiam

from traintest import train_dev_test
import argparse

class Args():
    def __init__(self,model_name,train=True,cv=0,mecab=True):

        self.cv = cv
        self.mecab=mecab
        self.train=train
        # self.source ="./{}/all_{}16000_fixed.txt".format(dataname,dataname)
        # self.category="./{}/all_chara.txt".format(dataname)
        self.outtype="cls"
        self.epoch = 30
        if mecab:
            self.n_vocab = 30000
        else:
            self.n_vocab = 16550
        self.n_embed = 64#2#128
        self.bgm_h = 54
        #self.categ_size=475#221
        #points directory to transfer gensim w2v model
        self.premodel=""

        self.n_hidden= 64#3#128
        #self.n_latent = 60#0
        self.max_len = 200
        self.layer = 1
        self.batchsize=256#5#32#6#0
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

        # if not os.path.exists(self.dataname):           os.mkdir(self.dataname)
        # if not os.path.exists(self.dataname + '/output'): os.mkdir(self.dataname + '/output')
        # if not os.path.exists(self.dataname + '/model'): os.mkdir(self.dataname + '/model')
        # if not os.path.exists(self.dataname + '/log'):  os.mkdir(self.dataname + '/log')
        # self.model_name = "./{}/model/".format(self.dataname) + self.model_name_base  # + ".npz"



def sampleTrain(cv):
    args = Args('LSTMMaxMore',True,cv)

    model = NStepLSTMSiam(args)
    x_tr, y_tr, x_dv, y_dv, x_te, y_te = model.getTrDvTe(args)
    # for ei,(x_e,y_e) in enumerate(zip(x_tr,y_tr)):
    #     print('x_txt',[" ".join([model.vocab.itos(word_id) for word_id in x_e2]) for x_e2 in x_e[0]])
    #     print('x_bgm',x_e[2])
    #     print('y_e',y_e)
    #     if ei==5:break
    model.vocab.save("./{}/{}/vocab_{}.bin".format(args.model_dir,args.dataname,args.n_vocab))
    train_dev_test(args, model, x_tr, y_tr, x_dv, y_dv, x_te, y_te)


def sampleTest(txtfile):
    args = Args(False)
    # model = LSTMSiam(args)
    model = NStepLSTMSiam(args)
    model.loadModel(args)

    xs_list = loadTest(txtfile,args.mecab,vocab=model.vocab)
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
    # sampleTrain(0)

    for ci in range(28):
        # try:
        sampleTrain(ci)
        # except:pass
    # print("this")
    # sampleTest("./test_bgm")

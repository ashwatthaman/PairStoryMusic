import pandas,os,codecs,subprocess
from util.vocabulary import vocabularize,Vocabulary

bgm_feature_dim=54
text_split_unit=10

def loadSiamTest():
    bgm_file = "test_bgm/bgmfeature_test_norm.csv"
    df_bgm = pandas.read_csv(bgm_file)
    bgmfeature_dict = {row[1].replace(".ogg",""):list(row[2:]) for ri,row in df_bgm.iterrows()}
    return bgmfeature_dict

def loadVocab(args):
    vocab_dir = args.model_dir+"/"+args.dataname+"/vocab_{}.bin".format(args.n_vocab)
    vocab = Vocabulary.load(vocab_dir)
    return vocab

def loadTest(txtfile,vocab):
    serif_delim = "「"
    if not os.path.exists(txtfile.replace(".txt", "_mecab.txt")):
        subprocess.call("mecab -O wakati {} > {}".format(txtfile,txtfile.replace(".txt","_mecab.txt")),shell=True)
    lines = [line.replace("▁","").strip() for line in codecs.open(txtfile.replace(".txt","_mecab.txt"),"r",encoding="utf-8").readlines()]
    fr=[]
    for line in lines:
        if serif_delim in line:
            if line.index(serif_delim)<10:
                fr.append(line)
            else:fr+=[l_e+'。' for l_e in line.split('。') if len(l_e)>3]
        else:fr+=[l_e+'。' for l_e in line.split('。') if len(l_e)>3]
    text_list,vocab = vocabularize([fr],vocab=vocab)
    json_list=[(lines,[0]*bgm_feature_dim,[""]) for lines in zip(*[iter(text_list[0])]*text_split_unit)]
    return json_list
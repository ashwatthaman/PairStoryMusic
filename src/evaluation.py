from collections import Counter
from sklearn.metrics import f1_score,precision_score,recall_score
import codecs,json
import numpy as np
import os,math,sys
sys.path.append("./")



def evalDiscrete(all_t,all_y):
    all_t = [t_e for y_e, t_e in zip(all_y, all_t) if t_e > -1]
    all_y = [y_e for y_e, t_e in zip(all_y, all_t) if t_e > -1]
    correct_num = len([1 for y_e, tag_e in zip(all_y, all_t) if y_e == tag_e])
    wrong_num = len([1 for y_e, tag_e in zip(all_y, all_t) if y_e != tag_e])
    if len(set(all_t))==2:
        avemethod='binary'
        eval_hash={}
        eval_hash["fscore"]=f1_score(all_t,all_y,average=avemethod)
        eval_val=eval_hash["fscore"]
        eval_hash["precision"]=precision_score(all_t,all_y,average=avemethod)
        eval_hash["recall"]=recall_score(all_t,all_y,average=avemethod)
        eval_hash["correct"]=correct_num
        eval_hash["wrong"]=wrong_num
        eval_hash["rateofone"]=round(sum(all_t) / len(all_t), 3)
    else:
        print('set_t',set(all_t))
        avemethod='micro'
        f_list=adjust_f1_score(all_t,all_y, average=None)
        eval_hash={fi:round(fscr,4) for fi,fscr in enumerate(f_list)}
        eval_hash['fmacro']=np.mean(f_list)
        eval_val=eval_hash['fmacro']
        eval_hash['count_p']=Counter(all_y).most_common()
        eval_hash['count_t']=Counter(all_t).most_common()
        print('f_list:{}'.format(f_list))
        print('fmacro:{}'.format(np.mean(f_list)))
    return eval_val,eval_hash

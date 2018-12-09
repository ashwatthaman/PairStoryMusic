from collections import Counter
from sklearn.metrics import f1_score,precision_score,recall_score
import codecs,json
import numpy as np
import os,math,sys
sys.path.append("./")
from rouge import Rouge

def evalRouge(all_t,all_y):
    rouge = Rouge()
    sgn = lambda x: round(float(x), 4)
    mean_scores=rouge.get_scores(all_y,all_t,avg=True)
    eval_hash={k+'-'+k2:sgn(v2) for k,v in mean_scores.items() for k2,v2 in v.items()}
    print('eval_hash',eval_hash)
    return eval_hash['rouge-l-r'],eval_hash


def evalDiscrete(all_t,all_y):
    # print('all_t;{}'.format(all_t))
    # print('all_y;{}'.format(all_y))
    all_t = [t_e for y_e, t_e in zip(all_y, all_t) if t_e > -1]
    all_y = [y_e for y_e, t_e in zip(all_y, all_t) if t_e > -1]
    # print("all_t:{}".format(Counter(all_t).most_common()))
    # print("all_y:{}".format(Counter(all_y).most_common()))
    correct_num = len([1 for y_e, tag_e in zip(all_y, all_t) if y_e == tag_e])
    wrong_num = len([1 for y_e, tag_e in zip(all_y, all_t) if y_e != tag_e])
    # print("corr:{},wron:{}".format(correct_num,wrong_num))
    # print("setallt:{}".format(set(all_t)))
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
        # print('count_p:{}'.format(eval_hash['count_p']))
        # print('count_t:{}'.format(eval_hash['count_t']))
    return eval_val,eval_hash

def evalRank(args,model,y_test_arr,y_pred_arr,x_list_arr,write,rank_n=3):
    filename="{}{}_{}".format(args.model_name_base.replace("_{}",""),args.manual,args.rep_thr)
    print("write_flag:{}".format(write))
    if write:
        fw = codecs.open(model.pwd+"./each_case/{}/output/result_{}_{}.csv".format(args.dataname,args.manual,filename), "w", encoding="utf-8")

        logw = codecs.open(model.pwd+"./each_case/log/log_{}.json".format(filename), "w", encoding="utf-8")
        log_hash = {}
        log_hash["args"]=args.__dict__

        # fw.write("subset_id,投稿,教師ランク,予測ランク,教師回帰,予測回帰,教師カルマ\n")
        fw.write(model.extractVisCols())
    else:
        fw = None
    def calcEval(y_test_arr,y_pred_arr,x_list_arr,fw,flag=False):
        t_rank_arr = [[] for ri in range(1, 10)];
        y_rank_arr = [[] for ri in range(1, 10)]
        subset_inds = (rj for ri in range(sum([len(y_test) for y_test in y_test_arr])) for rj in [ri] * 10)
        sumpredy = 0; mrr_arr=[]
        all_t = [];all_y = []
        for y_test,y_pred,x_list in zip(y_test_arr,y_pred_arr,x_list_arr):
            sumpredy += sum([p_e for y_p in y_pred for p_e in y_p])
            rank_yp = [val for yp_e in y_pred for val in makeRank(yp_e[:], rank_n)]
            rank_yt = [val for yt_e in y_test for val in makeRank(yt_e[:], rank_n)]
            for ri,rank_i in enumerate(range(1,10)):
                y_rank_arr[ri]+=[val for yp_e in y_pred for val in makeRank(yp_e[:], rank_i)]
                t_rank_arr[ri]+=[val for yt_e in y_test for val in makeRank(yt_e[:], rank_i)]
            y_raw_list=[val for yp_e in y_pred for val in yp_e]
            # y_attraw_list=[val for ya_e in y_attr for val in ya_e]
            t_raw_list=[val for yt_e in y_test for val in yt_e]
            for yp_e,yt_e in zip(y_pred[:],y_test[:]):
                mrr_arr.append(calcMRR(yt_e,yp_e))
            # print("y_raw_list:{}".format(y_raw_list[:10]))
            # print("t_raw_list:{}".format(t_raw_list[:10]))
            assert len(rank_yp)==len(rank_yt)
            x_list = [x_e for x in x_list for x_e in x]
            for x,t_rank,y_rank,t_raw,y_raw,sub_i in zip(x_list,rank_yt,rank_yp,t_raw_list,y_raw_list,subset_inds):
            # for x_in,t_rank,y_rank,t_raw,y_raw,sub_i in zip(x_list,rank_yt,rank_yp,t_raw_list,y_raw_list,subset_inds):
                x=x.replace('\n','')
                if flag:
                    if write:
                        fw.write("{},{},{},{}\n".format(sub_i,x,t_rank,y_rank))
            all_y+=rank_yp;all_t+=rank_yt

        if len(mrr_arr) > 0:
            mrr = sum(mrr_arr) / len(mrr_arr)
        else:
            mrr = 0
        eval_hash = {}
        map_arr = []
        for ri, rank_i in enumerate(range(1, 10)):
            eval_val_tmp, eval_hash_tmp = evalDiscrete(t_rank_arr[ri], y_rank_arr[ri])
            eval_hash["prec{}".format(rank_i)] = eval_val_tmp
            map_arr.append(eval_val_tmp)
        eval_val, eval_hash_tmp = evalDiscrete(all_t, all_y)
        eval_hash["rateofone"] = eval_hash_tmp["rateofone"]
        eval_hash["correct"] = eval_hash_tmp["correct"]
        eval_hash["wrong"] = eval_hash_tmp["wrong"]
        eval_hash["mrr"] = round(mrr, 3)
        eval_hash["map"] = round(sum(map_arr) / len(map_arr), 3)
        # return all_y,all_t,mrr_arr
        return eval_hash

    eval_arr = ["rateofone","mrr","map","correct","wrong"]+["prec{}".format(ki) for ki in range(1,10)]
    eval_all_hash = {eval_str:[] for eval_str in eval_arr}
    flag=True
    for _ in range(100):
        eval_hash = calcEval(y_test_arr,y_pred_arr,x_list_arr,fw,flag)
        flag=False
        for eval_str in eval_all_hash:
            eval_all_hash[eval_str].append(eval_hash[eval_str])
    eval_mean_hash = {}
    for eval_str in eval_all_hash:
        mean_eval = sum(eval_all_hash[eval_str])/len(eval_all_hash[eval_str])
        eval_mean_hash[eval_str]=mean_eval
        print("{}, max:{},mean:{}, min:{}".format(eval_str,max(eval_all_hash[eval_str]),mean_eval,min(eval_all_hash[eval_str])))

    eval_val = eval_mean_hash["prec3"]
    # print(" correct:{}".format(correct_num))
    # print(" wrong:{}".format(wrong_num))
    if write:
        log_hash["eval"] = eval_mean_hash
        log_hash["pred_test"] = {"pred":y_pred_arr,"test":y_test_arr}
        #### [logw.write("{}:{}\n".format(key,val)) for key,val in eval_hash.items()]
        fw.close()
        logw.write(json.dumps(log_hash,default=lambda x:x.__class__.__name__))
        logw.close()
    return eval_val

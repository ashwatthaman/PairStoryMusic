import pandas,os,codecs,glob
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")


def load_csv(csvname,label="正解"):
    df = pandas.read_csv(csvname)

    text_list = []
    id_set = sorted(set(df["id"]))
    bgm_list = []
    for id_ in id_set:
        df_tmp = df[df["id"]==id_]
        bgmn_score_tupl = [(bgmn,scr) for bgmn,scr in zip(df_tmp["BGMName"],df_tmp["予測float"])]
        bgmn_score_tupl = sorted(bgmn_score_tupl,key=lambda x:x[1],reverse=True)
        bgm_list.append(bgmn_score_tupl[0][0])
        text_list.append(list(df_tmp["投稿"])[0])
    return text_list,bgm_list



# def convert2html(book_title):
def convert2html(csvname):
    # csvname = "../test_bgm/{}_result.csv".format(book_title)
    book_title = csvname.split("/")[-1].replace("_result.csv","")
    audio_tag = "<br><br><audio src=\"{}\" controls></audio><br>"

    text_list, bgm_list = load_csv(csvname)
    bgm_dir = CDIR+"../test_bgm/"
    if not os.path.exists((bgm_dir+"html")):os.mkdir(bgm_dir+"html")

    html_path = bgm_dir+"html/main_{}.html"
    html_base = codecs.open(html_path.format("base"), encoding="utf-8").read()

    html_write = codecs.open(html_path.format(book_title), "w", encoding="utf-8")
    html_write.write(html_base)

    for text,bgm in zip(text_list,bgm_list):
        text = text.replace("<unk>","unk")
        # print(bgm)
        if os.path.exists(bgm_dir+bgm):
            # print(bgm_dir)
            html_write.write(audio_tag.format(bgm_dir + bgm))
            html_write.write(text.replace("\n", "\n <br>"))


    html_write.write("</head>\n")
    html_write.write("<body>\n")
    html_write.close()

if __name__=="__main__":
    csvfiles = glob.glob("../test_bgm/*_result.csv")

    # for book_title in ["meros","morg"]:
    for csvfile in csvfiles:
        convert2html(csvfile)


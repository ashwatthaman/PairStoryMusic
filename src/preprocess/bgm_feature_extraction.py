import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import pandas
'''
    function: extract_features
    input: path to mp3 files
    output: csv file containing features extracted
    
    This function reads the content in a directory and for each mp3 file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
'''
def extract_feature(path):
    id = 1  # Song ID
    feature_set = pd.DataFrame()  # Feature Matrix
    
    # Individual Feature Vectors
    songname_vector = pd.Series()
    tempo_vector = pd.Series()
    total_beats = pd.Series()
    average_beats = pd.Series()
    chroma_stft_mean = pd.Series()
    chroma_stft_std = pd.Series()
    chroma_stft_var = pd.Series()
    chroma_cq_mean = pd.Series()
    chroma_cq_std = pd.Series()
    chroma_cq_var = pd.Series()
    chroma_cens_mean = pd.Series()
    chroma_cens_std = pd.Series()
    chroma_cens_var = pd.Series()
    mel_mean = pd.Series()
    mel_std = pd.Series()
    mel_var = pd.Series()
    mfcc_mean = pd.Series()
    mfcc_std = pd.Series()
    mfcc_var = pd.Series()
    mfcc_delta_mean = pd.Series()
    mfcc_delta_std = pd.Series()
    mfcc_delta_var = pd.Series()
    rmse_mean = pd.Series()
    rmse_std = pd.Series()
    rmse_var = pd.Series()
    cent_mean = pd.Series()
    cent_std = pd.Series()
    cent_var = pd.Series()
    spec_bw_mean = pd.Series()
    spec_bw_std = pd.Series()
    spec_bw_var = pd.Series()
    contrast_mean = pd.Series()
    contrast_std = pd.Series()
    contrast_var = pd.Series()
    rolloff_mean = pd.Series()
    rolloff_std = pd.Series()
    rolloff_var = pd.Series()
    poly_mean = pd.Series()
    poly_std = pd.Series()
    poly_var = pd.Series()
    tonnetz_mean = pd.Series()
    tonnetz_std = pd.Series()
    tonnetz_var = pd.Series()
    zcr_mean = pd.Series()
    zcr_std = pd.Series()
    zcr_var = pd.Series()
    harm_mean = pd.Series()
    harm_std = pd.Series()
    harm_var = pd.Series()
    perc_mean = pd.Series()
    perc_std = pd.Series()
    perc_var = pd.Series()
    frame_mean = pd.Series()
    frame_std = pd.Series()
    frame_var = pd.Series()
    
    
    # Traversing over each file in path
    # file_data = [f for f in listdir(path) if isfile (join(path, f)) and ".sli" not in f and f[0]!="."]
    file_data = [f for f in listdir(path) if isfile (join(path, f)) and (".mp3" in f or ".ogg" in f) and ".sli" not in f]
    print("file_data " , file_data)
    for line in file_data:

        if ( line[-1:] == '\n' ):
            line = line[:-1]

        # Reading Song
        songname = path + line
        print("song_name:", songname)
        y, sr = librosa.load(songname, duration=60)
        S = np.abs(librosa.stft(y))
        
        # Extracting Features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
    
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
        
        # Transforming Features
        # songname_vector.set_value(id, line)  # song name
        songname_vector.at[id] = line
        tempo_vector.at[id] = tempo # tempo
        total_beats.at[id] = sum(beats)  # beats
        average_beats.at[id] = np.average(beats)
        chroma_stft_mean.at[id] = np.mean(chroma_stft)  # chroma stft
        chroma_stft_std.at[id] = np.std(chroma_stft)
        chroma_stft_var.at[id] = np.var(chroma_stft)
        chroma_cq_mean.at[id] = np.mean(chroma_cq)  # chroma cq
        chroma_cq_std.at[id] = np.std(chroma_cq)
        chroma_cq_var.at[id] = np.var(chroma_cq)
        chroma_cens_mean.at[id] = np.mean(chroma_cens)  # chroma cens
        chroma_cens_std.at[id] = np.std(chroma_cens)
        chroma_cens_var.at[id] = np.var(chroma_cens)
        mel_mean.at[id] = np.mean(melspectrogram)  # melspectrogram
        mel_std.at[id] = np.std(melspectrogram)
        mel_var.at[id] = np.var(melspectrogram)
        mfcc_mean.at[id] = np.mean(mfcc)  # mfcc
        mfcc_std.at[id] = np.std(mfcc)
        mfcc_var.at[id] = np.var(mfcc)
        mfcc_delta_mean.at[id] = np.mean(mfcc_delta)  # mfcc delta
        mfcc_delta_std.at[id] = np.std(mfcc_delta)
        mfcc_delta_var.at[id] = np.var(mfcc_delta)
        rmse_mean.at[id] = np.mean(rmse)  # rmse
        rmse_std.at[id] = np.std(rmse)
        rmse_var.at[id] = np.var(rmse)
        cent_mean.at[id] = np.mean(cent)  # cent
        cent_std.at[id] = np.std(cent)
        cent_var.at[id] = np.var(cent)
        spec_bw_mean.at[id] = np.mean(spec_bw)  # spectral bandwidth
        spec_bw_std.at[id] = np.std(spec_bw)
        spec_bw_var.at[id] = np.var(spec_bw)
        contrast_mean.at[id] = np.mean(contrast)  # contrast
        contrast_std.at[id] = np.std(contrast)
        contrast_var.at[id] = np.var(contrast)
        rolloff_mean.at[id] = np.mean(rolloff)  # rolloff
        rolloff_std.at[id] = np.std(rolloff)
        rolloff_var.at[id] = np.var(rolloff)
        poly_mean.at[id] = np.mean(poly_features)  # poly features
        poly_std.at[id] = np.std(poly_features)
        poly_var.at[id] = np.var(poly_features)
        tonnetz_mean.at[id] = np.mean(tonnetz)  # tonnetz
        tonnetz_std.at[id] = np.std(tonnetz)
        tonnetz_var.at[id] = np.var(tonnetz)
        zcr_mean.at[id] = np.mean(zcr)  # zero crossing rate
        zcr_std.at[id] = np.std(zcr)
        zcr_var.at[id] = np.var(zcr)
        harm_mean.at[id] = np.mean(harmonic)  # harmonic
        harm_std.at[id] = np.std(harmonic)
        harm_var.at[id] = np.var(harmonic)
        perc_mean.at[id] = np.mean(percussive)  # percussive
        perc_std.at[id] = np.std(percussive)
        perc_var.at[id] = np.var(percussive)
        frame_mean.at[id] = np.mean(frames_to_time)  # frames
        frame_std.at[id] = np.std(frames_to_time)
        frame_var.at[id] = np.var(frames_to_time)
        
        # print(songname)
        id = id+1
    
    # Concatenating Features into one csv and json format
    feature_set['song_name'] = songname_vector  # song name
    feature_set['tempo'] = tempo_vector  # tempo 
    feature_set['total_beats'] = total_beats  # beats
    feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set['chroma_stft_std'] = chroma_stft_std
    feature_set['chroma_stft_var'] = chroma_stft_var
    feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set['chroma_cq_std'] = chroma_cq_std
    feature_set['chroma_cq_var'] = chroma_cq_var
    feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set['chroma_cens_std'] = chroma_cens_std
    feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set['melspectrogram_std'] = mel_std
    feature_set['melspectrogram_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean  # mfcc
    feature_set['mfcc_std'] = mfcc_std
    feature_set['mfcc_var'] = mfcc_var
    feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    feature_set['mfcc_delta_std'] = mfcc_delta_std
    feature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rmse_mean'] = rmse_mean  # rmse
    feature_set['rmse_std'] = rmse_std
    feature_set['rmse_var'] = rmse_var
    feature_set['cent_mean'] = cent_mean  # cent
    feature_set['cent_std'] = cent_std
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set['spec_bw_std'] = spec_bw_std
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean  # contrast
    feature_set['contrast_std'] = contrast_std
    feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set['rolloff_std'] = rolloff_std
    feature_set['rolloff_var'] = rolloff_var
    feature_set['poly_mean'] = poly_mean  # poly features
    feature_set['poly_std'] = poly_std
    feature_set['poly_var'] = poly_var
    feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set['tonnetz_std'] = tonnetz_std
    feature_set['tonnetz_var'] = tonnetz_var
    feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set['zcr_std'] = zcr_std
    feature_set['zcr_var'] = zcr_var
    feature_set['harm_mean'] = harm_mean  # harmonic
    feature_set['harm_std'] = harm_std
    feature_set['harm_var'] = harm_var
    feature_set['perc_mean'] = perc_mean  # percussive
    feature_set['perc_std'] = perc_std
    feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean  # frames
    feature_set['frame_std'] = frame_std
    feature_set['frame_var'] = frame_var

    # Converting Dataframe into CSV Excel and JSON file
    game = path.split("rawscripts/")[-1].split("/")[0]
    print("game",game)
    feature_set.to_csv('../bgm_feature/bgmfeature_{}.csv'.format(game))
    feature_set.to_json('../bgm_feature/bgmfeature_{}.json'.format(game))
    # feature_set.to_csv('./test_bgm2/bgmfeature.csv')
    # feature_set.to_json('./test_bgm2/bgmfeature.json')

def normalizeTest():
    base_path = "/Users/kanoryu/Google ドライブ/codes/Serif/SerifDataCleaner/src_bgm/bgm_feature/bgmfeature_norm.csv"
    test_path = "./test_bgm1/bgmfeature.csv"

    df_base = pandas.read_csv(base_path)
    # print(df_base.columns)
    mean_ind = list(df_base.index)[-2]
    std_ind = list(df_base.index)[-1]
    # print(list(df_base['Unnamed: 0']))
    # for ri,row in df_base.iterrows():
    #     print(ri)
    mean_list = df_base.loc[mean_ind][2:]
    std_list = df_base.loc[std_ind][2:]
    # mean_list  = df_base[df_base['Unnamed: 0']=="14"][2:]
    # mean_list  = df_base[df_base['Unnamed: 0']=="mean"][2:]
    # std_list  = df_base[df_base['Unnamed: 0']=="std"][2:]
    # std_list  = df_base[df_base['Unnamed: 0']=="13"][2:]

    df_test = pandas.read_csv(test_path)
    col_list = list(df_base.columns)[2:]
    # print("col_list",col_list)
    # print(df_base[col_list].mean())
    # print("mean_list",mean_list)
    # print("std_list",std_list)
    df_std = (df_test[col_list] - mean_list) / std_list
    # songname_row = [title+"/"+songname for title,songname in zip(title_row,df_concat["song_name"])]

    df_std["songname"] = df_test["song_name"]
    # print("df_std",df_std)


    df_std = df_std[["songname"]+col_list]
    df_std.to_csv(test_path.replace(".csv","_norm.csv"))

def normalizeConcat():
    inre_softs = ['bokukimi', 'chusingura', 'chusingura_fd', 'miburo']
    parasol_softs = ["delivara","majicara",'haruno', 'kanoren', 'koiimo', 'qsplash', 'sakura', 'yumekoi']
    uguisu_softs = ["kaminoue","suisoginka"]
    alcot_softs = ["onigokko","onigokko_fd"]
    honeycomb_softs=["1_2summer","kicking_horse","fair_child","natsupochi","daitouryou","daitouryou_fd","realimouto"]
    akabee_softs = ["konboku","sono_yokogao","okibaganai","yayaokibaganai","lavender"]
    softs_list = inre_softs+parasol_softs+uguisu_softs+alcot_softs+honeycomb_softs+akabee_softs

    bgm_file_list=['../bgm_feature/bgmfeature_{}.csv'.format(title) for title in softs_list]
    df_list=[pandas.read_csv(bgm_file) for bgm_file in bgm_file_list]
    col_list_list = [df.columns for df in df_list]
    for col_list in col_list_list:
        print(len(col_list))
    title_row=[title for title,df in zip(softs_list,df_list) for _ in range(len(df))]
    df_concat=pandas.concat(df_list)
    df_concat.to_csv("../bgm_feature/bgmfeature_concat.csv")
    col_list = list(df_concat.columns)[2:]
    print('col_list',col_list)
    from scipy import stats
    df_concat[col_list]=(df_concat[col_list]-df_concat[col_list].mean())/df_concat[col_list].std()
    # df_concat[col_list]=df_concat[col_list].apply(stats.zscore, axis=0)
    songname_row = [title+"/"+songname for title,songname in zip(title_row,df_concat["song_name"])]
    # df_std["songname"] = songname_row
    df_concat["songname"] = songname_row
    # prt("df_std",df_std)


    # df_std = df_std[["songname"]+col_list]
    df_concat = df_concat[["songname"]+col_list]
    # df_std.to_csv(test_path.replace(".csv","_norm.csv"))
    df_concat.to_csv("../bgm_feature/bgmfeature_norm.csv")

# Extracting Feature Function Call
if __name__=="__main__":
    # extract_feature('Dataset/')

    path = '/Users/kanoryu/IdeaProjects/Corpus/NovelGame/rawscripts/'
    # inre_softs = ['bokukimi', 'chusingura', 'chusingura_fd']#, 'miburo']
    inre_softs = ['miburo']
    # parasol_softs = ['haruno', 'kanoren', 'koiimo', 'qsplash', 'sakura', 'yumekoi']
    # path_list = [path+inre+"/bgm/" for inre in inre_softs]
    # uguisu_softs = ["suisoginka","kaminoue"]
    # path_list= [path+ugs+"/bgm/" for ugs in uguisu_softs]

    # alcot_softs = ["onigokko","onigokko_fd","fair_child","natsupochi","daitouryou","realimouto"]
    alcot_softs = ["1_2summer"]#"daitouryou_fd","shunki_gentei","1_2summer"]
    akabee_softs = ["sono_yokogao"]#"lavender","konboku"]#""okibaganai", "yayaokibaganai"]
    path_list= [path+alcot+"/bgm/" for alcot in akabee_softs]
    # [extract_feature(p_e) for p_e in path_list]

    # honeycomb_softs = ["kicking_horse", "1_2summer"]
    # path_list= [path+alcot+"/data/bgm/" for alcot in honeycomb_softs]
    # [extract_feature(p_e) for p_e in path_list]

    # path_list = ["./test_bgm1/"]

    # parasol_softs = ["delivara", "majicara"]
    # path_list += [path + prsl + "/media/bgm/" for prsl in parasol_softs]
    # [extract_feature(p_e) for p_e in path_list]

    # normalizeTest()
    normalizeConcat()

# PairStoryMusic

テキスト（小説文）に合った音楽を選択するニューラルネットワークモデル。

bgm_feature_extraction.pyでtest_bgm以下のディレクトリに置かれた音楽ファイルから特徴量を抽出し、

sampleSiamLSTM.py -i {#入力テキストファイル} で入力テキストを分かち書き（MeCabのプリインストールが必要）し、10文単位でテキストの雰囲気に合った音楽の適合度を計算します。

計算結果は、test_bgm/test_result.csv に出力されます。

以下のライブラリが必要です。（バージョンは動作確認したバージョン）

* pandas==0.18.1
* chainer==4.2.0
* librosa==0.6.2

## Automatically Assigning Appropriate Music for Novels.

The implementation of the model explained in this article.(https://medium.com/@apictureofthefuture/automatically-assigning-music-suitable-for-a-novel-7ee477b7001c)

run bgm_feature_extraction.py and you can extract features from music files in the directory "text_bgm".

run sampleSiamLSTM.py -i {#input text_file} and segment texts into words(need to install MeCab), and the model computes compatibility of music and texts (10 sequential sentences).

the results will be written in test_bgm/test_result.csv

you need to install the library below to run the codes (I confirmed it works on the written version).

* pandas==0.18.1
* chainer==4.2.0
* librosa==0.6.2

# PairStoryMusic

テキスト（小説文）に合った音楽を選択するニューラルネットワークモデル。

bgm_feature_extraction.pyでtest_bgm以下のディレクトリに置かれた音楽ファイルから特徴量を抽出し、

sampleSiamLSTM.py -i {#入力テキストファイル} で入力テキストを分かち書きし、10文単位でテキストの雰囲気に合った音楽の適合度を計算します。

計算結果は、test_bgm/test_result.csv に出力されます。

以下のライブラリが必要です。（バージョンは動作確認したバージョン）

* pandas==0.18.1
* chainer==4.2.0
* librosa==0.6.2


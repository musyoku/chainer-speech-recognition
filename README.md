# [WIP] Character Level Speech Recognition

# Work in progress ...

## 注意

データがメモリに乗り切らないので必要になった時にファイルから読み出すようになっています。

そのため高速なSSDに音声ファイルを置き、Python 3の使用をおすすめします。

## Requirements

- Python 2 or 3
- Chainer 2+
- CuPy
- SciPy
- [python_speech_features](https://github.com/jameslyons/python_speech_features)

```
pip install python_speech_features
```

for data augmentation

- [python-acoustics](https://github.com/python-acoustics/python-acoustics) 

```
pip install acoustics
``` 

## TODO

- [x] CTC
- [x] 声道長歪み
- [x] 話速歪み
- [x] ランダム歪み
- [x] 大規模データの読み出し
- [x] 学習と読み出しの並列化
- [ ] 文字誤り率10%以下

## Tools

- [CSJ音声コーパスの前処理ツール](https://github.com/musyoku/csj-preprocesser)

## References

- [統計的手法を用いた音声モデリングの高度化とその音声認識への応用](https://www.gavo.t.u-tokyo.ac.jp/~mine/japanese/nlp+slp/IPSJ-MGN451004.pdf)
- [形態素解析も辞書も言語モデルもいらないend-to-end音声認識](https://www.slideshare.net/t_koshikawa/endtoend)
- [『日本語話し言葉コーパス』を用いた音声認識の進展](http://sap.ist.i.kyoto-u.ac.jp/lab/bib/report/KAW-orc04.pdf)
- [Deep Neural Networkに基づく日本語音声認識の基礎評価](https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_uri&item_id=94549&file_id=1&file_no=1)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
- [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)
- [Towards End-to-End Speech Recognition with Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.02720)
- [English Conversational Telephone Speech Recognition by Humans and Machines](https://arxiv.org/abs/1703.02136)
- [Japanese and Korean Voice Search](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [Gram-CTC: Automatic Unit Selection and Target Decomposition for Sequence Labelling](https://arxiv.org/abs/1703.00096)
- [Warp CTC](https://github.com/baidu-research/warp-ctc)
- [単語誤り最小化に基づく識別的リスコアリングによる音声認識](https://www.nhk.or.jp/strl/publica/rd/rd131/PDF/P28-39.pdf)
- [ベイズ推論を用いた連続音声からの言語モデル学習](http://www.phontron.com/paper/neubig10slp82.pdf)
- [End-to-End Speech Recognition Models](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1762&context=dissertations)
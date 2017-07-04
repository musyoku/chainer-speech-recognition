# Character Level Speech Recognition

in progress ...

データがメモリに乗り切らないので必要になった時にファイルから読み出すようになっています。

そのためSSDに音声ファイルを置くことをおすすめします。

## Requirements

- Python 2 or 3
- Chainer 2+
- SciPy
- [python_speech_features](https://github.com/jameslyons/python_speech_features)

for data augmentation

- [python-acoustics](https://github.com/python-acoustics/python-acoustics) 
- [Python-Wrapper-for-World-Vocoder](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) 

## TODO

- [ ] CTC
- [ ] 声道長歪み
- [ ] 話速歪み
- [ ] ランダム歪み
- [ ] 大規模データの読み出し

## Tools

- [CSJ音声コーパスの前処理ツール](https://github.com/musyoku/csj-preprocesser)

## References

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
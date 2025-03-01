# バックチャネル（相槌）生成システム

このリポジトリは音声入力に対して適切なタイミングで相槌を打つデモシステムです。機械学習モデルを使用して相槌のタイミングをリアルタイムで予測し、閾値を超えた場合に音声ファイルを再生します。

## 機能

- リアルタイム音声入力の処理
- HuBERTとLSTMモデルによる相槌タイミングの予測
- PyQt5によるリアルタイムグラフ表示
- 複数の音声ファイルからランダムに相槌を選択

## 必要環境

環境構築には以下の方法が利用できます：

### Conda環境を使用する場合

```bash
conda env create -f demo.yml
conda activate <環境名>
```

### macOSでpyenvを使用する場合

```bash
# pip のアップデート
python3 -m pip install --upgrade pip

# 必要なPythonモジュールのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install sounddevice
pip install numpy
pip install soundfile
pip install PyQt5 PyQtChart
```

## 使用方法

```bash
python demo_pyqt5.py
```

実行時に以下の選択を行います：

1. **モデルの選択**: 入力音声長に対応するモデルを選択
   - 0.5秒 (0.5s_CSJmodel.pth)
   - 1.0秒 (1s_CSJmodel.pth)
   - 2.0秒 (2s_CSJmodel.pth)

2. **音声ディレクトリの選択**: 相槌として使用する音声ファイルのディレクトリを選択
   - bc_wav/以下の各ディレクトリ（zundamon, satou, shrimin, tanakaなど）

3. **入力デバイスの選択**: 使用するマイクなどの音声入力デバイスを選択

## 注意事項

- 入力音声長は学習済みモデルの都合上、500ms、1000ms、2000msの3種類のみ対応しています
- 相槌の選択はタイミングのみを考慮しており、内容はランダムに選択されます
- 連続した相槌を防ぐため、一定時間（デフォルト1秒）の抑制期間が設けられています

## システム構成

システムは以下のコンポーネントで構成されています：

- **音声入力処理**: sounddeviceライブラリによるリアルタイム音声キャプチャ
- **特徴抽出**: HuBERTモデルによる音声特徴量の抽出
- **タイミング予測**: LSTMモデルによる相槌タイミングの予測
- **グラフ表示**: PyQt5とQChartによるリアルタイムグラフ表示
- **音声再生**: 予測値が閾値を超えた場合に相槌音声を再生

## ライセンス

[ライセンス情報をここに記載]

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

print("実行中… システムを起動しています。")

import os
import glob
import torch
import torch.nn as nn
import sounddevice as sd
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import time
import threading
import queue
import random
import soundfile as sf

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QLegend
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtGui import QPainter, QFont

# In[2]:

device = torch.device("cpu")

# In[3]:


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# In[4]:



# 入力デバイスを確認、選択する関数
def select_input_device():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    print("Available input devices:")
    for i, device in enumerate(input_devices):
        print(f"{i}: {device['name']}")
    
    selection = int(input("Select input device number: "))
    return input_devices[selection]['index']


# In[ ]:

# 実行の負荷が高い処理の前にメッセージを表示
print("モデルを読み込んでいます…")  
MODEL_ID = "rinna/japanese-hubert-base"

# HuBERTモデルとFeature Extractorの準備
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
hubert_model = HubertModel.from_pretrained(MODEL_ID).to(device)
print("モデルの読み込み完了。")

# 使用する中間層のインデックスを指定(0~11)
# LAYER_INDEX = 6  # ７番目


# In[6]:


def select_model_and_params():
    """
    ユーザーに選択肢を提示し、入力音声の長さに対応するモデルのパスとwindow_sizeを返す関数

    Returns:
        tuple: (model_path, window_size) - 選択されたモデルのパスと対応するwindow_size
    """
    # 利用可能なモデルとその対応する音声長（秒）
    available_models = [
        {"model": "0.5s_CSJmodel.pth", "window_size": 0.5},
        {"model": "1s_CSJmodel.pth", "window_size": 1.0},
        {"model": "2s_CSJmodel.pth", "window_size": 2.0}
    ]
    
    # 選択肢を表示
    print("\n選択可能なモデル一覧:")
    for i, model_info in enumerate(available_models, start=1):
        print(f"{i}: {model_info['window_size']}秒（{model_info['model']}）")

    # ユーザーに入力を求める
    while True:
        try:
            selection = int(input("\n使用するモデルの番号を選択してください（1〜3）: "))
            if 1 <= selection <= len(available_models):
                selected_model = available_models[selection - 1]
                model_path = f"model/{selected_model['model']}"
                window_size = selected_model['window_size']
                print(f"\n選択されたモデル: {model_path}")
                print(f"モデルへの入力音声長: {window_size}秒")
                return model_path, window_size
            else:
                print("1から3の番号を入力してください。")
        except ValueError:
            print("数値を入力してください。")

# In[ ]:


# ユーザーにモデルを選択してもらう処理
saved_model_pth, window_size = select_model_and_params()
print(f"使用するモデルのパス: {saved_model_pth}")
print(f"設定された入力音声長: {window_size}秒")

# 音声処理パラメータ
sample_rate = 16000
stride = 0.2  # 予測頻度(s)
buffer_size = int(window_size * sample_rate)
stride_size = int(stride * sample_rate)
bc_thre  = 0.80 # 予測のしきい値

# In[ ]:


# LSTMモデルの準備
input_dim = 768
hidden_dim = 128
output_dim = 1
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
lstm_model.load_state_dict(torch.load(saved_model_pth, map_location=device))
lstm_model.eval()


# In[9]:



def select_audio_files():
    """
    ユーザーに使用する音声ファイルのディレクトリを選択してもらい、
    選択したディレクトリ内のすべてのWAVファイルを返す関数
    
    Returns:
        list: 選択したディレクトリ内のすべての音声ファイルのパスのリスト
    """
    # 音声ファイルのディレクトリを選択
    base_dir = "bc_wav"
    # 利用可能なディレクトリを取得
    available_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not available_dirs:
        print(f"警告: {base_dir}ディレクトリ内にサブディレクトリが見つかりませんでした。")
        return []
    
    print("\n利用可能な音声ディレクトリ:")
    for i, dir_name in enumerate(available_dirs):
        print(f"{i}: {dir_name}")
    
    # ユーザーにディレクトリを選択してもらう
    while True:
        try:
            selection = int(input("\n音声ディレクトリを選択してください (番号): "))
            if 0 <= selection < len(available_dirs):
                selected_dir = available_dirs[selection]
                break
            else:
                print(f"0から{len(available_dirs)-1}までの番号を入力してください。")
        except ValueError:
            print("数値を入力してください。")
    
    # 選択したディレクトリ内のすべてのWAVファイルを取得
    selected_dir_path = os.path.join(base_dir, selected_dir)
    wav_files = glob.glob(f"{selected_dir_path}/*.wav")
    
    # 相対パスに変換
    bc_audio_files = [f"bc_wav/{selected_dir}/{os.path.basename(f)}" for f in wav_files]
    
    print(f"\n選択された音声ディレクトリ: {selected_dir}")
    print(f"読み込まれた音声ファイル数: {len(bc_audio_files)}")
    if bc_audio_files:
        print("音声ファイル:")
        for file in bc_audio_files:
            print(f"  - {file}")
    else:
        print(f"警告: 選択されたディレクトリ {selected_dir_path} にWAVファイルが見つかりませんでした。")
    
    return bc_audio_files

# In[ ]:


# 次に音声ファイル選択関数を使用
bc_audio_files = select_audio_files()

print("\n設定の概要:")
print(f"使用するモデルのパス: {saved_model_pth}")
print(f"設定されたwindow_size: {window_size}秒")
print(f"使用する音声ファイル: {len(bc_audio_files)}個")

# In[11]:


# バックチャネル音声ファイルの読み込み
bc_audios = []
bc_srs = []
for file in bc_audio_files:
    audio, sr = sf.read(file)
    bc_audios.append(audio)
    bc_srs.append(sr)

# In[12]:


# 相槌の再生 (関数名を英語に変更)
def play_backchanneling():
    global last_backchanneling_time, suppression_duration # グローバル変数を参照

    current_time = time.time() # 現在時刻を取得

    # 前回相槌再生時刻から抑制時間以上経過しているか確認
    if current_time - last_backchanneling_time >= suppression_duration:
        # ランダムに相槌音声を選択
        idx = random.randint(0, len(bc_audios) - 1)
        audio = bc_audios[idx]
        sr = bc_srs[idx]
        sd.play(audio, blocking=False, samplerate=sr)
        last_backchanneling_time = current_time # 最後に相槌を再生した時刻を更新
        print("相槌再生") # デバッグ用メッセージ (動作確認用)
    else:
        print("相槌抑制中") # デバッグ用メッセージ (動作確認用)

# #### カーネルは必ず再起動してから実行！
# #### 連続相槌の抑制時間は0.5-1秒が目安

# In[13]:


# グローバル変数
time_data = []
confidence_data = []
last_prediction_time = 0
audio_buffer = np.zeros(buffer_size)
start_time = None
display_duration = 20 # グラフの表示時間（秒）

last_backchanneling_time = 0  # 最後に相槌を再生した時刻を記録する変数
suppression_duration = 1  # 相槌音声再生後の抑制時間（秒）【ユーザー指定: 0.5秒】

# In[14]:


# グラフのリセット
def reset_graph():
    global time_data, confidence_data, start_time
    time_data = []
    confidence_data = []
    start_time = time.time()

# In[15]:


# グラフの更新
def update_graph():
    global series, time_data, confidence_data, axis_x

    current_time = time.time() - start_time

    # 表示範囲内のデータのみ表示
    time_data_filtered = [t for t in time_data if t <= current_time]
    confidence_data_filtered = confidence_data[:len(time_data_filtered)]

    series.clear()
    for t, c in zip(time_data_filtered, confidence_data_filtered):
        series.append(t, c)

    # x軸の範囲を更新
    axis_x.setRange(0, display_duration)

    # グラフが限界に達したらリセット
    if current_time >= display_duration:
        reset_graph()

# In[16]:


# 音声入力と予測の処理
def process_audio(indata, frames, time_info, status):
    global audio_buffer, last_prediction_time, time_data, confidence_data, start_time

    if start_time is None:  # start_time が None の場合のみ初期化
        start_time = time.time() - stride # 最初の推論結果がグラフの開始になるように調整

    audio_buffer = np.roll(audio_buffer, -len(indata))
    audio_buffer[-len(indata):] = indata[:, 0]

    current_time = time.time()

    if current_time - last_prediction_time >= stride and np.any(audio_buffer):  # より緩和されたチェック
        inputs = feature_extractor(audio_buffer, sampling_rate=sample_rate, return_tensors="pt").input_values
        inputs = inputs.to(device)

        with torch.no_grad():
            wav2vec_output = hubert_model(inputs).last_hidden_state

        lstm_input = wav2vec_output
        prediction = lstm_model(lstm_input)

        result = torch.sigmoid(prediction).item()
        confidence = result

        if result > bc_thre:
            threading.Thread(target=play_backchanneling).start()

        last_prediction_time = current_time

        time_data.append(current_time - start_time)
        confidence_data.append(confidence)

# In[17]:


# グラフの初期化 (修正後)
def init_graph():
    global chart, series, threshold_series, layout, axis_x, axis_y

    chart = QChart()
    series = QLineSeries()
    threshold_series = QLineSeries()
    chart.addSeries(series)
    chart.addSeries(threshold_series)

    # フォント設定
    font = QFont() # QFontオブジェクトを作成
    font.setPointSize(30) # フォントサイズを12ポイントに設定 (調整可能)

    # タイトルフォント設定
    chart_title_font = QFont() # タイトル用に別のQFontオブジェクトを作成しても良い
    chart_title_font.setPointSize(40) # タイトルは少し大きめに設定 (調整可能)
    chart_title_font.setBold(True) # タイトルを太字に (必要に応じて)
    chart.setTitle("相槌生成タイミング予測システム")
    chart.setTitleFont(chart_title_font) # タイトルにフォントを適用


    axis_x = QValueAxis()
    axis_x.setTitleText("経過時間")
    axis_x.setRange(0, display_duration)  # 初期表示範囲を設定

    # X軸タイトルフォント設定
    axis_x_font = QFont()
    axis_x_font.setPointSize(30) # 軸タイトルフォントサイズ設定 (調整可能)
    axis_x.setTitleFont(axis_x_font) # X軸タイトルにフォントを適用
    axis_x.setTickType(QValueAxis.TicksDynamic)
    axis_x.setTickInterval(1.0)  # strideに合わせて格子線を表示
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)
    threshold_series.attachAxis(axis_x)


    axis_y = QValueAxis()
    axis_y.setTitleText("期待値")
    axis_y.setRange(0, 1)
    axis_y.setTickCount(11)

    # Y軸タイトルフォント設定
    axis_y_font = QFont()
    axis_y_font.setPointSize(30) # 軸タイトルフォントサイズ設定 (調整可能)
    axis_y.setTitleFont(axis_y_font) # Y軸タイトルにフォントを適用
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)
    threshold_series.attachAxis(axis_y)


    # 閾値の線を初期化時に一度だけ描画
    threshold_series.append(0, bc_thre)
    threshold_series.append(display_duration, bc_thre)

    chart_view = QChartView(chart)
    chart_view.setRenderHint(QPainter.Antialiasing)

    layout.addWidget(chart_view)

    # 凡例の追加
    legend = chart.legend()
    legend.setVisible(True)
    legend.setAlignment(Qt.AlignTop)

    # 凡例フォント設定
    legend_font = QFont()
    legend_font.setPointSize(30) # 凡例フォントサイズ設定 (調整可能)
    legend.setFont(legend_font) # 凡例にフォントを適用

    series.setName("期待値")
    threshold_series.setName("閾値")

    reset_graph()  # グラフの初期化時にリセット

# In[ ]:


# メイン処理
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    window.resize(1200, 900)
    layout = QVBoxLayout()
    window.setLayout(layout)

    init_graph()

    print("音声デバイスを検索中…")
    device_index = select_input_device()
    start_time = time.time()

    q = queue.Queue()
    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=sample_rate,
            callback=process_audio,
            blocksize=stride_size,  # ブロックサイズを明示的に設定
            latency='low'  # レイテンシーを低く設定
            ):
            
            print("Recording... Press Ctrl+C to stop.")
            
            timer = QTimer()
            timer.timeout.connect(update_graph)
            timer.start(100)

            window.show()
            app.exec_()
            q.get()
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        QCoreApplication.quit()

import subprocess
import os
from moviepy import AudioFileClip
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import math
import pandas as pd

from threading import Lock

lock = Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基于sherpa-onnx的event分类模型参数
executable_path = "./sherpa-onnx/bin/sherpa-onnx-offline-audio-tagging" # sherpa-onnx可执行文件
model_path = "./sherpa-onnx/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx" # 模型架构和参数文件
labels_path = "./sherpa-onnx/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv" # sherpa-onnx分类标签文档

# 验证集地址
dataset_dir = './dataset/validate/audio'
label_path = './dataset/validate/meta.txt'
# 训练统计后的表格地址
all_event_counts_path = './all_event_counts.csv'

scenes = ['working', 'commuting', 'entertainment', 'home']
# 数据集的label到四个场景分类的映射
label_scene_map = {
    'bus':'commuting',
    'cafe/restaurant':'entertainment',
    'car':'commuting',
    # 'city_center':,
    # 'forest_path':,
    # 'grocery_store':,
    'home':'home',
    # 'library':,
    'metro_station':'commuting',
    'office':'working',
    # 'residential_area':,
    'train':'commuting',
    'tram':'commuting',
    # Lakeside beach (outdoor)
    # Urban park (outdoor)
}
match_threshold = 0.9
# 分窗大小
window_duration = 5

# 调用sherpa-onnx判断音频对应的event
def find_match_event(audio_part_path):
    command = [
        executable_path,
        f"--zipformer-model={model_path}",
        f"--labels={labels_path}",
        audio_part_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        events = []
        for line in result.stderr.strip().split('\n'):
            if 'AudioEvent(name="' in line:
                parts = line.split('AudioEvent(name="')
                if len(parts) > 1:
                    name_part = parts[1].split('"', 1)[0]
                    prob_part = parts[1].split('prob=', 1)[1].split(')')[0]
                    events.append((name_part, float(prob_part)))
        events = sorted(events, key=lambda x: x[1], reverse=True)

        match_events = []
        for event in events:
            if(event[1]>=match_threshold):
                match_events.append(event)
        if(len(match_events)<2):
            match_events = events[:2]

        return match_events

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
        print(f"Error output: {e.stderr}")
        return []

# 存储音频切片，并判断event类型
def process_audio_segment(audio_segment):
    try:
        temp_audio_file = f"temp_audio_segment_{uuid.uuid4()}.wav"
        with lock:
            audio_segment.write_audiofile(temp_audio_file, codec='pcm_s16le')
        match_events = find_match_event(temp_audio_file)
        return match_events
    except Exception as e:
        logger.error(f"Error writing audio segment: {e}")
        return []
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file) 

# 对音频分窗处理，统计该音频中各event数量
def process_audio(audio_path, event_counts, window_duration=5):
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    try:
        all_match_events = []
        audio_segments = []
        for start_time in range(0, int(audio_duration), window_duration):
            end_time = min(start_time + window_duration, audio_duration)
            audio_segments.append(audio.subclipped(start_time, end_time))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_audio_segment, audio_segment)for audio_segment in audio_segments]
            for future in as_completed(futures):
                all_match_events.extend(future.result())

        for event in all_match_events:
            if event[0] in event_counts:
                event_counts[event[0]] += 1
            else:
                event_counts[event[0]] = 1
        return math.ceil(int(audio_duration)/window_duration)
    except Exception as e:
        logger.error(f"Error processing video {audio_path}: {e}")

# 预测单个音频所处的环境
def predict(audio_path):
    event_counts = {}
    num = process_audio(audio_path, event_counts, window_duration)
    df = pd.read_csv(all_event_counts_path)

    print(audio_path)
    for event_name, count in event_counts.items():
        print(f"Event: {event_name}, Count: {count}")

    min = 0
    min_scene = ""
    for scene in scenes:
        distance = 0
        for event in df['EventTypes'].tolist():
            if(event in event_counts):
                count = event_counts[event]
            else:
                count = 0
            distance += (df.loc[df['EventTypes'] == event, scene].values[0] - count / num * 18)**2
        if(min == 0 or distance < min ):
            min = distance
            min_scene = scene
    return min_scene

# 读入训练集的label字典
def load_labels_to_dict(meta_file_path):
    labels_dict = {}
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    labels_dict[filename] = label
        return labels_dict
    except FileNotFoundError:
        print(f"Error: The file {meta_file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# 验证 
def validate():
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    sum_num = 0
    right_num = 0
    labels_dict = load_labels_to_dict(label_path)
    for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.wav'):
                    if labels_dict[f"audio/{file}"] in label_scene_map:
                        sum_num += 1
                        if(predict(os.path.join(root, file)) == label_scene_map[labels_dict[f"audio/{file}"]]):
                            right_num += 1
                    else:
                        print(f"label-{labels_dict[f"audio/{file}"]} not exist!")
                # if sum_num == 50:
                #     break

    print(f"\ntotal num:{sum_num}   right num:{right_num}")
    print(f"right rate:{right_num/sum_num}")


if __name__ == "__main__":
    start_time = time.time()

    validate()

    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

# README

### 基本说明

​	本项目旨在根据音频的环境音，判断当前音频所处的环境，属于一个四分类任务，输出结果包括working(工作)，commuting(通勤)，entertainment(娱乐)，home(家庭)四种类型。实现的原理是将音频进行分窗，在滑动窗口的过程中判断每个窗口的event，该过程是通过预训练好的模型sherpa-onnx实现。在训练过程中，我们会统计训练集的所有音频，在每种环境下各个event出现频率的情况，以表格形式存储。之后，我们同样使用sherpa-onnx分析待预测音频各个窗口的event类型及其出现的次数，基于先前的表格，将其聚类到其中某一类以判断所处环境。



### 文档结构

- `dataset`：数据集
  - `train`
    - `audio`：音频集合
    - `meta.txt`：音频label ( 格式：`"audio/name.wav label"` )
  - `validate`
  - `predict`
- `sherpa-onnx`：event分类模型
  - `bin`：sherpa-onnx可执行文件
  - `sherpa-onnx-zipformer-small-audio-tagging-2024-04-15`：预训练的sherpa-onnx模型参数
- `record`：备份训练过的记录，使用时将其替代根目录下的`all_event_counts.csv`
  - `all_event_counts_v1.csv`
  - . . . . . .
- `train.py`
- `validate.py`
- `predict.py`
- `requirements.txt`
- (*)`all_event_counts.csv`：经过`train.py`统计过后训练集中每种event的出现频率 ( 分母为18，即对于18个窗口大小的音频中，每种event出现的次数 )，`predict.py`会根据此统计表，将当前音频聚类到某一种场景下



### 使用说明

#### 环境配置

```
pip install -r requirements.txt
```



#### train.py

##### 参数说明 ( `*`表示可根据使用情况进行修改的参数 )：

- **executable_path**：sherpa-onnx可执行文件
- **model_path**：模型架构和参数文件
- **labels_path**：sherpa-onnx分类标签文档
- (*) **dataset_dir**：训练集文档地址
- (*) **label_path**：训练集的label地址
- (*) **window_duration**：窗口大小，单位为s
- (*) **label_scene_map**：Dictionary类型，由于收集的数据集使用的label不同于我们需要的分类，所以使用label_scene_map将其映射到我们所设的四种场景
- (*) **match_threshold**：使用sherpa-onnx判断event类型时，如果概率大于match_threshold时，都会被认为是符合当前音频窗口的event类型，否则取概率最大的两个

##### api说明：

```python
# 调用sherpa-onnx判断音频对应的event
def find_match_event(audio_part_path)
```

- input：audio_part_path——音频切片地址
- output：match_events——List类型，表示当前音频的event类型

```python
# 存储音频切片，并判断event类型
def process_audio_segment(audio_segment)
```

- input：audio_segment——音频切片
- output：match_events——List类型，表示当前音频的event类型

```python
# 对音频分窗处理，统计该音频中各event数量
def process_audio(audio_path, event_counts, window_duration=5)
```

- input：audio_path——音频地址，event_counts——空的Dictionary类型，用于统计每个event出现的次数，window_duration——窗口大小
- output：当前音频的分窗的数量

```python
# 读入训练集的label字典
def load_labels_to_dict(meta_file_path):
```

- input：meta_file_path——训练集的标签地址
- output：labels_dict——Dictionary类型，key为训练集中的音频名称，value为对应的训练集下的label

```python
# 统计汇总生成表格
def generate_table(file_path):
```

- input：file_path——sherpa-onnx分类标签文档，用于统计所有event种类

##### 运行命令：

```
python train.py
```

( 如果后续要将window_duration作为参数放到运行命令中，代码可能还要改一下 )

运行结果是统计训练集中所有音频的event出现情况，绘制成表格存储在根目录下，命名为`all_event_counts.csv`



#### validate.py

##### 参数说明 ( `*`表示可根据使用情况进行修改的参数 )：

基本情况与train.py中一样

- **(*) all_event_counts_path**：train.py生成的表格路径，可以更换成record中的其他记录

##### api说明：

```python
# 预测单个音频所处的环境
def predict(audio_path):
```

- input：audio_path——待预测的音频地址
- output：scene——预测最可能的场景类型

##### 运行命令：

```
python validate.py
```

运行结果是预测验证集下的音频所处场景，输出预测的总样本数，正确的数量以及正确率



#### predict.py

运行命令：

```
python predict.py
```

运行结果是对测试集下的所有音频进行分析，输出每个音频的名称及其对应的预测场景
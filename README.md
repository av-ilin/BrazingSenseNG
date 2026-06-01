# BrazingSense

**BrazingSense** — исследовательский прототип системы машинного зрения для мониторинга процесса индукционной пайки волноводных трактов космических аппаратов.

Проект выполняется в рамках выпускной квалификационной работы на тему:

> **«Машинное зрение для управления процессом индукционной пайки волноводных трактов космических аппаратов»**

Индукционная пайка является технологическим процессом, в котором важно не только нагреть зону соединения, но и вовремя определить наступление ключевых стадий: активацию флюса, начало плавления припоя, активное протекание припоя и последующую стабилизацию соединения. При визуальном контроле оператору сложно устойчиво различать эти моменты, поскольку изменения могут быть локальными, кратковременными и зависящими от бликов, засветов, ракурса и состояния поверхности.

Главная задача проекта — по видеопотоку зоны пайки определять текущую стадию технологического процесса и формировать быстрый сигнал о начале активной пайки, который потенциально может использоваться оператором или управляющей системой для фиксации температуры.

В проекте процесс пайки представлен четырьмя стадиями:

```text
0 — inactive_preparation
1 — flux_activation
2 — active_brazing
3 — stabilization
```

Такое разбиение выбрано потому, что оно отражает основные визуально и технологически различимые состояния процесса.

`inactive_preparation` — подготовительная стадия, когда зона пайки ещё не демонстрирует выраженных признаков активного расплава. На этом этапе возможен прогрев, изменение освещения и постепенное изменение внешнего вида поверхности, но активная пайка ещё не началась.

`flux_activation` — стадия активации флюса. Визуально она может проявляться изменением прозрачности, блеска, локальной текстуры и светлых областей в зоне пайки. Эта стадия важна для оператора, поскольку показывает приближение к активной фазе процесса, но сама по себе ещё не означает, что припой начал протекать.

`active_brazing` — технологически наиболее важная стадия. Она соответствует началу плавления и протекания припоя, изменению формы припоя, появлению признаков расплава и заполнения шва. Именно эта стадия наиболее важна для формирования управляющего или рекомендательного сигнала, связанного с фиксацией температуры.

`stabilization` — стадия после активной пайки, когда процесс переходит к стабилизации/выдержке. Визуально изменения становятся менее резкими, а система должна фиксировать, что активная фаза уже была пройдена.

Такое разбиение позволяет решать две связанные задачи:

```text
1. Операторский мониторинг:
   определить текущую стадию процесса.

2. Управляющий триггер:
   определить момент начала active_brazing.
```

---

## 0. Резюме

К системе предъявлялись следующие требования.

### 1. Работа по видеопотоку

Система должна принимать кадры из видео или камеры, выделять область интереса зоны пайки и выполнять анализ без ручного вмешательства на каждом кадре.

### 2. Определение стадии пайки

Для каждого кадра система должна формировать предсказание одной из четырёх стадий:

```text
inactive_preparation
flux_activation
active_brazing
stabilization
```

При этом для операторского интерфейса важна не только сырая покадровая классификация, но и устойчивая стадия процесса без хаотичных скачков.

### 3. Устойчивая технологическая последовательность

Реальный процесс пайки развивается последовательно:

```text
inactive_preparation → flux_activation → active_brazing → stabilization
```

Поэтому поверх нейросетевой модели была добавлена state machine. Она запрещает невозможные обратные переходы и подтверждает переход к следующей стадии только после нескольких устойчивых предсказаний. Это делает вывод системы более пригодным для операторского мониторинга.

### 4. Быстрый сигнал начала активной пайки

Отдельно исследуется событие:

```text
active_brazing_started
```

Оно считается наступившим, когда модель устойчиво фиксирует признаки стадии `active_brazing`. В демонстрационной системе это событие отображается как:

```text
Neural Trigger: ON
HOLD TEMPERATURE: ON
```

Этот сигнал следует понимать как рекомендательный сигнал прототипа о необходимости фиксации температуры или внимания оператора, а не как готовую промышленную команду управления нагревателем.

### 5. Ограничение по задержке

Целевое ограничение по времени обработки кадра:

```text
не более 50 мс на кадр
```

В проекте были проверены разные варианты моделей:

```text
ResNet18 224×224 — качественная модель для GPU/CUDA.
ResNet18 64×64 — CPU-friendly вариант ResNet.
MobileNetV3 Small — лёгкая CPU-friendly модель.
```

### 6. Возможность работы без GPU

Поскольку целевой компьютер оператора может не иметь мощной GPU, отдельно исследовались CPU-friendly модели. Для этого были проверены уменьшенные входные размеры ResNet18 и облегчённая MobileNetV3 Small. Это позволило оценить компромисс между качеством классификации и временем инференса.

### 7. Резервный OpenCV-триггер

Дополнительно был исследован CV-trigger на классических признаках изображения. Он не заменяет нейросетевую модель, но может использоваться как резервный или диагностический сигнал. Такой подход полезен, если на целевом железе нейросетевой инференс окажется нежелательным или избыточным.

## Итоговая логика системы

Финальная архитектура прототипа состоит из нескольких уровней:

```text
Video frame
  ↓
ROI crop
  ↓
Neural stage classifier
  ↓
Raw Stage + probabilities
  ↓
State Machine
  ↓
Stable Stage
  ↓
Active Brazing Trigger
  ↓
HOLD TEMPERATURE signal
```

Дополнительно может быть включён резервный CV-trigger:

```text
ROI frame sequence
  ↓
OpenCV features
  ↓
CV Score
  ↓
CV Trigger
```

Таким образом, система решает не только задачу классификации кадров, но и задачу формирования устойчивого технологического состояния процесса.

На demo-video отображаются следующие значения:

```text
Raw Stage: мгновенное предсказание нейросетевой модели
Stable Stage: устойчивая стадия после state machine
Confidence: уверенность модели в raw-предсказании
P(active_brazing): вероятность стадии active_brazing
Neural Trigger: факт срабатывания нейросетевого триггера active_brazing_started
CV Score: значение резервного OpenCV-сигнала
CV Trigger: факт срабатывания резервного CV-trigger
HOLD TEMPERATURE: рекомендательный сигнал фиксации температуры
Inference: время обработки кадра
Time: время кадра в видео
```

`HOLD TEMPERATURE: ON` означает, что прототип считает событие начала активной пайки подтверждённым. В текущей версии этот сигнал формируется на основе нейросетевого триггера `P(active_brazing)`.

---

## 1. Что делает система

Финальная система решает две связанные задачи.

### 1.1. Операторский мониторинг стадии пайки

По каждому кадру видео система определяет одну из четырёх стадий:

```text
0 — inactive_preparation
1 — flux_activation
2 — active_brazing
3 — stabilization
```

Сырые предсказания нейросети дополнительно проходят через state machine, чтобы оператор видел не скачущую покадровую классификацию, а устойчивую технологическую последовательность:

```text
inactive_preparation → flux_activation → active_brazing → stabilization
```

### 1.2. Сигнал начала активной пайки

Отдельно формируется событие:

```text
active_brazing_started
```

В демонстрационной системе оно отображается как:

```text
Neural Trigger: ON/OFF
HOLD TEMPERATURE: ON/OFF
```

Смысл:

- `Neural Trigger: ON` — нейросетевой триггер подтвердил начало активной пайки;
- `HOLD TEMPERATURE: ON` — прототип сформировал рекомендательный сигнал фиксации температуры;
- `CV Trigger: ON/OFF` — опциональный резервный OpenCV-триггер, основанный на классических признаках изображения.

---

## 2. Финальная архитектура

Финальная архитектура проекта:

```text
Video frame
  ↓
ROI crop
  ↓
Neural stage classifier
  ↓
Raw Stage + probabilities
  ↓
State Machine v1
  ↓
Stable Stage
  ↓
Active brazing trigger from P(active_brazing)
  ↓
Hold Signal
```

Опционально:

```text
ROI frame
  ↓
OpenCV features
  ↓
CV trigger score
  ↓
CV Trigger
```

Итоговые выводимые значения на demo-video:

```text
Raw Stage: ...
Stable Stage: ...
Confidence: ...
P(active_brazing): ...
Neural Trigger: ON/OFF
CV Score: ...
CV Trigger: ON/OFF
HOLD TEMPERATURE: ON/OFF
Inference: ... ms
Time: ... s
```

---

## 3. Структура репозитория

Актуальная структура верхнего уровня:

```text
BrazingSense/
├── bash/
├── configs/
├── data/
│   ├── annotations/
│   ├── processed/
│   └── raw/
├── docs/
├── models/
│   └── checkpoints/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   └── brazing_sense/
└── requirements.txt
```

### Основные директории

| Папка                 | Назначение                                                      |
| --------------------- | --------------------------------------------------------------- |
| `data/raw/`           | исходные `.MOV`-видео процесса пайки                            |
| `data/processed/`     | извлечённые кадры по FPS                                        |
| `data/annotations/`   | интервальная разметка, frame-level labels, train/val/test split |
| `configs/`            | ROI, стадии, параметры state machine                            |
| `notebooks/`          | исследовательские ноутбуки 00–10                                |
| `scripts/`            | CLI-скрипты для подготовки данных, обучения/инференса/benchmark |
| `src/brazing_sense/`  | код переиспользуемых модулей                                    |
| `models/checkpoints/` | сохранённые PyTorch checkpoint моделей                          |
| `reports/`            | метрики, CSV/JSON-отчёты, demo-video, графики                   |

---

## 4. Данные и разметка

### Исходные видео

Исходные видео лежат в:

```text
data/raw/
```

В набор входят видео:

```text
MVI_6265.MOV
MVI_6266.MOV
MVI_6268.MOV
...
MVI_6283.MOV
```

### Интервальная разметка стадий

Основной файл интервальной разметки:

```text
data/annotations/stage_intervals.csv
```

Формат:

```csv
video_id,video_path,stage_name,start_s,end_s
MVI_6266,data/raw/MVI_6266.MOV,inactive_preparation,0.0,27.8
MVI_6266,data/raw/MVI_6266.MOV,flux_activation,27.8,30.8
MVI_6266,data/raw/MVI_6266.MOV,active_brazing,30.8,40.8
MVI_6266,data/raw/MVI_6266.MOV,stabilization,40.8,56.72
```

### Frame-level датасеты

В проекте были собраны датасеты на 3 FPS и 10 FPS:

```text
data/annotations/frame_labels_3.csv
data/annotations/frame_labels_10.csv
```

Статистика:

```text
data/annotations/frame_dataset_stats_3.json
data/annotations/frame_dataset_stats_10.json
```

### Splits

Разбиение выполняется по `video_id`, а не по отдельным кадрам, чтобы избежать утечки почти одинаковых соседних кадров между train/val/test.

```text
data/annotations/splits_3/
data/annotations/splits_10/
```

Внутри:

```text
train.csv
val.csv
test.csv
split_stats.json
```

### Blacklist / hard cases

Файл:

```text
data/annotations/blacklist.txt
```

Используется для исключения видео с выраженными аномалиями или нестабильной видимостью. В процессе исследования видео `MVI_6265` рассматривалось как hard-case из-за засветов и неоднозначного изменения флюса.

---

## 5. Конфиги

### `configs/stages.yaml`

Описание стадий процесса.

### `configs/roi.yaml`

Финальный ROI зоны пайки:

```yaml
x: 470
y: 280
w: 430
h: 290
```

ROI включает волновод, область флюса, нижний шов и зону возможного протекания припоя.

### `configs/state_machine.yaml`

Параметры финальной state machine:

```text
type: irreversible_majority_transition
min_confirm_frames: 3
window_size: 7
allow_backward_transitions: false
allow_stage_skipping: false
```

---

## 6. Ноутбуки

### `00_dataset.ipynb`

Первичная работа с видео и датасетом.

### `01_dataset_review.ipynb`

Проверка frame-level датасета:

- количество кадров;
- распределение стадий;
- проверка train/val/test split;
- визуальная проверка кадров.

### `02_roi_selection.ipynb`

Подбор ROI зоны пайки.

Результат:

```text
configs/roi.yaml
```

Финальный ROI:

```text
x=470, y=280, w=430, h=290
```

### `03_cv_feature_analysis.ipynb`

OpenCV baseline:

- извлечение признаков ROI;
- анализ признаков по стадиям;
- LogisticRegression / RandomForest / GradientBoosting;
- temporal smoothing;
- error pairs;
- вывод о пригодности OpenCV baseline.

Итог OpenCV baseline после smoothing:

```text
accuracy:    0.6296
macro_f1:    0.5809
weighted_f1: 0.6964
```

### `04_neural_baseline.ipynb`

Первичный neural baseline на 3 FPS:

- PyTorch Dataset;
- ResNet18;
- MobileNetV3 Small;
- class weights;
- confusion matrix;
- timeline plots;
- quick latency check.

Ключевой результат:

```text
ResNet18 3 FPS:
accuracy:    0.9347
macro_f1:    0.8489
weighted_f1: 0.9307
```

### `05_final_neural_stage_classification.ipynb`

Финальный stage-classifier на 10 FPS:

- датасет 10 FPS;
- ResNet18 base;
- ResNet18 с усилением `active_brazing`;
- MobileNetV3 Small;
- выбор финальной модели;
- оценка `active_brazing`.

Ключевой вывод:

- 10 FPS улучшил представление коротких стадий;
- `active_brazing` стал определяться лучше;
- основной качественный классификатор — ResNet18.

### `06_stage_state_machine.ipynb`

Исследование state machine:

- raw predictions;
- majority smoothing;
- irreversible state machine v1;
- state machine v2/v3/v4;
- анализ без hard-case `MVI_6265`;
- итоговый выбор state machine v1.

Лучший вывод без hard-case `MVI_6265`:

```text
State machine v1:
accuracy:    0.9581
macro_f1:    0.9155
weighted_f1: 0.9562
```

### `07_active_brazing_trigger.ipynb`

Триггер начала активной пайки на основе вероятностей stage-модели и CV baseline.

Лучший нейросетевой trigger:

```text
score:          P(active_brazing)
threshold:      0.2
confirm_frames: 7
detected:       3/3
MAE:            0.70 s
within_1.0s:    100%
within_2.0s:    100%
```

### `08_cv_active_brazing_trigger.ipynb`

Финальное исследование OpenCV fallback trigger.

Лучший CV-rule:

```text
score_col:      score_cv_v1_motion_texture
threshold:      0.5
confirm_frames: 5
detected:       3/3
MAE:            0.67 s
within_2.0s:    100%
```

Вывод:

- CV-trigger пригоден как explainable fallback;
- основной trigger всё равно лучше строить на `P(active_brazing)`.

### `09_resnet_input_size_cpu.ipynb`

Исследование ResNet18 с меньшим входным разрешением для CPU:

- ResNet18 128×128;
- ResNet18 96×96;
- ResNet18 64×64;
- fine-tune;
- quality vs latency.

Ключевой CPU-friendly результат:

```text
ResNet18 64×64 fine-tuned:
accuracy:    0.9174
macro_f1:    0.8328
weighted_f1: 0.9128
CPU p95:     ~39.8 ms
```

### `10_mobilenet_evolutionary_search.ipynb`

Упрощённый эволюционный подбор гиперпараметров MobileNetV3 Small:

- image size;
- learning rate;
- weight decay;
- active_brazing boost;
- augmentation;
- fitness-функция;
- top candidates;
- CPU-friendly demo.

Цель:

```text
найти MobileNetV3 Small конфигурацию, которая работает на CPU < 50 мс/кадр
и сохраняет приемлемое качество определения стадий.
```

Финальный CPU-demo был выполнен на `mobilenet_evo_best_finetuned_best`.

---

## 7. Скрипты

### `scripts/video_info.py`

Печатает информацию о видео:

```bash
python3 scripts/video_info.py data/raw/MVI_6266.MOV
```

### `scripts/create_empty_stage_annotations.py`

Создаёт шаблон CSV для интервальной разметки стадий.

### `scripts/make_time_grid.py`

Создаёт временную сетку для одного видео.

### `scripts/make_time_grid_all.py`

Создаёт временную сетку для всех видео.

### `scripts/build_frame_dataset.py`

Извлекает кадры из видео по интервальной разметке и формирует frame-level labels.

Пример для 3 FPS:

```bash
python3 scripts/build_frame_dataset.py \
  --intervals data/annotations/stage_intervals.csv \
  --output-frames data/processed/frames_3 \
  --output-labels data/annotations/frame_labels_3.csv \
  --save-stats data/annotations/frame_dataset_stats_3.json \
  --fps 3 \
  --overwrite
```

Пример для 10 FPS:

```bash
python3 scripts/build_frame_dataset.py \
  --intervals data/annotations/stage_intervals.csv \
  --output-frames data/processed/frames_10fps \
  --output-labels data/annotations/frame_labels_10.csv \
  --save-stats data/annotations/frame_dataset_stats_10.json \
  --fps 10 \
  --overwrite
```

### `scripts/split_frame_dataset.py`

Создаёт train/val/test split по `video_id`, с опциональным blacklist.

Пример:

```bash
python3 scripts/split_frame_dataset.py \
  --labels data/annotations/frame_labels_10.csv \
  --output-dir data/annotations/splits_10 \
  --blacklist data/annotations/blacklist.txt \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --overwrite
```

### `scripts/analyze_cv_features.py`

Извлекает OpenCV-признаки из кадров.

Пример для ROI 10 FPS:

```bash
python3 scripts/analyze_cv_features.py \
  --labels data/annotations/frame_labels_10.csv \
  --output-csv reports/cv_features_10/frame_features_roi.csv \
  --summary-csv reports/cv_features_10/summary_by_stage_roi.csv \
  --figures-dir reports/figures/cv_features_roi_10 \
  --roi 470,280,430,290 \
  --overwrite
```

### `scripts/run_neural_inference_video.py`

Финальный inference/demo-скрипт.

Он формирует demo-video с overlay:

```text
Raw Stage
Stable Stage
Confidence
P(active_brazing)
Neural Trigger
CV Score
CV Trigger
HOLD TEMPERATURE
Inference
Time
ROI rectangle
```

#### Финальное качественное demo на ResNet18

```bash
PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/final_neural_stage_classification_10/resnet18_10fps_balanced_best_10fps.pt \
  --output reports/demo/MVI_6266_final_system_demo.mp4 \
  --model resnet18 \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device auto
```

#### Demo с CV-trigger

```bash
PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/final_neural_stage_classification_10/resnet18_10fps_balanced_best_10fps.pt \
  --output reports/demo/MVI_6266_final_system_demo_cv.mp4 \
  --model resnet18 \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --enable-cv-trigger \
  --cv-threshold 0.38 \
  --cv-confirm-frames 2 \
  --device auto
```

#### CPU demo на MobileNet

```bash
PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/mobilenet_evolutionary_search/mobilenet_evo_best_finetuned_best.pt \
  --output reports/demo/MVI_6266_final_system_demo_cpu_mobilenet.mp4 \
  --model mobilenet_v3_small \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device cpu
```

### `scripts/benchmark_neural_latency.py`

Финальный latency benchmark.

Мерит:

```text
frame_read_ms
preprocess_ms
model_ms
neural_trigger_ms
cv_trigger_ms
postprocess_ms
total_ms
```

Пример CPU benchmark для MobileNet:

```bash
PYTHONPATH=src python3 scripts/benchmark_neural_latency.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/mobilenet_evolutionary_search/mobilenet_evo_best_finetuned_best.pt \
  --model mobilenet_v3_small \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device cpu \
  --output reports/neural_latency_mobilenet_evo/mobilenet_cpu_latency.csv \
  --summary-output reports/neural_latency_mobilenet_evo/mobilenet_cpu_summary.json
```

Пример CPU benchmark для ResNet18 64:

```bash
PYTHONPATH=src python3 scripts/benchmark_neural_latency.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/resnet_input_size_cpu/resnet18_64_finetuned_best.pt \
  --model resnet18 \
  --roi 470,280,430,290 \
  --image-size 64 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device cpu \
  --output reports/neural_latency_resnet_cpu/resnet18_64_cpu_latency.csv \
  --summary-output reports/neural_latency_resnet_cpu/resnet18_64_cpu_summary.json
```

---

## 8. Кодовые модули

### `src/brazing_sense/control/state_machine.py`

Финальная state machine v1.

Назначение:

- запрещает обратные переходы;
- запрещает перескоки стадий;
- подтверждает переход только после нескольких raw-предсказаний следующей стадии;
- отдаёт стабильную операторскую стадию.

Используется в:

```text
scripts/run_neural_inference_video.py
scripts/benchmark_neural_latency.py
```

---

## 9. Ключевые checkpoints

### Основные модели

Основной качественный ResNet18 stage-classifier.

```text
models/checkpoints/final_neural_stage_classification_10/resnet18_10fps_balanced_best_10fps.pt
```

Базовый MobileNetV3 Small.

```text
models/checkpoints/final_neural_stage_classification_10/mobilenet_v3_small_10fps_balanced_best_10fps.pt
```

### CPU-friendly ResNet

CPU-friendly ResNet18 64×64.

```text
models/checkpoints/resnet_input_size_cpu/resnet18_64_finetuned_best.pt
```

### MobileNet evolutionary

MobileNetV3 Small с эволюционно подобраными гиперпараметрами.

```text
models/checkpoints/mobilenet_evolutionary_search/mobilenet_evo_best_finetuned_best.pt
```

MobileNetV3 Small с эволюционно подобраной классификационной головой.

```text
models/checkpoints/mobilenet_architecture_evolution/mobilenet_arch_evo_039_best.pt
```

---

## 10. Demo-video

Основные demo-video лежат в:

```text
reports/demo/
```

Видео `MVI_6265` вынесены в:

```text
reports/demo/trash/
```

поскольку `MVI_6265` рассматривался как hard-case из-за засветов и неоднозначного поведения флюса.

---

## 11. Отчёты и метрики

### OpenCV baseline

```text
reports/cv_feature_analysis_3/
reports/cv_features_3/
reports/cv_features_10/
```

### Neural baseline 3 FPS

```text
reports/neural_baseline_3/
```

### Final neural stage classification 10 FPS

```text
reports/final_neural_stage_classification_10/
```

Ключевые файлы:

```text
resnet18_base_test_metrics_10.json
resnet18_base_test_metrics_smoothed_10.json
resnet18_base_test_predictions_10.csv
```

### State machine

```text
reports/stage_state_machine/
```

### Active brazing trigger

```text
reports/active_brazing_trigger/
```

Ключевые файлы:

```text
stage_probability_trigger_grid.csv
trigger_methods_summary.csv
```

### CV active brazing trigger

```text
reports/cv_active_brazing_trigger/
```

Ключевой файл:

```text
final_cv_trigger_comparison.csv
```

### CPU optimization

```text
reports/resnet_input_size_cpu/
reports/mobilenet_evolutionary_search/
reports/neural_latency_resnet_cpu/
reports/neural_latency_mobilenet_evo/
```

---

## 12. Основные результаты

### OpenCV baseline

```text
OpenCV + smoothing:
accuracy:    0.6296
macro_f1:    0.5809
weighted_f1: 0.6964
```

Вывод: годится как explainable baseline, но недостаточен как финальное решение.

### Neural stage-classifier

```text
ResNet18 3 FPS:
accuracy:    0.9347
macro_f1:    0.8489
weighted_f1: 0.9307
```

На 10 FPS была улучшена стадия `active_brazing`.

### State machine без hard-case MVI_6265

```text
State machine v1:
accuracy:    0.9581
macro_f1:    0.9155
weighted_f1: 0.9562
```

### Active brazing trigger

```text
P(active_brazing), threshold=0.2, confirm_frames=7:
detected:       3/3
MAE:            0.70 s
within_1.0s:    100%
within_2.0s:    100%
```

Для быстрого управляющего демо использовался режим:

```text
threshold=0.4
confirm_frames=1
```

### CV fallback trigger

```text
score_cv_v1_motion_texture:
threshold:      0.5
confirm_frames: 5
detected:       3/3
MAE:            0.67 s
within_2.0s:    100%
```

### CPU-friendly модели

```text
ResNet18 64×64 fine-tuned:
accuracy:    0.9174
macro_f1:    0.8328
CPU p95:     ~39.8 ms
```

```text
MobileNetV3 Small CPU:
mean:         ~25 ms
p95:          ~39 ms
```

---

## 13. Как воспроизвести основной pipeline

### 1. Установить зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для GPU-версии PyTorch рекомендуется устанавливать `torch` и `torchvision` по официальной инструкции PyTorch с учётом версии CUDA.

### 2. Собрать frame-level датасет

```bash
python3 scripts/build_frame_dataset.py \
  --intervals data/annotations/stage_intervals.csv \
  --output-frames data/processed/frames_10 \
  --output-labels data/annotations/frame_labels_10.csv \
  --save-stats data/annotations/frame_dataset_stats_10.json \
  --fps 10 \
  --overwrite
```

### 3. Создать split

```bash
python3 scripts/split_frame_dataset.py \
  --labels data/annotations/frame_labels_10.csv \
  --output-dir data/annotations/splits_10 \
  --blacklist data/annotations/blacklist.txt \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --overwrite
```

### 4. Запустить финальное demo-video

```bash
PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/final_neural_stage_classification_10/resnet18_10fps_balanced_best_10fps.pt \
  --output reports/demo/MVI_6266_final_system_demo.mp4 \
  --model resnet18 \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device auto
```

### 5. Запустить CPU-demo

```bash
PYTHONPATH=src python3 scripts/run_neural_inference_video.py \
  --video data/raw/MVI_6266.MOV \
  --checkpoint models/checkpoints/mobilenet_evolutionary_search/mobilenet_evo_best_finetuned_best.pt \
  --output reports/demo/MVI_6266_final_system_demo_cpu_mobilenet.mp4 \
  --model mobilenet_v3_small \
  --roi 470,280,430,290 \
  --image-size 224 \
  --postprocess state_machine \
  --trigger-threshold 0.4 \
  --trigger-confirm-frames 1 \
  --device cpu
```

---

## 14. Ограничения

Текущая система является исследовательским прототипом.

Основные ограничения:

- ограниченное число видео;
- разметка выполнена визуально;
- отсутствует синхронизация с реальными температурными данными;
- отсутствует промышленный стенд и целевая камера;
- `MVI_6265` является hard-case из-за засветов и неоднозначной видимости флюса;
- latency зависит от конкретного CPU/GPU;
- `HOLD TEMPERATURE` является рекомендательным сигналом прототипа, а не сертифицированным управляющим контуром.

---

## 15. Итог

В результате исследования построен прототип модуля машинного зрения для определения стадии индукционной пайки и формирования сигнала начала активной пайки.

Финальная система включает:

```text
ROI crop
→ neural stage classifier
→ state machine
→ stable stage
→ P(active_brazing) trigger
→ HOLD TEMPERATURE signal
→ optional CV fallback trigger
```

Проект закрывает основную техническую цель дипломной работы и содержит несколько направлений развития:

- расширение датасета;
- синхронизация с температурой;
- проверка на целевом промышленном стенде;
- ONNX/quantization;
- multi-task модель для стадии и hold-сигнала;
- улучшение устойчивости к засветам и hard-case видео.

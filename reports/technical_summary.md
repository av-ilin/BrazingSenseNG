# Technical Summary: BrazingSense Baseline

## 1. Постановка задачи

В проекте BrazingSense рассматривается задача машинного зрения для определения стадии процесса индукционной пайки волноводных трактов космических аппаратов.

На текущем этапе задача сведена к покадровой классификации ROI-изображения зоны пайки на 4 стадии:

```text
0 — inactive_preparation
1 — flux_activation
2 — active_brazing
3 — stabilization
```

Основная цель текущего технического этапа — проверить, возможно ли по видеоданным устойчиво определять стадию процесса пайки и обеспечить скорость обработки кадра не более 50 мс.

## 2. Данные и разметка

Исходный набор состоял из 18 видеозаписей процесса пайки. Для каждого видео была выполнена интервальная разметка стадий процесса. На основе этой разметки был сформирован frame-level датасет с частотой 3 FPS.

После первичного анализа были исключены 3 видеозаписи с выраженными аномалиями процесса:

```text
MVI_6270
MVI_6273
MVI_6278
```

Итоговый набор после blacklist:

```text
total_videos: 15
total_frames: 2580
```

Разбиение выполнялось по `video_id`, а не по отдельным кадрам, чтобы избежать утечки почти одинаковых соседних кадров между train, validation и test.

Итоговый split:

```text
train: 10 видео, 1731 кадр
val:    2 видео,  328 кадров
test:   3 видео,  521 кадр
```

Распределение классов:

```text
train:
inactive_preparation — 1072
flux_activation      — 148
active_brazing       — 229
stabilization        — 282

val:
inactive_preparation — 222
flux_activation      — 29
active_brazing       — 42
stabilization        — 35

test:
inactive_preparation — 302
flux_activation      — 41
active_brazing       — 58
stabilization        — 120
```

## 3. Область интереса

Для анализа использовалась область интереса, включающая волновод, область флюса, нижний шов и зоны возможного протекания припоя.

Финальный ROI:

```yaml
default_roi:
    x: 470
    y: 280
    w: 430
    h: 290
```

Использование ROI позволило убрать значительную часть фона, оснастки и неизменяемых областей кадра, сосредоточив анализ на зоне, где происходят основные визуальные изменения процесса пайки.

## 4. OpenCV baseline

В морфологической ветке были рассчитаны классические признаки компьютерного зрения:

```text
brightness_mean
brightness_std
value_mean
value_std
saturation_mean
saturation_std
hue_mean
lab_l_mean
lab_a_mean
lab_b_mean
red_mean
green_mean
blue_mean
red_green_diff
red_blue_diff
white_area_ratio
specular_highlight_ratio
dark_area_ratio
warm_area_ratio
edge_density
laplacian_var
frame_diff_score
```

На основе этих признаков были проверены классические модели машинного обучения. Лучший результат показала `LogisticRegression`.

Итог OpenCV baseline на test:

```text
accuracy:    0.6219
macro_f1:    0.5740
weighted_f1: 0.6904
```

После простого temporal smoothing:

```text
accuracy:    0.6296
macro_f1:    0.5809
weighted_f1: 0.6964
```

Основная проблема OpenCV baseline — низкая точность определения стадии `flux_activation`.

Для `flux_activation` после smoothing:

```text
precision = 0.1750
recall    = 0.8537
f1-score  = 0.2905
```

Это означает, что модель часто находит настоящую активацию флюса, но слишком часто ошибочно относит к этой стадии кадры из других этапов.

Основные ошибки OpenCV baseline:

```text
inactive_preparation → flux_activation: 103
stabilization → flux_activation: 42
active_brazing → flux_activation: 22
stabilization → active_brazing: 20
```

Вывод по OpenCV baseline: классические признаки по ROI позволяют частично различать стадии процесса и хорошо подходят как объяснимый baseline, однако в текущем виде они недостаточны для финального решения задачи определения стадии пайки.

## 5. Neural baseline

В нейросетевой ветке были проверены две архитектуры:

```text
ResNet18
MobileNetV3 Small
```

Обе модели использовали предобученные веса ImageNet. Входом модели являлся ROI-кадр, приведённый к размеру 224×224. Для учёта дисбаланса классов использовалась взвешенная функция потерь `CrossEntropyLoss`.

На validation split:

```text
ResNet18:
best_val_macro_f1 = 0.8902
best_val_accuracy = 0.9482

MobileNetV3 Small:
best_val_macro_f1 = 0.8579
best_val_accuracy = 0.9360
```

Лучшей моделью стала `ResNet18`.

Итог ResNet18 на test:

```text
accuracy:    0.9347
macro_f1:    0.8489
weighted_f1: 0.9307
```

Classification report:

```text
                      precision    recall  f1-score   support

inactive_preparation     1.0000    0.9834    0.9917       302
     flux_activation     0.6406    1.0000    0.7810        41
      active_brazing     1.0000    0.5000    0.6667        58
       stabilization     0.9160    1.0000    0.9562       120

            accuracy                         0.9347       521
           macro avg     0.8892    0.8709    0.8489       521
        weighted avg     0.9524    0.9347    0.9307       521
```

Основные ошибки ResNet18:

```text
active_brazing → flux_activation: 18
active_brazing → stabilization: 11
inactive_preparation → flux_activation: 5
```

Таким образом, нейросетевая модель существенно лучше OpenCV baseline определяет стадии процесса. Наиболее уверенно распознаются `inactive_preparation` и `stabilization`. Стадия `flux_activation` также определяется значительно лучше, чем в OpenCV baseline. Основная нерешённая проблема — недостаточная полнота распознавания стадии `active_brazing`.

## 6. Temporal smoothing для нейросети

Для нейросетевой модели была проверена простая временная постобработка в виде majority-vote smoothing по окну из 5 кадров.

ResNet18 без smoothing:

```text
accuracy:    0.9347
macro_f1:    0.8489
weighted_f1: 0.9307
```

ResNet18 + temporal smoothing:

```text
accuracy:    0.9367
macro_f1:    0.8477
weighted_f1: 0.9315
```

Classification report после smoothing:

```text
                      precision    recall  f1-score   support

inactive_preparation     1.0000    0.9901    0.9950       302
     flux_activation     0.6508    1.0000    0.7885        41
      active_brazing     1.0000    0.4828    0.6512        58
       stabilization     0.9160    1.0000    0.9562       120

            accuracy                         0.9367       521
           macro avg     0.8917    0.8682    0.8477       521
        weighted avg     0.9532    0.9367    0.9315       521
```

Smoothing немного повысил общую accuracy и weighted-F1, однако практически не изменил macro-F1 и немного снизил recall для `active_brazing`. Это связано с тем, что `active_brazing` является относительно короткой стадией, и простое сглаживание может частично размывать её границы.

Вывод: для нейросетевой модели простое temporal smoothing не является ключевым улучшением. В дальнейшем более перспективной является технологическая логика переходов между стадиями, а не простое majority-vote сглаживание.

## 7. Сравнение OpenCV и neural baseline

Сравнение лучших результатов:

```text
OpenCV baseline + smoothing:
accuracy:    0.6296
macro_f1:    0.5809
weighted_f1: 0.6964

Neural baseline ResNet18:
accuracy:    0.9347
macro_f1:    0.8489
weighted_f1: 0.9307

Neural baseline ResNet18 + smoothing:
accuracy:    0.9367
macro_f1:    0.8477
weighted_f1: 0.9315
```

Прирост ResNet18 относительно OpenCV baseline + smoothing:

```text
accuracy:    +0.3051
macro_f1:    +0.2680
weighted_f1: +0.2343
```

Нейросетевой подход оказался существенно более эффективным для определения стадии пайки. OpenCV baseline остаётся полезным как объяснимый диагностический инструмент, но финальная система определения стадии должна строиться на нейросетевой модели.

## 8. Demo-video inference

Был реализован скрипт:

```text
scripts/run_neural_inference_video.py
```

Скрипт выполняет инференс по исходному видео и формирует demo-video с overlay:

```text
Raw Stage
Smooth Stage
Confidence
Inference
Time
ROI-прямоугольник вокруг зоны пайки
```

Таким образом был получен демонстрационный прототип операторского мониторинга:

```text
исходное видео → ROI → ResNet18 → стадия пайки на каждом кадре → demo-video
```

## 9. End-to-end latency benchmark

Был реализован скрипт:

```text
scripts/benchmark_neural_latency.py
```

В отличие от первичного model-only benchmark, данный скрипт измеряет полный pipeline:

```text
read frame
→ ROI crop
→ BGR/RGB conversion
→ resize
→ normalize
→ model.forward
→ softmax/argmax
→ temporal smoothing
→ result
```

Benchmark проводился на CUDA-среде для полных видео `MVI_6266` и `MVI_6279`.

Результаты для `MVI_6266`:

```text
total_ms mean: 6.45 мс/кадр
total_ms p95:  7.80 мс/кадр
estimated FPS mean: ~155 FPS
estimated FPS p95:  ~128 FPS
```

Результаты для `MVI_6279`:

```text
total_ms mean: 6.43 мс/кадр
total_ms p95:  7.82 мс/кадр
estimated FPS mean: ~155 FPS
estimated FPS p95:  ~128 FPS
```

Целевое ограничение:

```text
total_ms <= 50 мс/кадр
```

На текущей CUDA-среде полный pipeline уверенно укладывается в ограничение 50 мс/кадр с большим запасом. Значение p95 находится около 7.8 мс/кадр, что примерно в 6 раз быстрее целевого ограничения.

При этом данный вывод относится к текущей вычислительной среде. Для промышленного или операторского внедрения benchmark необходимо повторить на целевом компьютере и с реальным способом получения видеопотока.

## 10. Текущий статус

На данном этапе реализованы:

```text
1. Разметка стадий пайки по видео.
2. Frame-level датасет на 3 FPS.
3. Blacklist аномальных видео.
4. Video-level train/val/test split.
5. ROI selection.
6. OpenCV baseline.
7. Neural baseline.
8. Сравнение ResNet18 и MobileNetV3 Small.
9. Temporal smoothing.
10. Demo-video inference.
11. End-to-end latency benchmark.
```

Текущий прототип уже решает задачу операторского определения стадии пайки на уровне исследовательского baseline.

## 11. Основные выводы

1. Использование ROI является важным условием для анализа процесса пайки, поскольку значимые визуальные изменения занимают только часть кадра.

2. OpenCV-признаки позволяют частично различать стадии процесса, но недостаточно устойчивы для финального определения стадии.

3. Нейросетевой подход на основе ResNet18 существенно превосходит OpenCV baseline по всем основным метрикам.

4. Модель уверенно определяет `inactive_preparation`, `flux_activation` и `stabilization`.

5. Основная проблема текущей модели — недостаточная полнота распознавания стадии `active_brazing`.

6. Простое majority-vote smoothing почти не улучшает нейросетевую модель и может слегка размывать короткую стадию `active_brazing`.

7. Полный end-to-end pipeline на текущей CUDA-среде работает значительно быстрее целевого ограничения 50 мс/кадр.

8. Текущая версия является годной основой для системы операторского мониторинга стадий пайки.

## 12. Следующие направления работы

Дальнейшее развитие проекта целесообразно вести в нескольких направлениях:

```text
1. Улучшение распознавания стадии active_brazing.
2. Проверка датасета с большей частотой кадров, например 10 FPS.
3. Подбор class weights или WeightedRandomSampler для active_brazing.
4. Проверка моделей с учётом временного контекста.
5. Разработка state machine для технологически допустимых переходов.
6. Отдельное исследование события active_brazing_started.
```

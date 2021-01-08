# Object Tracking Code Migration (Tracking Only)

## 기존 코드 분석 및 이해

1. 비디오 설정 하기 

`run_tracker_evaluation.py` 의 16번째 라인에 다음과 같은 코드가 있다.

```python
hp, evaluation, run, env, design = parse_arguments()
```

` parse_arguments.py`의 `parse_arguments()` 함수를 확인해 보면 `parameters/evaluation.json` 이라는 파일에서 설정값을 읽어오는 것을 알 수 있다.

```json
{
	"n_subseq": 3,
	"dist_threshold": 20,
	"stop_on_failure": 0,
	"dataset": "validation",
	"video": "all", // data sequece의 이름을 넣거나 전체일 경우 'all'
	"start_frame": 0  // 항상 처음 사진이 exampler가 된다.
} 
```

2. evaluation 하는 부분 분석

`run_tracker_evaluation.py`의 71~82번째 라인을 통해서 evaluation을 하는 것으로 보여 진다. 이 부분의 코드를 분석해보자. 전체 코드 중 중요한 부분은 

```python
# 1. 사진데이터 로드 및 물체의 자리 인식
gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video) 
# 2. exampler data의 중앙부분 찾기
pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame]) 
bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                        filename, image, templates_z, scores, evaluation.start_frame)
_, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
print evaluation.video + \
' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
' -- Precision AUC: ' + "%.2f" % precision_auc + \
' -- IOU: ' + "%.2f" % iou + \
' -- Speed: ' + "%.2f" % speed + ' --'
print
```



2. exampler data의 중앙 찾기

`region_to_bbox.py`의 소스코드 일부분이다. 이미지의 중앙의 위치와 이미지의 너비, 높이를 반환하는 것을 알 수 있다.

```python
x = region[0]
y = region[1]
w = region[2]
h = region[3]
cx = x+w/2
cy = y+h/2
return cx, cy, w, h
```



3. 

```
# Set size for use with tf.image.resize_images with align_corners=True.
# For example,
#   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
# instead of
# [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
final_score_sz = hp.response_up * (design.score_sz - 1) + 1
```



4. build Siames Network
```
# build TF graph once for all
filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)
```
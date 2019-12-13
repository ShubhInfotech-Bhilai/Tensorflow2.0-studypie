### Classification
- Threashold(임계값)
- 확률을 그대로 사용하거나 변환해 사용할 수 있음
- 특정 임계값을 토대로 T / F로 변환 가능
	- 물론 임계값을 조정하면 어떻게 되는지가 달라짐

	
### Confusion Matrix
- <img src="https://www.dropbox.com/s/0kxsu1czlnuuhff/Screenshot%202019-12-14%2001.52.22.png?raw=1">


### 정확성(Accuracy)
- <img src="https://www.dropbox.com/s/9wmmyz82n56ypmu/Screenshot%202019-12-14%2001.52.55.png?raw=1">

- 하지만 클래스 불균형이 존재할 경우엔 정확성만으로 모든 것을 평가할 순 없음. 이럴 경우 정밀도(Precision), 재현율(Recall)을 봐야함

### 정밀도(Precision)
- <img src="https://www.dropbox.com/s/na7tyf8vk9xb9gr/Screenshot%202019-12-14%2001.54.16.png?raw=1">


### 재현율(Recall)
- <img src="https://www.dropbox.com/s/rapbuxz7ssy6p28/Screenshot%202019-12-14%2001.54.45.png?raw=1">


### 정밀도와 재현율의 관계
- 모델의 효과를 완전히 평가하려면 정밀도, 재현율 모두 검사해야 함
- 이 두 값은 상충할 경우가 있음
- 정밀도가 향상되면 재현율이 감소되고 반대의 경우도 마찬가지


### ROC Curve
- 모든 임계값에서 분류 모델의 성능이 어떻게 되는지 알려줌
- TPR(참 양성 비율), FPR(허위 양성 비율)이 존재
- TPR = TP/(TP+FN)
- FPR = FP/(FP+TN)

- <img src="https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg">

### AUC
- ROC 곡선 아래 영역
- AUC가 이상적인 이유
	- 척도 불변(scale-invariant) : 절대값이 아닌 예측이 얼마나 잘 평가되는지 측정함
	- classification-threshold-invariant : AUC 어떤 임계값이 선택되었는지 상관없이 모델의 예측 품질을 측정
- AUC의 단점
	- 척도 불변이 항상 이상적이진 않음. 잘 보정된 확률 결과가 필요한 경우 AUC론 알 수 없음
	- 분류 임계값 불변이 항상 이상적인 것은 아님. FN, FP 비용에 큰 차이가 있으면 한가지 유형의 분류 오류를 최소화하는 것은 윟머할 수 있음
		- 이런 유형의 최적화에 유용한 측정항목은 아님

		
### Prediction bias
- 로지스틱 회귀 예측은 편향되지 않아야 함
- 예측 평균과 관찰 평균은 유사해야 함
- prediction bias = average of predictions - average of labels in data set
- 0이 아닌 유의미한 예측 편향은 모델 어디에 버그가 있다는 것. 양성인 라벨이 발생하는 빈도가 모델이 잘못되었음을 보여줌
- 평균적으로 전체 이메일의 1%가 스팸이라고 할 경우, 평균적으로 메일의 1%는 스팸일 수 있다고 예측해야 함. 평균을 내면 결과가 1%가 아닌 20%라면 예측 편향을 드러냄
- 예측 편향의 원인
	- 불완전한 특성 세트
	- 노이즈 데이터 세트
	- 결함이 있는 파이프라인
	- 편향된 학습 샘플
	- 지나치게 강한 정규화
- 예측 편향을 줄이기 위해 모델의 결과를 조정하는 캘리브레이션 레이어(calibration layer)를 추가해 학습한 모델을 사후 처리하는 방법으로 예측 편향을 수정할 수 있음
- 모델의 편향이 +3%라면 평균 예측을 3% 낮추는 레이어를 추가할 수 있음
- 하지만 캘리브레이션 레이어 추가가 바람직한 것은 아님
	- 원인이 아닌 증상만 수정함
	- 최신 상태로 유지하기 어려운 불안정한 시스템이 구축됨
- 가능하면 사용하지 않는게 좋고, 이 레이어가 모델의 모든 문제를 수정해 의존도가 높아지는 경향이 있음 => 유지 보수가 더 어려울 수 있음



### Bucketing and Prediction Bias
- 버킷을 대상으로 예측 편향을 검사해야 함
- 버킷 구성
	- 타겟 예측을 선형으로 분류
	- 분위 형성
- <img src="https://developers.google.com/machine-learning/crash-course/images/BucketingBias.svg">
- Why are the predictions so poor for only part of the model? Here are a few possibilities:
	- The training set doesn't adequately represent certain subsets of the data space.
	- Some subsets of the data set are noisier than others.
	- The model is overly regularized. (Consider reducing the value of lambda.)
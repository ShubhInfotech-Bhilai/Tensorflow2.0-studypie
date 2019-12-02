
## 주요 ML 용어 및 선형 회귀 이해하기

### Introduction to ML
- Example : particular instance of data, x
- Labeled example has {features, label}: (x, y)
	- Used to train the model
- Unlabeld example has {features, ?}: (x, ?)
	- Used for making predictions on new data
- Model : to predict labels: y`
	- Defined by internal parameters, which are learned

---

### Framing
- Label
	- 단순 선형 회귀 분석에서 우리가 예측하는 y 변수, 밀의 미래, 동물의 종류, 오디오 클립 등 거의 모든것이 가능
- Features
	- 단순한 Feature부터 많은 Feature까지 가짐
- Model
	- Training : 모델을 만들거나 학습. 모델 레이블이 지정된 예제를 보고 피쳐 레이블 관계를 학습
	- Inference : 모델을 레이블이 없는 예제에 적용하는 것. 훈련된 모델을 사용해 예측함
- Regression
	- continuous 값 예측
	- 캘리포니아의 집의 가치는?
	- 사용자가 이 광고를 클릭할 확률은?
- Classification
	- 불연속 값을 예측
	- 이메일이 스팸인지 아닌지?
	- 이 사진은 개, 고양이, 햄스터인지?    

### Descending into ML
- Linear Regression	
	- <img src="https://developers.google.com/machine-learning/crash-course/images/CricketLine.svg">
	- $$y' = b + w_{1} x_{1}$$
	- y' : 예측 레이블
	- b : 바이어스(y절편), 가끔 $$w_{0}$$
	- $$w_{1}$$ : 피쳐 1의 weight, 기울기
	- $$x_{1}$$ : Feature
- Training and Loss
	- Training : 모든 가중치와 레이블이 있는 데이터에서 weight와 bias에 좋은 값을 찾는 것
		- Supervised learning에서 알고리즘은 loss를 최소화하는 모델을 찾아 만듬
		- Expirical risk minimization이라고 함(경험적인 위험 최소화?)
	- loss는 나쁜 예측에 대한 패널티
		- 모델 예측이 단일 예에서 얼마나 나쁜지를 나타내는 숫자
		- 모형 예측이 완벽하면 loss는 0
		- 왼쪽 모델이 loss가 큼
		- <img src="https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png">
	- MSE : 전체 데이터셋에 대한 average squared loss
		- 개별 example에 대한 모든 제곱 손실을 합한 후 example 수로 나눔
		- $$MSE = \frac 1N \sum(y-prediction(x))^{2}$$
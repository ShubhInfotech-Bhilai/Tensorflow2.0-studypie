## Reducing Loss

### Iterative Approach
- 반복을 통해 어떻게 손실을 줄이는지 확인할 예정
- <img src="https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg">
- 전체 손실이 변하지 않거나 매우 느리게 변할때까지 계산함. 이럴 경우 수렴했다고 함


### Gradient Descent
- 전체 데이터 세트에 상상할 수 있는 모든 w1 값의 손실 함수를 계산하는 것은 매우 비효율적
- 경사하강법은 w1에 대한 시작값을 선택하고 손실 곡선의 기울기 계산함
	- 기울기는 편미분의 벡터라 어느 방향이 더 정확하거나 부정확한지 알려줌
	- 기울기는 벡터라 방향과 크기를 나타냄


### Learning rate
- 경사하강법에서 기울기에 학습률(learning rate), 보폭이라는 스칼라는 곱해 다음 지점을 결정함
- 하이퍼 파라미터로 지정함
- lr이 작으면 속도가 매우 느리고 너무 크면 최소값을 지날 수 있음
- 적당한 lr을 찾아야 효율적으로 결과를 얻음


### Optimizing Learning Rate
- 인터랙티브하게 lr을 설정해서 어떻게 결과가 보이는지 보여줌! 시각적으로 좋다


### Stochastic Gradient Descent
- 경사하강법에서 Batch : 기울기를 계산할 때 사용하는 example의 수
- 배치가 전체 데이터세트라고 가정했지만, 실제론 작은 만큼을 계산함
- 배치의 크기가 커지면 중복의 가능성도 커지고, 적당한 중복은 노이즈가 있을 수 있음
	- 반복당 배치 크기만 사용함. 반복이 충분하면 SGD가 효과가 있지만 노이즈가 심해짐
- 따라서 미니 배치 확률적 경사하강법을 사용하기도 함. 10~1000개 사이로 구성 
- 적은 자원일 경우 small batch는 loss 줄어드는 속도가 빠르지만 멀티 GPU라면 Large Batch가 좋을듯 
### Tensorflow 2.0
- 2.0 사용하기 매우 쉬워짐
	- Session 안녕
- <img src="https://www.dropbox.com/s/vvaodefeptwkxqv/Screenshot%202019-12-08%2022.14.06.png?raw=1">
	
- Sequential API : 일반적인 분이라면 이정도면 ㅇㅋ
- Functional API
	- Custom Layers, Metrics, Losses
- Subclassing : 연구자

- Session을 쓰지 않고 함수를 씀
- Sequential API
	- Keras.models.Sequential에서 layer들을 넣으면 아키텍쳐 구성
	- Compile
	- Fit
	- Evaluate
- Functional API
	- Visual QnA
	- 일반적이지 않은 Input, Output
	- Layer간 공유되는 경우
	- Auto Encoder
- Custom Layer
	- 새성자가 있고 call이 있고
	- 전문가용

- model.fit()으로 학습해도 되고
	- 사실 코드를 까보면 커스터마이징할 수 있음
	- tf.GradientTape()를 사용 	
- Multi-GPU 학습이 쉬움
	- strategy =  tf.distribute.MirroredStrategy() 정의하고
	- with strategy.scope()를 지정하면 됨
- Numpy처럼 TF 2.0을 사용할 수 있음


### 머신러닝 기초 Part 1
- 미니배치
	- 2의 제곱수를 사용하는 이유는 벡터 계산이 2의 제곱수가 입력될 때 더 빠르기 때문
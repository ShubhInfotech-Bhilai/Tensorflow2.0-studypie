### Tensorflow 사용하기

<img src="https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg">


- Estimator : 하이레벨 API
- tf.layers, tf.losses, tf.metrics : 일반 모델용
- tensorflow : 로우레벨 API
- Tensorflow는 Graph protocol buffer와 graph runtime 2가지로 이루어짐
	- 자바 컴파일러와 JVM과 유사함
- 하이레벨일수록 유연성이 떨어짐 



### tf.estimator API
- 대부분 sklearn과 호환됨

```
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```
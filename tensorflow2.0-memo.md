
### Tensorflow 2.0 메모
- 기억할 API 메모해두기
- 꼭 Tensorflow 이야기가 아니여도 유용한 것들은 저장
- tf hub, tf dataset
	
	```
	import tensorflow_hub as hub
	import tensorflow_datasets as tfds
	```
	
- 즉시 실행 모드 활성화

	```
	tf.executing_eagerly())
	```
	
- tfds.Split 함수
	- 왜 split이 아니라 Split이지..
	- [참고 문서](https://github.com/tensorflow/datasets/blob/master/docs/splits.md)
	- DatasetBuilder가 서브셋을 나눠줌
	- 데이터 구성시 `tfds.load()` 또는 `tfds.DatasetBuilder.as_dataset()` 사용해 `tf.data.Dataset` 인스턴스를 생성시, split할 포인트를 지정할 수 있음 

- tfds.load 함수
	- tf.data.Dataset에 있는 데이터셋을 load

	```
	tfds.load(name, split=None, data_dir=None, batch_size=None, in_memory=None, shuffle_files=False, 
	download=True, as_supervised=False, decoders=None, with_info=False, builder_kwargs=None, 
	download_and_prepare_kwargs=None, as_dataset_kwargs=None, try_gcs=False)
	```
	
- 텐서플로 Hub에서 임베딩 모델 사용하기

	```
	embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], 
	                           dtype=tf.string, trainable=True)
	hub_layer(train_examples_batch[:3])
	
	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	
	model.summary()
	```
		
- Keras Flow
	
	```
	Model 생성한 후, Layer 추가
	# optimizer, metrics 지정
	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
   history = model.fit(train_data.shuffle(10000).batch(512), epochs=20, 
   validation_data=validation_data.batch(512), verbose=1)
   # 평가
   results = model.evaluate(test_data.batch(512), verbose=2)
	for name, value in zip(model.metrics_names, results):
		print(f"{name}-{value:.3f}")
	```	
	
- Keras Callback Print
	- [Document](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)

	```
	class PrintDot(keras.callbacks.Callback):
	    def on_epoch_end(self, epoch, logs):
	        if epoch % 100 == 0:
	            print('')
	        print('.', end='')
	        
    EPOCHS = 1000

	history = model.fit(normed_train_data, train_labels,
	                    epochs=EPOCHS, validation_split=0.2, verbose=0,
	                    callbacks=[PrintDot()])
	```	
	
- Early Stopping

	```
	model = build_model()

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
	
	history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, 
	                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
	
	plot_history(history)
	```
	
- y_true, prediction Graph

	```
	test_predictions = model.predict(normed_test_data).flatten()

	plt.scatter(test_labels, test_predictions)
	plt.xlabel('True Values [MPG]')
	plt.ylabel('Predictions [MPG]')
	plt.axis('equal')
	plt.axis('square')
	plt.xlim([0,plt.xlim()[1]])
	plt.ylim([0,plt.ylim()[1]])
	_ = plt.plot([-100, 100], [-100, 100])
	```	
	
- 데이터를 스트리밍으로 공급하는 tf.data.Dataset을 사용해 데이터 변환
	- [Document](https://www.tensorflow.org/guide/data)
	- `tf.data.Dataset.from_tensor_slices()` 사용

	```
	def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
	    def input_function():
	        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
	        if shuffle:
	            ds = ds.shuffle(1000)
	        ds = ds.batch(batch_size).repeat(num_epochs)
	        return ds
	    return input_function
		
	train_input_fn = make_input_fn(dftrain, y_train)
	eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
	```
	
- Layer dense_features 사용시 Warning
	- dtype float64 => float32로 캐스팅함
		
	```
	tensorflow:Layer dense_features is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2
	``` 
	
	- 디폴트로 float64 하고 싶으면 아래 명령어 사용

	```
	tf.keras.backend.set_floatx('float64')
	```
	
- tf.estimator
	- 데이터를 iterate하며 받도록 만들기
	- tf.estimator.LinearClassifier로 정의
	- train
	- evaluate

	```
	linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
	linear_est.train(train_input_fn)
	result = linear_est.evaluate(eval_input_fn)
	
	print(result)
	```

- tf.features_column.crossed_column()
	- 범주형 Feature에 feature cross를 제공함 
	- binning한 뒤 feature cross
	- 예를 들면 아래 위도 경도를 5개 구간으로 분할하고 feature corss
	- <img src="https://www.dropbox.com/s/ks0ebi0mkp5rblf/Screenshot%202019-12-07%2000.44.03.png?raw=1">

	```
	age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
	
	derived_feature_columns = [age_x_gender]
	linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
	linear_est.train(train_input_fn)
	result = linear_est.evaluate(eval_input_fn)
	
	print(result)
	```
	
	- LinearClassifier의 feature_columns 인자 넣을때 +로 연결하는게 신기함
	
- ROC Curve

	```
	from sklearn.metrics import roc_curve
	from matplotlib import pyplot as plt
	
	fpr, tpr, _ = roc_curve(y_eval, probs)
	plt.plot(fpr, tpr)
	plt.title('ROC curve')
	plt.xlabel('오탐률(false positive rate)')
	plt.ylabel('정탐률(true positive rate)')
	plt.xlim(0,)
	plt.ylim(0,)
	```	
	
	

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
	

- L2 Regularization
	- kernel_regularizer=keras.regularizers.l2로 추가
	
	```
	l2_model = keras.models.Sequential([
	    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
	                       activation='relu', input_shape=(NUM_WORDS,)),
	    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
	                       activation='relu'),
	    keras.layers.Dense(1, activation='sigmoid')
	])
		
	l2_model.compile(optimizer='adam',
	                 loss='binary_crossentropy',
	                 metrics=['accuracy', 'binary_crossentropy'])
		
	l2_model_history = l2_model.fit(train_data, train_labels,
	                                epochs=20,
	                                batch_size=512,
	                                validation_data=(test_data, test_labels),
	                                verbose=2)
    ```	
    
- numpy printoption 수정

	```
	np.set_printoptions(precision=3, suppress=True)
	```
	
- show_batch
	- 배치에 어떤 데이터가 들어가있는지 확인

	```
	def show_batch(dataset):
	    for batch, label in dataset.take(1):
	        for key, value in batch.items():
	            print("{:20s}: {}".format(key,value.numpy()))
	```    
	
- numeric feature의 리스트를 선택하고 하나의 컬럼으로 pack하는 전처리 클래스

	```
	class PackNumericFeatures(object):
	    def __init__(self, names):
	        self.names = names
	
	    def __call__(self, features, labels):
	        numeric_features = [features.pop(name) for name in self.names]
	        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
	        numeric_features = tf.stack(numeric_features, axis=-1)
	        features['numeric'] = numeric_features
	
	        return features, labels
	```	
	
- functools.partial을 사용해 MEAN, STD를 normalizer에 바인딩

	```
	normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
	
	numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
	numeric_columns = [numeric_column]
	numeric_column
	```	
	
- tf.keras.layers.DenseFeatures
	- [문서](https://www.tensorflow.org/api_docs/python/tf/keras/layers/DenseFeatures)
	- 주어진 feature_columns를 기반으로 dense 텐서를 생성하는 레이어

	```
	numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
	numeric_layer(example_batch).numpy()
	```
	
- tf.feature_column.categorical_column_with_vocabulary_list
	- vocabulary_list에 기반해 string -> integer 매핑

	```
	categorical_columns = []
	for feature, vocab in CATEGORIES.items():
	  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
	        key=feature, vocabulary_list=vocab)
	  categorical_columns.append(tf.feature_column.indicator_column(cat_col))
	```
	
- 전처리 레이어를 만든 후, 모델 만들기

	```
	model = tf.keras.Sequential([
	  preprocessing_layer,
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dense(1, activation='sigmoid'),
	])
	
	model.compile(
	    loss='binary_crossentropy',
	    optimizer='adam',
	    metrics=['accuracy'])
	```	
	
	
- TFRecord
	- 데이터를 효율적으로 읽으려면 데이터를 직렬화(serialize)해서 선형으로 읽을 수 있는 파일로 저장하면 도움이 됨
	- 데이터가 네트워크를 통해 스트리밍되는 경우 특히 그럼
	- 데이터 전처리를 캐싱할 때 유용할 수 있음
	- TFRecord는 binary record를 저장하기 위한 간단한 형식
	- Protocol buffer는 구조화된 데이터의 효율적인 직렬화를 위한 플랫폼간, 언어간 라이브러리
		- 프로토콜 메세지는 .proto 파일로 정의되고 메세지 유형을 이해하는 쉬운 방법 중 하나	
	- tf.Example 메세지(프로토버퍼)는 {"string":value} 매핑을 나타냄. TFX와 같은 higher-level API에서 사용됨
	- 이런 방식은 유용하지만 선택 사항
	- tf.data를 사용하지 않고 데이터를 읽는 부분에서 병목이 없다면 굳이 TFRecords를 사용하지 않아도 됨
	- 데이터셋 퍼포먼스 팁이 궁금하면 [Data Input Pipeline Performance](https://www.tensorflow.org/guide/data_performance) 참고
	- TFRecord 파일은 리코드들을 포함함
	- 각 record는 byte string을 포함함
	- CRCs는 [문서](https://en.wikipedia.org/wiki/Cyclic_redundancy_check) 참고
	- tf.data에서도 가능함
	- [tfrecord colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/tfrecord.ipynb#scrollTo=x2LT2JCqhoD_) 참고하면 좋음

- tf.Example
	- tf.Example는 {"string": tf.train.Feature}로 매핑됨 
	- tf.train.Feature message는 총 3가지 메세지를 받을 수 있음
	- 궁금하면 [proto buffer 구현](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) 참고	
	- 1) tf.train.BytesList
		- string
		- byte
	- 2) tf.train.FloatList
		- float(float32)
		- double(float64)
	- 3) tf.train.Int64List
		- bool
		- enum
		- int32
		- uint32
		- int64
		- uint64
- Tensorflow type을 tf.Example 호환 가능한 tf.trainFeature로 변환하고 싶은 경우 사용할 함수 

	```
	# The following functions can be used to convert a value to a type compatible
	# with tf.Example.
	
	def _bytes_feature(value):
	    """Returns a bytes_list from a string / byte."""
	    if isinstance(value, type(tf.constant(0))):
	        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
	def _float_feature(value):
	    """Returns a float_list from a float / double."""
	    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
	
	def _int64_feature(value):
	    """Returns an int64_list from a bool / enum / int / uint."""
	    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	```
	
- 스칼라 이외의 feature를 같이 처리하고 싶은 경우
	- tf.serialize_tensor를 사용해 텐서를 이진 문자열로 변환
	- 문자열은 tf에서 스칼라임
	- 이진 문자열을 다시 텐서로 변환하고 싶으면 tf.parse_tensor 사용
- proto message를 binary-string으로 serialize하기
	- feature.SerializeToString() 사용
	
	```
	feature = _float_feature(np.exp(1))
	
	feature.SerializeToString()
	```	
	
- Feature를 proto로 Serialize하는 예시(tf.Example)

	```
	def serialize_example(feature0, feature1, feature2, feature3):
	    """
	    Creates a tf.Example message ready to be written to a file.
	    """
	    # Create a dictionary mapping the feature name to the tf.Example-compatible
	    # data type.
	    feature = {
	        'feature0': _int64_feature(feature0),
	        'feature1': _int64_feature(feature1),
	        'feature2': _bytes_feature(feature2),
	        'feature3': _float_feature(feature3),
	    }
	
	    # Create a Features message using tf.train.Example.
	
	    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	    return example_proto.SerializeToString()
	```
	
- tf.data에서 TFRecord로 Serialize	


	```
	# The number of observations in the dataset.
	n_observations = int(1e4)
	
	# Boolean feature, encoded as False or True.
	feature0 = np.random.choice([False, True], n_observations)
	
	# Integer feature, random from 0 to 4.
	feature1 = np.random.randint(0, 5, n_observations)
	
	# String feature
	strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
	feature2 = strings[feature1]
	
	# Float feature, from a standard normal distribution
	feature3 = np.random.randn(n_observations)
	
	features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
	
	def tf_serialize_example(f0,f1,f2,f3):
	    tf_string = tf.py_function(
	        serialize_example,
	        (f0,f1,f2,f3),  # pass these args to the above function.
	        tf.string)      # the return type is `tf.string`.
	    return tf.reshape(tf_string, ()) # The result is a scalar
	
	serialized_features_dataset = features_dataset.map(tf_serialize_example)
	serialized_features_dataset    
	```
	
- tf.io를 사용해 순수 파이썬 함수로 TFRecord 접근 가능

```
# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)
        
        
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)	
```

- 학습시 checkpoint callback 함수 사용하기

	```
	checkpoint_path = "training_1/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	
	# 체크포인트 콜백 만들기
	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
	                                                 save_weights_only=True,
	                                                 verbose=1)
	
	model = create_model()
	
	model.fit(train_images, train_labels,  epochs = 10,
	          validation_data = (test_images,test_labels),
	          callbacks = [cp_callback])  # 훈련 단계에 콜백을 전달합니다
	
	# 옵티마이저의 상태를 저장하는 것과 관련되어 경고가 발생할 수 있습니다.
	# 이 경고는 (그리고 이 노트북의 다른 비슷한 경고는) 이전 사용 방식을 권장하지 않기 위함이며 무시해도 좋습니다.
	```

- checkpoint weight load

	```
	model.load_weights(checkpoint_path)
	loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
	```    
	
- checkpoint callback 매개변수
	- 파일 이름에 에포크 번호 포함
	
	```	
	# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
	checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	
	cp_callback = tf.keras.callbacks.ModelCheckpoint(
	    checkpoint_path, verbose=1, save_weights_only=True,
	    # 다섯 번째 에포크마다 가중치를 저장합니다
	    period=5)
	
	model = create_model()
	model.save_weights(checkpoint_path.format(epoch=0))
	model.fit(train_images, train_labels,
	          epochs = 50, callbacks = [cp_callback],
	          validation_data = (test_images,test_labels),
	          verbose=0)
	```	
	
- 수동으로 weight 저장하기

	```
	# 가중치를 저장합니다
	model.save_weights('./checkpoints/my_checkpoint')
	
	# 가중치를 복원합니다
	model = create_model()
	model.load_weights('./checkpoints/my_checkpoint')
	
	loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
	```	
	
- 모델 전체 저장하기(HDF5 파일)
	- 저장시 weight 값, 모델 설정(구조), 옵티마이저 설정을 저장함
	
	```
	model = create_model()
	
	model.fit(train_images, train_labels, epochs=5)
	
	# 전체 모델을 HDF5 파일로 저장합니다
	model.save('my_model.h5')
	
	# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
	new_model = keras.models.load_model('my_model.h5')
	new_model.summary()
	
	loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
	print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
	```	
	
	
	
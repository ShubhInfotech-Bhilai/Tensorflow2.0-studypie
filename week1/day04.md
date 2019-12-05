### Generalization
- 일반화는 모델이 이를 만들기 위해 사용된 것과 같은 분포에서 추출된 이전에 보지 못했던 새로운 데이터에 제대로 적합할 수 있는지를 나타냅니다
- 모델 성능을 높게 만들려고 오버피팅하면, 새로운 사례에 robust하지 않음
- ML 기본 가정
	- 독립적이고 동일한 방식으로 추출(iid)
	- 분포가 정상성을 보이며 시간이 지나도 변하지 않는다
	- train, valid, test를 같은 분포에서 추출한다



<img src="https://developers.google.com/machine-learning/crash-course/images/GeneralizationB.png">

- 적합해 보이지만 매우 빡빡함


<img src="https://developers.google.com/machine-learning/crash-course/images/GeneralizationC.png">

- 매우 과적합
- 복잡한 것보다 간단한 공식이나 이론을 선택해야 한다
	- ML 모델이 덜 복잡할수록 샘플의 특성 때문이 아니어도 좋은 경험적 결과를 얻을 가능성이 높음

	
### 학습 및 테스트 세트
- 데이터가 하나뿐이라면?
- Train / Test 분리
	- Test로 학습하지 않기!
- Test
	- 통계적으로 유의미한 결과를 도출할 만큼 커야 합니다.
	- 데이터 세트를 전체적으로 나타내야 합니다. 즉, 평가 세트가 학습 세트와 같은 특징을 가지도록 선별해야 합니다.	

### 검증 세트
- Train 학습 => Test 평가 여러번 체크
- workflow 개선 => 검증(validation) 세트 활용
- Train => Validation으로 모델 검증 => Test 에서 결과 확인
- 과적합 안될 확률 높아짐

<img src="https://developers.google.com/machine-learning/crash-course/images/PartitionThreeSets.svg">	
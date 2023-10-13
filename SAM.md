# Segment Anything Model(SAM)

논문을 읽는 이유

- K-예능, 드라마의 Text를 OCR으로 Text Detection & Recognition 후에 영어로 Translation해서 동일한 Style으로 Transfer 하는 프로젝트 목표와 일치하는 Task이다.
- 폰트 스타일을 적용시킨 생성 이미지에서 텍스트 segment를 해야하는 Task에 대해서 SAM 모델의 task, model, dataset을 파악하여 Fine Tuning하기 위해 논문을 리뷰한다.

Aim

- 신속한 분할 작업, 엔지니어링을 통한 데이터 주석 구동
- 다양한 작업을 통해 제로샷 전송이 가능한 SAM 모델
- 10억 개 이상의 마스크 데이터 세트 SA-1B 수집을 위한 데이터 엔진의 세 가지 상호 연결된 구성요소를 도입하여 분할을 위한 기초 모델 구축

Abstract

- 데이터 수집 루프에서 효율적인 모델을 통해 가장 큰 segmentation 데이터셋 구축
- 11M 라이센스 및 개인정보 보호 이미지에 10억 개 이상의 마스크 사용
- 제로샷을 새로운 이미지 배포 및 작업으로 전송이 가능하도록 신속하게 설계, 훈련

Task

- Next token 예측 task는 기초 모델 사전훈련, 신속한 엔지니어링으로 다양한 다운스트림 작업해결
- NPL에서 세그먼트화로 번역하는 것으로 시작
- prompt → 전경/ 배경 지점, 박스 또는 마스크, 자유 형식 텍스트나 일반적으로 이미지에서 분할할 내용을 나타내는 모든 정보
- 신속한 분할 작업은 prompt가 지정된 유효한 분할 마스크를 반환
- valid 마스크의 요구조건은 prompt가 모호하고 여러 개체를 참조하는 경우에도 의미가 있음

<img width="528" alt="스크린샷 2023-08-28 오후 12 19 10" src="https://github.com/Yu-Miri/Paper/assets/121469490/3e88741b-1bbe-4392-aef4-846c2c1e9b8a">


- 출력은 이러한 개체 중 하나 이상에 대해 적절한 마스크여야 함
- 이 요구조건은 언어 모델이 애매한 프롬프트에 대해 일관된 응답을 출력하기 기대하는 것과 유사함
- 자연스러운 사전 훈련 알고리즘과 프롬프트를 통해 다운스트림 분할(**downstream**은 내가 최종적으로 만들고자 하는 모델) 작업으로 제로샷 전송을 위한 일반적인 방법으로 이어지기 때문에 선택
- pretraining
  - 각 훈련 샘플에 대한 일련의 프롬프트(point, box, mask)를 시뮬레이션 하고 모델의 마스크 예측을 실제와 비교하는 자연스러운 사전 훈련 알고리즘을 상호적인 segment에 적용
    - 자동 주석을 포함하여 모호성을 수반하는 사용 사례에서 사전 훈련 모델이 효과적
- zeroshot transfer
  - pretraining을 통해 inference 시간에 모든 프롬프트에 적절히 대응할 수 있는 능력 갖춤
  - 최종 작업은 적절한 프롬프트를 엔지니어링하여 해결 가능
- related tasks
  - 상호적인 segmentation, 엣지 detection, 픽셀화, 객체 영역 생성, 전경 및 배경 분할, 유사한 분할, 인스턴스 분할, 전체적인 분할
  - 신속한 segmentation task 목표는 신속한 엔지니어링을 통해 광범위하게 기존 및 새로운 분할 작업에 적응하도록 모델 생성
  - 신속한 segmentation을 위해 훈련된 모델이 더 큰 시스템에서 구성요소로 작용하게 되어 추론 시간에 새롭고 다른 작업 수행이 가능함

<img width="828" alt="스크린샷 2023-08-28 오후 12 19 29" src="https://github.com/Yu-Miri/Paper/assets/121469490/71b7938d-8b57-4d72-b1e8-81a2eecc9f42">


- heavy 웨이트 이미지 인코더는 다양한 입력 prompt에서 효율적으로 의문을 가져 상각된 real time 속도로 객체 마스크를 생성할 수 있는 이미지 임베딩 출력
- 둘 이상의 애매한 prompt는 sam에서 유효한 마스크 여러개와 관련된 신뢰도 점수를 출력

Model

- 이미지 인코더, prompt 인코더, 빠른 마스크 디코더
- 이미지 인코더
  - 확장성과 사전훈련 방법에 영향을 받아 사전훈련된 ViT로 고해상도 입력을 처리
  - 이미지당 한 번 실행, 모델에 prompt 표시 전에 적용
- prompt 인코더
  - point, box, text와 고밀도 마스트의 두 가지 prompt set을 고려
  - CLIP의 대기중인 텍스트 인코더로 각 prompt의 유형에 대해 학습된 임베딩과 자유 형식 텍스트로 요약된 위치 인코딩으로 point와 box를 나타냄
  - prompt 마스크 층은 conv를 통해 임베딩되며, 이미지 임베딩과 함께 요소적으로 합산됨
- 마스크 디코더
  - 이미지 임베딩, 프롬프트 임베딩 및 출력 토큰을 마스크에 효율적으로 매핑
  - transformer 디코더 블락을 수정했으며, 동적 마스크 예측의 head 사용
  - 수정된 디코더 블락은 prompt self attention과 cross attention(prompt와 image embedding 상호적으로)에서 모든 임베딩 업데이트
  - 두 블락을 실행하고 나서 이미지 임베딩을 업샘플링하고 MLP는 출력 토큰을 동적 선형 분류기에 매핑하여 각 이미지 위치에서 마스크 전경 확률 계산
- 모호성 해결
  - 하나의 출력에서 애매한 prompt 제공되면 모델은 유효한 마스크를 여러개 평균화함 → 다중을 예측하도록 모델 수정
  - 단일 프롬프트에 대한 출력 마스크 : 가장 일반적인 경우에 3개의 마스크 출력이 충분함
  - 포함된 마스크에는 종종 최대 3개의 depth
  - mask의 순위를 매기기 위해 각 마스크에 대한 confidence score( 추정된 IOU ) 예측
- 효율성
  - 사전 계산된 이미지 임베딩이 주어지면 prompt 인코더와 mask 디코더는 CPU에서 최대 50ms 단위로 실행
  - 런타이미 성능을 통해 모델의 real time 상호적 prompt를 원활히 표시
- loss & training
  - focal loss & dice loss의 선형 조합으로 마스크 예측을 감독
  - 기하학적 프롬프트 혼합을 사용해서 prompt segmetation task를 위한 훈련
  - 마스크당 11라운드의 프롬프트를 무작위로 샘플링하여 상호적으로 시뮬레이션하고 SAM이 데이터 엔진에 원활히 통합되도록 함
- dataset engine
  - ViT-H : 더 많은 mask를 수집해 이미지 인코더의 크기 조정 → 6번 모델 재교육
    - 마스크 당 평균 annotation 시간은 34초에서 14초로 감소
    - 극한점이 있는 경계상자 레이블링보다는 느리지만 coco에 대한 mask annotation보다 6.5배 빠름
  - SAM이 개선되면서 이미지당 평균 마스크 수 20개에서 44개로 증가
  - 마스크의 다양성을 늘려서 SAM 모델 성능 향상, annotator가 덜 두드러지는 물체에 초점을 맞추기 위해 먼저 자신있는 마스크를 자동으로 감지하고, 이런 마스크로 채워진 이미지가 있는 annotator를 제시하고 추가적인 annotation을 하도록 했음
  - 주석 없는 객체 → 신뢰 가능한 마스크 detection을 위해 일반적은 object 범주를 사용해 모든 1단계 마스크에 대해 bounding box detection을 train
  - 새로 수집된 데이터에 대해서도 주기적으로 모델 재교육
  - 마스크당 평균 annotation 시간은 34초
  - 자동 마스크를 포함해서 44개에서 72개로 마스크 개수 증가
  - 자동 스테이지 : 마지막에 annotation은 완전히 자동화되었음, 이전 단계의 다양한 마스크를 포함해 모델을 크게 개선할 수 있는 충분한 마스크 수집하고, 애매한 경우에도 유효한 마스크를 예측할 수 있도록 모호성 인식 모델 개발
  - 32x32의 규칙적인 point grid로 모델에 자극을 주고, 각 점에 대해 유효한 객체에 해당되는 마스크 set 예측
  - 모호성 인식 모델을 통해 point가 부분이나 하위 부분에 있으면 하위부분과 부분 및 전체 개체를 반환
  - 모델의 IOU 예측 모듈은 신뢰가능한 마스크를 선택하는 데 사용, 안정적인 마스크만 식별하고 선택 ( 0.5 +- u에서 임계값 설정하면 유사한 마스크 생성의 경우 마스크가 안정적이라고 봄)
  - 신뢰할 수 있고, 안정적인 마스크를 선택하고 중복을 걸러내기 위해 Non-Maximum Suppression 적용
  - 더 작은 마스크 품질을 더욱 향상시키기 위해 여러 개 중첩 확대 이미지 크롭도 처리

Discussion

- Foundation Model : 사전 훈련된 모델은 머신러닝 초기부터 다운스트림 작업에 적용, 규모에 맞게 광범위한 데이터에 대해 train하고, 광범위한 다운스트림 작업에 적응할 수 있는 모델이라는 기반 모델
  - 기초 모델에서 self supervise train 역할을 강조하는 접근 방식의 한 측면을 대조
  - self-supervised 기술로 초기화, 대부분의 기능은 대규모 감독 훈련에서 나옴
- 구성
  - 대형 시스템에서 구성요소로 사용
  - SAM으로 이런 종류의 구성을 쉽게 만드는 것이 목표
  - SAM이 광범위한 분할 프롬프트에 대한 유효한 마스크 예측을 요구하여 달성하는 것
  - 웨어러블 기기에 의해 detect된 시선 point로 프롬프트 표시 (추가 교육 x)

논문을 읽고 난 후 고찰

- Dataset 구조 및 SAM 모델의 구조 파악이 난해하여 적용하지 못했다.
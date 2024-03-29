
# Style Retention Network for scene Text editing

논문을 읽는 이유

- K-예능, 드라마의 Text를 OCR으로 Text Detection & Recognition 후에 영어로 Translation해서 동일한 Style으로 Transfer 하는 프로젝트 목표와 일치하는 Task이다.
- 해당 논문은 원본 이미지의 배경과 텍스트 스타일을 유지하면서 원하는 텍스트로 대체할 수 있도록 한다.

### 용어 정리

|Image | Description |
| --- | --- |
| i_s | 원본 텍스트 이미지 |
| i_t | 타겟 텍스트 이미지 |
| tsk | 타겟 텍스트의 골격 이미지 |
| t_t | i_s 이미지의 텍스트 스타일을 타겟 텍스트에 적용시킨 이미지 |
| t_b | i_s 이미지에서 텍스트를 제거한 이미지 |
| mask_t | 이진화된 폰트 스타일 이미지 |
| t_f | 최종 변환 텍스트 이미지 |

### SRNet의 구조



Text conversion module

- Input = origin Img (Is) + Target text (Tt)
    - Is에서 전경스타일(글꽃, 색상, 기하학적 변형 등의 text style 포함) 추출 후 It로 전송
    - 대상 텍스트의 의미체계와 origin 이미지의 텍스트 스타일이 있는 이미지 Ot 출력
    - Ot = GT (It, Is ).
    - Is에서 텍스트 스타일 전송 후에 It에서 텍스트 골격을 유지해야 함 → 골격 가이드
    

Background inpainting module

- Input = origin Img
- 모든 텍스트 스트로크 픽셀 제거 후 적절한 텍스처로 채워진 배경 이미지 Ob 출력
- GB : Background Generate / DB : Background inpainting 판별자

Fushion module

- Input = It(target text image) + 배경 → It를 Of or Tf 연결해서 일관성 측정
- Decoder 업샘플링 단계에서 inpainting 모듈의 디코더 특징맵을 동일한 해상도로 해당 특징 맵에 연결 → 배경 세부 정보가 실질적으로 복원된 이미지 출력
- Gf 와 Of 사용해서 합성이미지를 생성
- adversarial loss(dicriminator), VGG-loss(dicriminator = 왜곡을 줄이고 사실적인 이미지를 만들기 위해 도입), perceptual loss, style loss
- perceptual loss Lper : 사전 훈련된 네트워크의 활성화맵 사이의 거리 측정을 정의해서 정답과 틀렸으면 벌점 부여
- Style loss Lstyle :  스타일의 차이 계산

Discriminator(DB)

- Input = Tb(배경의 정답값) ↔Ob랑 비교해서 True 여부 판단
- Is(소스 이미지)와 Ob(지워져 있는 출력 배경) 연결

Training and Inference

- training
    - **Tsk , Tt, Tb, Tf의 감독 하에 It, Is를 입력**
    - 텍스트 대체한 이미지 Ot 출력
    - 적대적 훈련
        - 배경 일관성을 위해 Is,Ob랑 Is,Tb가 DB에 공급
        - 정확한 결과를 위해 It,Of랑 It,Tf가 DF에 공급
- Inference
    - 텍스트 이미지 + 스타일 이미지 → 편집된 이미지의 지워진 결과 출력
    - 전체 이미지 = bbox annotation에 따라 타켓 패치를 잘라 네트워크에 공급 후 결과를 원래 위치에 붙여 넣어서 시각화
- Dataset
    - Synthetic Data
        - 글꼴, 색상, 변형 파라미터 무작위 선택해서 스타일이 지정된 텍스트 생성하고 배경 이미지에 렌더링
        - 이미지 골격화해서 해당 배경, 전경 텍스트 및 텍스트 골격을 지상 실측으로 얻는다
        - 텍스트 이미지 높이 64 조정, 동일한 종횡비 유지
        - 훈련 50000개, 테스트 500개 이미지 포함
    - Real-world dataset
        - **ICDAR 2013 → 국제문서 분석 및 인식회의에서 조직한 자연장면 텍스트 데이터셋**
        - 229개 훈련사진, 233개 테스트 사진 포함해 자연장면에서 가로 영어 텍스트의 감지 및 인식에 중점( 자세한 label과 가로 직사각형 annotation)
        - 모든 이미지에는 하나 이상의 텍스트 상자 존재, bbox에 따라 텍스트 영역을 자르고
        - 합성 데이터에 대해서만 모델 교육, 실제 데이터는 테스트로 사용
        

- Detail
    - Adam optimizer → 훈련단계 : 안정될때까지 beta1 = 0.5, beta2 = 0.999 채택
    - Learning rate = 학습률 초기에 2x10-4 설정, 30 에폭 후에 2x10-6으로 감소
    - **α = θ2 = 1, β = θ1 = 10, θ3 = 500을 선택하여 back propagation에서 각 부분의 loss gradient norm을 닫는다.**
    - 스펙트럼 정규화를 제너레이터와 디스크리미네이터에 적용
    - 배치 정규화는 제너레이터에만 적용
    - 배치 8, 입력 이미지는 비율 변경 안 하고 w x 64로 크기 조정
    - 이미지 너비는 평균 너비로 조정
    - 테스트할 때 원하는 결과를 얻기 위해 가변적 너비의 이미지를 입력할 수 있음
    - 모델 훈련하는데 8시간 → 단일 **NVIDIA TITAN Xp그래픽카드**
    
- 평가지표
    - MSE
    - 노이즈에 대한 피크 신호 비율을 계산하는 PSNR
    - 두 이미지 간 평균 구조적 유사성 지수를 계산하는 SSIM
        - Synthetic data에 대해서만 적용(실제 데이터 세트는 쌍을 이룬 데이터 x)
        - 낮은 MSE, 높은 SSIM, 높은 PSNR 결과가 Ground True와 유사하다
        - 실제 데이터 세트에서 생성된 결과는 recognition 정확도 계산해서 퀄리티 평가
        - 네트워크에 입력으로 넣는 이미지가 잘린 이미지이기 때문에 잘린 영역에서만 해당 메트릭 계산
        - backbone이 VGG와 유사한 모델로 대체된 어텐션 기반의 텍스트 인식 모델
    
- Ablation study
    - skeleton guided module
        - 골격 모듈 제거 후 훈련동안에 텍스트 골격 감독 정보가 부족해서 전송 후에 텍스트 구조가 부분적으로 휘어져 파손되기 쉽다
        - 이에 비해 전체 모듈 방식은 원본 텍스트 변형을 올바르게 학습

-pix2pix 네트워크가 여러 스타일 변환을 구현하도록 하기 위해 스타일 이미지와 대상 텍스트를 입력으로 연결

-단어수준의 텍스트를 삭제하는 것으로 목표

지워야 하는 그림에서 자유롭게 텍스트 영역 선택

한계점

텍스트 구조가 매우 복잡하거나 글꼴 모양이 특이한 경우

1 : 텍스트 자체에 그림자가 져있는 경우 원본 텍스트의 그림자가 여전히 이미지에 출력

2 : 복잡한 공간 구조를 가지 텍스트 스타일 추출 x, 배경 삭제 결과도 차선책

3 : 텍스트를 둘러싼 경계

→ 더 많은 글꼴 효과로 훈련 세트 보강!!

논문을 읽고난 후 고찰

논문의 구조를 파악하여 커스텀 데이터셋을 제작해 학습하였지만 한계점으로 인해 결과가 좋지 않았다.

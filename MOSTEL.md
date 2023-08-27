
# **Exploring Stroke-Level Modifications for Scene Text Editing**

### 용어 설명

| Image | Description |
| --- | --- |
| i_s | 원본 텍스트 이미지 |
| i_t | 타겟 텍스트 이미지 |
| guide_s | 이진화된 Masking 원본 텍스트 이미지 |
| guide_t | 이진화된 Masking 타겟 텍스트 이미지 |
| o_s | i_s 이미지에서 텍스트를 제거한 이미지 |
| o_t | 타겟 텍스트에 i_s의 폰트 스타일을 적용시켜 생성한 이미지 |

| Module | Description |
| --- | --- |
| BRM[Background Reconstruction Module] | 배경에서 유지해야 하는 영역을 분할 및 보존하여 텍스트를 지워주는 모듈 |
| TMM[Text Modification Module] | 판독 불가능한 서체와 스타일을 추출하여 원본 이미지의 스타일을 학습하고 번역된 텍스트에 스타일을 적용하는 모듈 |
| TPS[Thin Plate Splines] | 텍스트 윤곽을 파악할 수 있는 Anchor Point를 얻어 텍스트의 방향에 대한 정보를 제공하는 모듈 |
| PSP[Pyramid Scene Parsing] | • 이미지의 크기를 여러 단위로 조정하여 하나의 입력 이미지에 대해서 다양한 크기를 학습하는 모듈 |
| SLM[Stroke Level Modification] | 배경의 무결성을 위해 픽셀의 변화를 최소화하는 모듈 |

----
논문을 읽는 이유

    - K-예능, 드라마의 Text를 OCR으로 Text Detection & Recognition 후에 영어로 Translation해서 동일한 Style으로 Transfer 하는 프로젝트 목표와 일치하는 Task이다.
    - 해당 논문은 원본 이미지의 배경과 텍스트 스타일을 유지하면서 원하는 텍스트로 대체할 수 있도록 한다.

## Abstract

Goal : 원본 이미지의 배경과 텍스트 스타일 유지하면서 원하는 텍스트로 대체한다.

1. Implicit decoupling structure.
    1. 전체 이미지 편집 → 배경과 텍스트 영역의 Transfer Rule을 동시에 학습해야 한다.
2. Domain gap.
    1. 편집된 실제 장면 텍스트 부족 → 합성 데이터셋에서만 훈련이 잘 될 수 있다.

---

## Introduction

1. 편집 영역을 명시적으로 나타내는 Stroke Guidance Map
    1. Image 수준에서 배경 부분과 텍스트 부분의 픽셀을 직접 수정하여 복잡한 배경에 대해서 잘 학습할 수 있도록 하여 모델이 텍스트 영역의 Editing Rule에 집중할 수 있도록 한다.
2. 레이블이 지정되어 있는 합성 데이터셋, 레이블링이 되지 않은 실제 장면 텍스트 데이터셋을 사용한 Semi supervised Hybrid Learning
    1. Semi supervised  Learning
        1. 레이블이 없는 데이터로부터 추가적인 정보를 추출하고, 데이터의 활용도를 높여 모델의 성능을 향상시킨다.
        2. 훈련 데이터 셋의 확장이 가능하며, 모델이 다양한 패턴, 변동성을 학습하여 일반화 성능을 향상시킨다.
    2. Hybrid Learning
        1. 각기 다른 기계 학습 기법의 장점을 조합하여 모델의 성능을 향상시킨다.
        2. 모델이 다양한 관점에서 데이터를 분석하고 학습하여 일반화 능력을 향상시킨다.
        3. 다양한 상황에 적응할 수 있는 유연성이 생기며, 알고리즘을 조합하여 문제의 특성과 데이터의 특징에 맞게 모델을 조정할 수 있어 다양한 도메인에서 일반화한다.
        4. 최적화에 도움을 주며, 서로 다른 알고리즘이 상호 보완적인 경우에 효과적이다.

### SRNet과 MOSTEL의 차이점
<img width="553" alt="스크린샷 2023-08-27 오후 3 46 45" src="https://github.com/Yu-Miri/Paper/assets/121469490/9dbba2da-5d54-4d9f-a188-58ed2605e489">


- SRNet method
    - 동시에 Text Conversion으로부터 텍스트 스타일을 뽑고 Background Reconstruction에서 배경 추출해서 두 결과값을 fushion하여 출력한다.
    - 편집 과정에서 변경되는 픽셀(varying region)에서 배경 영역에 변화를 일으킨다.
    - 배경의 텍스처를 보존하기 어렵고, 동시에 작업이 이루어지는 결과로 Bias가 발생하여 성능이 떨어진다.
- MOSTEL method
    - Background Reconstruction 모듈에서는 유지해야하는 영역을 분할 및 보존하며, Text Modification 모듈에서는 판독이 불가능한 서체와 스타일을 추출한다.
    - 편집 부분을 나타내는 Editing Guidance를 통해 변하지 않는 배경 영역 보존과 텍스트 영역의 스타일을 수정하는 것을 명시적으로 분할하고, 필터링하여 Stroke-level modification으로 이어진다.
    - 두 모듈에 Stroke Level Modification이 구현되는데, 이를 통해서 텍스트 영역의 편집 영역을 명시적으로 나타내고, 배경 영역의 변화를 최소화하도록 하여 안정적인 출력에 기여한다.
    - 불필요한 중간 결과 제거를 위해 erase-write 패러다임을 적용하여 Editing Train의 난이도를 감소시키고, 생성된 텍스트 스타일과 원본의 일관성을 보장한다.
    - Image 수준에서는 모든 픽셀을 수정하여 Editing train의 난이도를 높인다.
    - 생성된 텍스트 이미지를 명확하고 읽기 쉽게 하기 위해 장면 텍스트 recognizer가 훈련 단계에서 사용되며, Recognizer를 채택하는 것이 모든 측정 기준에 유익하다.
    - 성능 평가 데이터셋 : Tamper-Syn2k, Tamper-Scene
  
<img width="556" alt="스크린샷 2023-08-27 오후 3 47 13" src="https://github.com/Yu-Miri/Paper/assets/121469490/7f1d0fd5-82fe-45c4-8ad0-99e85bdf8a48">

---

## Realated Work

Text Image Synthesis

- DNN 모델 훈련을 위한 trick으로, 텍스트 탐지 및 인식의 정확도 향상을 위해 합성 텍스트 이미지를 생성한다.
- 단어를 생성해서 배경 이미지의 적합한 영역에 텍스트를 삽입한다.

Style Transfer

- 참조 이미지에서 다른 대상 이미지로 시각적인 텍스트 스타일을 전송한다.
- 입력 공간에 내장해서 디코딩하여 원하는 이미지를 생성하는 인코더-디코더 아키텍처 사용
- 교환되지 않은 도메인 데이터에 대한 매핑 관계 일반화를 위해 주기적으로 일관성 손실을 도입한다.
- 텍스트 효과의 거리 기반 필수 특성을 분석, 모델링하고 활용하여 fushion한다.

Scene Text Editing

- 글꼴 적응 신경이 설계된 것이다.
- Background Inpainting과 Text Transfer, Fushion을 통해 단일 문자의 편집을 진행하는 SRNet에서는 문자 수준의 수정에 있어서 길이가 변경된 단어 대체가 불가능하여 실제 응용 분야에서 성능이 제한되었다.
- SRNet에서 확장시켜 공간 변환을 텍스트 스타일에서 분리해 텍스트 변환 모듈의 학습 난이도를 낮추는 TPS 모듈 도입
- Stroke level modification으로 가독성이 높은 텍스트 이미지를 생성한다.
- 레이블이 지정된 합성 데이터셋과 실제 데이터셋에서 준지도 학습 방식으로 훈련된다.
  
<img width="576" alt="스크린샷 2023-08-27 오후 3 47 34" src="https://github.com/Yu-Miri/Paper/assets/121469490/0bf485a5-9409-4ebf-b503-5cd29f11c2bf">

Methodology

- MOSTEL : 모델이 identity mapping network로 전락하는 것을 피하고, 배경과 텍스트 분리하기 위한 방법론
- Input : 원본 이미지(I_s), 타겟 텍스트 이미지(I_t)
- output : 배경 인페인팅 이미지(O_s), O_s와 텍스트 스타일이 Fushion된 이미지(O_t)

Background Reconstruction Module(BRM)

- 3개의 Downsampling Layer 인코더 - 3개의 Upsampling Layer 디코더로 구성된다.
- 준지도 하이브리드 학습을 통해 훈련 데이터셋과 실제 데이터셋의 도메인 격차를 줄인다.
- 학습되는 Image는 Background Inpainting Image를 생성하기 위해 삭제된 후에 모방한 스타일의 동일한 텍스트가 재구성된 Background에 쓰여진다.
- Pyramid Scene Parsing(PSP)
    - 하나의 Input Image에 대해서 모든 픽셀이 직접 수정되면 배경 영역에 불필요한 변화가 발생되므로, 모든 픽셀을 수정하지 않고, 다양한 배경에 대해서 다중 스케일을 학습할 수 있도록 한다.
    - size를 [1, 2, 3, 6]단위로 바꿔가면서 학습하여 다중 스케일 특징을 향상시킨다.
- Stroke Level Modification(SLM)
    - Editing Guidance Map의 출력이 Stroke 영역의 분할 작업에 해당된다.
    - 배경의 무결성 최소화를 위한 모듈으로, 픽셀의 변화를 최소화하고 명시적으로 Editing Point를 guide한다.
- Editing Guidance(EG)
    - 배경 인페인팅 이미지(O_s) 생성과 최종 Fushion된 이미지(O_t) 생성할 때 EG를 통해 예측한다.
    - 변하지 않는 배경 영역 텍스트 영역의 스타일 영역을 명시적으로 나타낸다.
    - 텍스트가 있는 부분의 점들을 저장해서 위치정보 저장
- 예측된 O_s와 Guide_s가 SLM에 입력되어 Background Inpainting Image(O_s)가 출력된다.
- SLM으로부터 안정적인 편집 이미지 생성 → 배경 영역은 원본 이미지 Is에서 상속되어 원본 이미지 불변 영역과의 일관성 극대화
- 변하지 않는 영역 및 다양한 영역을 구별하지 않고 텍스트 스타일 영역의 Editing Rule에만 집중할 수 있어 작업을 단순화한다.

Text Modification Module(TMM) : Pre-Transformation & Modification Module

- Pre-transformation
    - 원본 이미지(I_s), 타겟 텍스트 이미지(I_t)의 특수한 부분적 변환을 적용시켜 전처리하고, Background Feature와 Text Style을 혼합하여 Edited Image 생성
    - Swap Text : Feature Extraction과 FC Layer를 통해 텍스트 윤곽에 위치한 anchor point(I_s의 제어 포인트)를 얻는다.
    - TPS Module
        - anchor point를 통한 텍스트 윤곽을 파악하고, I_s의 기하학적 특성에 따라 I_s의 동일한 텍스트 방향으로 I_t의 방향을 조정하는 데 사용한다.
        - 다른 Text Style에서 부분적으로 속성을 분리하여 Style Transfer의 난이도를 향상시킨다.
        - 여러 텍스트 스타일으로부터 공간 속성을 분리해서 Text Style의 올바른 정보를 전송할 수 있도록 한다.
    - I_s로부터 복잡한 Background Texture는 모델이 Text Style을 학습하는데 방해되기 때문에 BMM에서 예측한 Guide_s를 통해 Background 필터링하여 noise를 제거한다.
    - 정확하게 스타일을 생성하기 위해 **Style Augmentation**을 진행한다.
        - Random Rotation(-15도 ~ 15도까지의 각도), Random Filp(0.5의 확률)의 데이터 증강 방식은 **여러 텍스트 스타일 학습에 영향을 주지 않는다.**

- Modification Module
    - BRM과 동일한 인코더-디코더 구조
    - 디코더에서 Target Text Feature는 BRM의 Upsampling 기능과 더해져 Edited Image로 예측한 O_t와 Guide_t를 생성한다.
    - Gradient Blocked : Semi-Supervised Learning을 채택하는 경우에 네트워크가 다양한 Text Style에 대해서 변환되지 않고 출력될 mapping 가능성을 방지하기 위해 적용한다.[skip connection과 같은 기능]
    - BRM과 동일하게 Editing Guidance가 도입된다.
    - 네트워크가 Text 영역에 집중하고 변하지 않는 배경으로부터 주의를 주기 위해 Inference 단계에서만 SLM을 적용한다.
    - Text Image를 잘 읽게하기 위해 Pretrained Recognizer를 도입한다.
    - G는 그램 행렬, 밸런스 팩터는 µv1과 µv2는 각각 1과 500으로 설정한다
- Loss
  
    <img width="529" alt="스크린샷 2023-08-27 오후 3 48 27" src="https://github.com/Yu-Miri/Paper/assets/121469490/bee61928-3dd9-4027-a46c-42fc879390ee">
    
    - 예측된 Background Inpainting Image(O_s^)와 예측된 Fushion Image(O_t)는 **GAN loss(생성된 입력 텐서의 로그값 평균을 계산하고 음수를 취하여 Loss 계산)하므로** 와 **L2 loss(정답값 입력텐서와 생성한 입력 텐서 간 차이의 제곱을 계산하여 평균을 구하는 L2 loss)**로 최적화한다.
    - Editing Guidance의 Guide_s, Guide_t Image는 **Dice loss(정답값 입력 텐서와 생성한 입력 텐서 간의 교집합을 계산하여 평균을 구하고 1에서 뺀 값으로 정의되므로, Dice Loss는 정답과 생성한 타깃 간의 유사성이 높을 수록 손실이 작아진다. 즉, Loss를 낮추는 방향으로 학습)과 생성한** 채택한다.
    - 사실적인 Image Generator를 위해 **VGG loss(MSE 평균제곱의 오차를 계산하는 VGG Loss는 Perceptual Loss + Style Loss : 정답값 입력 텐서와 생성한 입력 텐서 사이의 절댓값 차이를 계산하여 계산된 차이의 평균을 구하는 L1 loss, 절댓값의 평균이므로 낮추는 방향으로 학습)를** 채택한다.
    - Recognizer Loss : **Cross Entropy Loss(정답값 입력 텐서와 생성한 입력 텐서의 로그 확률을 계산하여 음수를 취한 후 평균을 계산)**를 사용한다.
    - Discriminator Loss : Binary Cross Entropy Loss의 음수를 사용하여 손실을 계산
    - Generator Loss : GANLoss+L2Loss+DiceLoss

- Real Data
    - 심하게 왜곡되고 인식할 수 없는 이미지를 필터링하여 총 7,725개의 이미지를 선택하여 Tamper-Scene을 구성
    
- Semi Supervised Hybrid Learning
    - Real scene text image에 필요한 label이 없기 때문에 텍스트를 자체로 변환하는 패러다임을 채택한다.
    - 텍스트 없는 배경을 생성하기 위해 원본 이미지(I_s)가 BRM에서 먼저 인페인팅된다.
    - 배경이 없는 Text Style과 Augmented Style을 사용하여 학습 난이도를 높였다.
    - TMM과 BRM의 Output이 Fushion될 때 BRM 디코더 부분의 Gradient를 block하여 인코더-디코더가 원본 이미지의 Text로 복원하지 못하도록 한다.

- Experiment
    - Synthetic Data
        - MOSTEL Supervised Learning을 위해 15만 개의 Label이 지정된 이미지를 생성한다.
        - 평가를 위해 2000개의 Image Dataset인 Tamper-Syn2k를 구성한다.
        - Image는 폰트, 크기, 색상, 부분 변환, 배경 이미지와 같이 동일한 스타일과 다양한 텍스트로 Rendering 된다.
        - Augmentation : Random Rotation, Curve, Perceptive 적용
    - Real Data
        - 34,625개의 이미지를 포함한 Real Scene Text Image Dataset인 MLT-2017을 사용하여 34,625개의 이미지를 포함한 실제 장면 텍스트 이미지를 사용한다.
        - Recognizer Loss 계산을 위해 Text annotation이 필요하다.
        - 평가를 위해 여러 Scene Text Dataset을 조합하여 심하게 왜곡되고 인식이 불가능한 이미지를 필터링하여 7,725개의 Dataset인 Tamper-Scene을 구성한다.
    - Implementation Details
        - Input Image Size : 256x64
        - Optimizer : b1 = 0.9, b2 = 0.999인 Adam optimizer
        - Learning Rate : 5x10^-5
        - 14개 레이블이 지정된 Synthetic Image와 2개의 Annotation되지 않은 Real Scene Image로 구성하여 Batch Size는 16, Epoch은 300,000으로 훈련하였으며, Pytorch 구현, NVIDIA 2080Ti GPU에서 훈련되었다.
    - Evaluation Metrics
        - MSE, L2 distance
        - PSNR : 이미지나 비디오의 최대 신호 잡음의 비율을 측정
        - SSIM : 이미지 품질의 평균적인 구조적 유사성을 측정
        - FID : Incept V3에 의해 추출된 Feature 사이의 distance
        - PSNR, SSIM, MSE, FID 높을수록 성능이 우수하다.
        - Real Scene Text Image는 Pretrained Model 3과 공식 Text Recognizer 알고리즘을 사용하여 인식 정확도를 채택한다.

MOSTEL의 기능 비교

<img width="566" alt="스크린샷 2023-08-27 오후 3 49 07" src="https://github.com/Yu-Miri/Paper/assets/121469490/e10d3160-0eef-4ff7-9a25-c2529264ff13">

Source Image : 원본 이미지

SRNet : SRNet의 Output

w/o : Mostel에 추가된 기능들을 각각 제거했을 때의 Output

MOSTEL : Mostel의 Output

Conclusion

- MOSTEL(MOdifying Scene Text image at strokE Level) : End-to-End로 학습 가능한 Framework 제안
- Scene Text Editing에 대한 한계적인 성능을 내면서도, Synthetic Data, Real Data 간의 도메인 사이의 문제를 해결하기 위한 기능들과 Editing Guidance가 제안된다.
- Background Filtering, Style Augmentation, Gradient Block 등을 적용하고, Semi-Superviesed Hybrid Learning을 도입한다.

논문을 읽고 난 후 고찰
- Background Module과 Text Style Transfer Module을 나누지 않고 동시에 작업하는 SRNet과 비교하여 추론한 결과 MOSTEL은 배경을 최대한 유지한 채 바꿀 Text에 대해서 style을 적용할 수 있었다. 이는 논문에서 제공된 성능 개선 전략들이 부합하다는 것을 나타내며, 영어로 학습된 모델이었기에 한국어로 Fine Tuning하기 위해 해당 논문의 github에서 공유되어 있는 weight로 한국어 Source Image를 추가로 학습시켰다.

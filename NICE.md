## f**low 모델을 이용한 MNIST inpainting**

### 최대 가능도 추정(Maximum Likelihood Estimation, MLE) 방법

- “주어진 데이터에 대해 확률모델 p(X)의 파라미터를 조정하여 데이터가 확률 분포에서 나올 확률이 최대화 될 가능성이 높은 모수를 찾는 방법”
- ex) 정규 분포의 가능도 함수 : 주어진 평균과 분산을 가진 정규 분포에서 주어진 데이터가 나올 확률 → 모델의 파라미터로 평균과 분산을 가지며, MLE를 사용하여 주어진 데이터에 가장 적합한 평균과 분산을 추정

**Likelihood vs. Probability**

- **가능도(Likelihood)** : 어떤 시행을 충분히 수행한 후에 결과(sample)를 토대로 경우의 수의 가능성을 도출하는 것
- 충분히 수행하더라도 추론(inference)이므로 가능성의 합이 1이 되지 않을 수 있음
    - 주어진 데이터에 대한 관점으로부터 접근하여 데이터가 어떤 파라미터 값을 가질 때 어떤 확률 분포에서 생성되었는지 고려
- **확률(Probability)** : 어떤 시행에서 특정 결과(sample)가 나올 가능성이며, 시행 전에 모든 경우의 수의 가능성이 정해져 있어 총합이 1(100%)
    - 사건 자체에 대한 관점으로부터 접근하여 특정 사건이 어떤 확률 분포에서 발생하는지 고려

→ 파라미터에 대한 특정 모델, 분포의 value 값을 찾아 가능도를 확인한다면, 가능도 중에서도 가장 높은 가능도가 존재할 것이다.

MLE : 주어진 데이터를 제일 잘 설명하는 모델(확률 분포의 파라미터)을 찾는 것

<img width="273" alt="스크린샷 2023-10-13 오후 1 52 28" src="https://github.com/Yu-Miri/Paper/assets/121469490/bca268a5-7ae9-48cb-952d-098c7c278d6a">

최대 가능도 추정 문제와 같이 Original Image와 Corrupted Image를 통해 Inpainting 문제 해결

## NICE(NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION)

### Abstract

- 비선형 독립 성분 추정 : 복잡한 고차원 밀도를 모델링하기 위한 딥러닝 프레임워크
- 데이터의 비선형성 결정론적 변환 : 변환된 데이터가 요인화된 분포를 따르도록 잠재 공간(데이터의 특징이나 표현을 나타내는 공간)에 매핑하여 독립적인 잠재 변수를 생성하도록 학습
- 확률적 생성 모델으로서 신경망 구성 블록을 조합해 데이터의 비선형성을 효과적으로 모델링하여 복잡한 데이터 분포를 표현

### Introduction

- 비지도 학습의 핵심은 ‘알려지지 않은 구조를 복잡한 데이터를 어떻게 포착하는가’이며, 딥러닝은 가장 중요한 변동 요인들을 포착할 수 있는 데이터의 좋은 표현에 대한 학습에 의존한다.
- What is a good representation?
    - 데이터의 분포가 모형화하기 쉬운 것

### LEARNING BIJECTIVE TRANSFORMATIONS OF CONTINUOUS PROBABLITIES

- 각각 공간 X에 존재하는 N개의 유한 데이터셋 D에 대한 매개변수 밀도 제품군에서 확률 밀도를 학습하는 문제를 고려
- 변수의 변화공식을 사용하여 데이터 분포의 거의 모든 곳에서 비선형 변환 f를 최대 가능성을 통해 더 단순한 분포로 학습

    <img width="323" alt="스크린샷 2023-10-13 오후 1 53 01" src="https://github.com/Yu-Miri/Paper/assets/121469490/b7cd607d-397a-43d0-8e54-bc07a983bed7">

- 가역적 전처리 : 단순히 데이터를 수축함으로써 임의로 가능성을 증가

### ARCHITECTURE

**TRIANGULAR STRUCTURE**

- 모델의 아키텍처는 Jacobian 행렬식이 다루기 쉬우며, 계산이 간단한 역투영 계열, 순방향(인코더) 및 역방향(디코더)을 얻기 위해 중요
- 계층화 or 합성 변환을 사용하여 순방향 및 역방향 계산을 계층의 계산 구성
- 아핀 변환 고려 : 컴퓨터 그래픽스, 이미지 처리, 컴퓨터 비전 및 컴퓨터 그래픽스 디자인과 같은 다양한 분야에서 사용되며, 기하하적으로 객체를 이동, 회전, 크기 조정, 왜곡을 통해 객체의 조작이 가능하여 이미지나 그래픽을 원하는 형태로 변형시킬 수 있다. 또한, 객체에 대한 정보를 얻을 때 크기와 방향의 변동이 존재할 수 있어 이를 보정하기 위해 객체 추적 및 패턴 매칭에 사용되고, 매칭에 더하여 컴퓨터 비전 및 패턴 인식에서 객체의 크기와 방향을 보정하는 데 사용된다.

**COUPLING LAYER**

- 삼각형 Jacobian 행렬식을 갖는 사영 변환에 대한 Layer
- 입력 데이터를 결합하거나 변환하고, 이를 통해 생성된 출력 데이터를 역변환하여 원래 입력 데이터를 추출하는 데 사용되며, 신경망 및 기계학습 모델의 구조에서 유용하다.
- 확률적 변환을 수행하는 확률적 생성 모델에서 사용된다.
- **General coupling layer**
    - 두 부분 데이터 세트를 결합하고, 결과를 변환하여 출력 데이터를 생성한다.
    - 역변환을 통해 입력 데이터를 다시 추출할 수 있어 데이터 분포의 특징 학습과 생성에 유용하다.
    
- **Additive Coupling Layer**
    - 간단한 형태로 결합 법칙을 통해 두 입력 데이터를 간단히 더하여 결합한다.
    - 이전 결합층과 동일하게 결합을 더하여 비교적 간단하게 역변환을 수행하여 원래 입력 데이터를 추출한다.

- **Combining Coupling Layer**
    - 여러 개의 결합층을 조합하여 복잡한 변환을 얻는다.
    - 4개의 결합층을 사용하여 입력 데이터의 서로 다른 부분을 교차로 변환하는 데, 이를 통해 모든 차원이 서로 영향을 미쳐 더 복잡한 변환을 구현한다.
    - Jacobian 행렬을 고려하여 서로 영향을 미치도록 한다.

**ALLOWING RESCALING**

- 데이터 분포의 특징을 조절하기 위해 스케일링 행렬 사용
- Additive Coupling Layer는 입력 데이터를 결합할 때 단위 Jacobian 행렬식을 가짐으로써 데이터 분포의 부피를 보존하면서 데이터를 변환한다.
- Jacobian 행렬식을 유지하면서 데이터 분포의 특정 측면을 조절하기 위해 최상위 계층에 대각선 스케일링 행렬 S를 도입하여 출력 데이터의 각 차원에 대해 스케일을 적용한다.
- 주성분 분석의 고유 스펙트럼(데이터의 핵심 특징을 나타내는)과 유사한 스케일링 행렬 S로부터 데이터 일부 차원의 가중치 조절이 가능하여 데이터 분포의 특정 측면을 강조하거나 억제하는 데 사용된다.
    <img width="310" alt="스크린샷 2023-10-13 오후 1 53 25" src="https://github.com/Yu-Miri/Paper/assets/121469490/0d46c5fe-9025-44b8-bda8-d4319fc2a380">

- 특정 차원에 대해 큰 값을 가진다면 덜 중요한 차원, 작은 값을 가진다면 중요한 차원을 의미하여 이를 통해 데이터 분포의 특징을 조절한다.

**PRIOR DISTRIBUTION**

- 확률적인 모델에서 모델 파라미터의 사전 분포를 나타내며, 이는 학습 과정에서 파라미터 조정과 데이터를 적합하게 만들기 위한 정보를 제공
- 이전 분포는 일반적으로 특정한 확률 분포 함수인 가우스 분포와 로지스틱 분포를 사용한다.
- **가우시안 분포** : 연속적인 데이터에 대한 사전 분포로 사용되며, 데이터가 평균 주변에서 집중된 경우에 사용된다.
- **로지스틱 분포** : 이진 분류 문제에서 사용되며, 극단적인 값으로부터 멀리 떨어진 값을 가질 수 있다. 제공된 [starter.py] 파일에서는 로지스틱 분포를 선택한다.
- 이전 분포는 모델 학습 과정에서 관측 데이터와 결합하여 사후 분포를 계산하는 데 사용되며, 사후 분포는 모델의 최종 파라미터 값에 대한 더 정확한 추정을 제공하여 예측 및 모델 일반화 능력에 영향을 미친다.

### RELATED METHODS

- 변형 자동 인코더(VAE)로부터 효율적 근사 추론을 하며, 확률적 인코더와 디코더를 사용하고, 훈련 중 reconstruction 항을 포함한다. 이를 통해, 디코더는 인코더를 대략 반전시킨다.
- VAE는 데이터의 로그 가능성에 대한 변동 하한을 최대화하는 방식으로 훈련되어 빠른 샘플링 기술을 제공하는데, 하한 기준을 사용하면 불필요한 노이즈를 추가하여 부자연스러운 샘플 생성이 될 수 있다.
- 결정론적 디코더를 사용하여 낮은 수준의 노이즈를 제거하고, 생성된 샘플을 자연스럽게 만들어 문제를 완화시킨다.
- NICE는 VAE 기반의 생성모델이지만, 데이터를 변환하고 다시 역변환할 수 있는 구조로 모델링한다.
- 즉, 입력 데이터의 중요한 정보를 잠재 공간에 인코딩하고 디코딩하여 재구성하는 방법으로 학습한다.
- 데이터의 Log Likelihood 및 Entropy 항을 활용하는데, Log Likelihood를 극대화하여 데이터 분포의 다양성을 고려하면서 학습하고, 스케일링 행렬을 추가하여 데이터 분포의 측정 측면을 조절한다.

### EXPERIMENTS

**LOG-LIKELIHOOD AND GENERATION**

<img width="393" alt="스크린샷 2023-10-13 오후 1 54 06" src="https://github.com/Yu-Miri/Paper/assets/121469490/98f03c79-ca97-4edc-b4f7-b5819fb98c9b">

그림 3: 아키텍처 및 결과로, MNIST 데이터셋이므로 이전 분포는 Logistic 분포를 선택한다.

**INPAINTING**

<img width="414" alt="스크린샷 2023-10-13 오후 1 54 19" src="https://github.com/Yu-Miri/Paper/assets/121469490/496c459d-5fe6-4b76-8d2b-bf321b52678e">

그림 6: MNIST의 인페인팅

- 이미지의 복원 부분을 선택하여 masking하고, masking된 부분에 대해서 Log Likelihood를 최대화한다.
- masking된 부분의 값을 복원하기 위해 확률적으로 업데이트를 수행하는데, Log Likelihood의 기울기를 계산하고, 가우스 노이즈를 추가하여 masking된 부분의 값을 복원한다.

### CONCLUSION

- 이미지로 된 훈련 데이터를 데루는 고도의 비선형적인 변환을 학습하고, Log Likelihood 최대화를 결합하여 데이터를 효과적으로 분석할 수 있는 새로운 아키텍쳐와 프레임워크를 제시한다.
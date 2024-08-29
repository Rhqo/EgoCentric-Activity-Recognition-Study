# EgoCentric Activity Recognition Study

### [Survey Git](https://github.com/EgocentricVision/EgocentricVision?tab=readme-ov-file#devices)

## EGO4D (Challenge)

https://ego4d-data.org/

## Motivation

[MISAR: A Multimodal Instructional System with Augmented Reality](https://arxiv.org/abs/2310.11699)

# 0. Video Processing (UniFormer, UniFormerV2)

[https://github.com/Sense-X/UniFormer](https://github.com/Sense-X/UniFormer)

[https://github.com/OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)

![Screenshot from 2024-08-27 11-03-47.png](./Images/0.png)

$$
X = DPE(X_{in}) + X_{in},\\
Y = MHRA(Norm(X)) + X,\\
Z = FFN(Norm(Y)) + Y
$$

### DPE (Dynamic Position Embedding)

위치 정보를 인코딩하는 데 사용되는 방법

기존 모델들은 상대적 위치 임베딩을 사용했지만, 입력 크기에 대해 interpolation이 필요하거나, self-attention 매커니즘이 변경될 때 성능이 떨어진다.

이를 위해 DPE는 deepwise convolution과 zero padding을 사용하여 입력 형태에 따라 동적으로 임베딩을 조정할 수 있도록 만들었다.

Deepwise convolution은 경량화 되어있어 계산 효율성과 정확성 사이의 균형을 유지하는 데 도움이 되며, zero padding은 이웃한 토큰들 간의 관계를 점진적으로 고려함으로써 절대적 위치 정보를 잘 포착하게 해준다.

시각적 데이터의 공간적 및 시간적 순서를 유지하는 능력을 향상시켜, 특히 비디오 분류와 같은 작업에서 더 나은 표현 학습을 가능하게 한다.

### MHRA (Multi-Head Relation Aggregator)

CNN과 self-attention을 결합하여 효율적인 토큰 관계 학습을 수행한다.

여러 개의 head를 통해 각기 다른 유형의 관계를 학습하며, 각 head는 Relation Aggregator (RA)를 사용하여 특정한 관계를 계산한다.

- Local MHRA : 작은 범위 내의 토큰 관계를 학습, CNN의 합성곱 필터와 유사. 인접한 토큰들이 비슷한 시각적 내용을 가지는 경향이 있어 계산의 효율성이 높다.
- Global MHRA : 깊은 층에서는 긴 거리의 의존성을 학습한다. Self-attention (transformer) 매커니즘과 유사. Video 인식 작업에서 공간과 시간적 차원을 모두 고려하도록 조정되었다.

RA는 token context encoding과 token affinity learning으로 구성된다.

> $U \in R^{C\times C}$ 는 N개의 헤드를 통합하기 위한 학습 가능한 매개변수 행렬
> 
> 
> $V_n(X) \in R^{L \times \frac{C}{N}}$ 는 token context encoding
> 
> $A_n \in R^{L \times L}$ 는 token affinity
> 

# 1. Video Language Pretraining (EgoT2, LaViLa)

## 1.1 [LaViLa](https://github.com/facebookresearch/LaViLa) (22.12)

![Screenshot from 2024-08-27 14-31-27.png](./Images/1.png)

Rephraser encoder-decoer : T5-large

Narrator Video Encoder : **TimeSformer (TSF)**

Narrator Text Decoder : GPT-2XL

### 1.1.1 Narrator (Visually-conditioned LLM)

**Architecture**

시각적 입력을 기반으로 자동으로 설명을 생성하는 LLM.

전체 video에서 original annotations (X, Y)에서 학습되며, (X’, Y’)의 dense한 annotation을 생성한다.

LLM의 standard architecture를 따르지만, 몇개의 cross-attention 모듈을 추가했다.

→Query는 text 토큰이고, Key와 Value는 video frame에서 추출된 임베딩이다.

학습의 초기에는 시각적 임베딩의 비중을 줄이고, 점차 늘려나가기 위해 tanh-gating을 사용했다.

**Training**

Ground-truth annotation을 사용

**Inference**

- NARRATOR에게 시각적 입력과 함께 `<s>`라는 시작 토큰 제공
- 종료 토큰 `</s>`가 나올 때까지 다음 토큰의 확률 분포에서 순차적으로 샘플링을 하여 텍스트 시퀀스를 생성
- 각 단계에서 NARRATOR는 Nucleus Sampling을 사용해 다음 토큰을 선택
    - Nucleus Sampling은 전체 확률 질량의 대부분을 포함하는 토큰 집합에서 선택하는 방법으로, beam search보다 더 다양하고 자연스러운 텍스트를 생성할 수 있지만, 문장의 전반적인 의미와는 관련이 적은 정보나 노이즈를 포함할 수 있다.
- Nucleus Sampling으로 인해 발생할 수 있는 노이즈를 줄이기 위해, 동일한 시각적 입력에서 여러 번 샘플링을 반복하여 다양한 설명을 생성
- 생성된 설명 중 품질이 낮은 것들은 후처리 단계에서 필터링
    - 듀얼 인코더 모델을 사용해 시각적 임베딩과 텍스트 임베딩 간의 유사도를 계산하여 가장 관련성이 높은 설명만 선택된다.

### 1.1.2 Rephraser (Standard LLM)

Narrator로부터 생성된 data는 ground-truth pair보다 몇배는 많기 때문에 paraphrasing이 필요하다.

단순히 단어를 바꾸는 것뿐만 아니라 문장의 뉘앙스나 스타일을 변화시켜 더 자연스러운 문장을 생성할 수 있다.

Rephraser를 통해 모델이 다양한 텍스트 표현에 노출되면, 모델의 일반화 능력이 향상되어 새로운 데이터에 대해서도 더 잘 대응할 수 있다.

다양한 표현을 생성하더라도 Rephraser는 의미를 유지하면서 불필요한 노이즈를 최소화하는 방식으로 텍스트를 변형한다.

Annotation 결과는 (X, Y’’)으로 표현할 수 있다.

### Downstream tasks

Epic-Kitchens-100

- Multi-Instance Retrieval (EK-100 MIR)
- Action Recognition (EK-100 CLS)

Ego4D

- Multiple Choice Questions (EgoMCQ)
    
    ```bash
    질문: "사용자가 찾고 있는 것은 무엇인가?"
    a) 과일
    b) 냄비
    c) 칼
    d) 접시
    답: "c"
    ```
    
- Natural Language Query (EgoNLQ)
    
    ```bash
    질문: "사용자가 가장 마지막으로 구입한 물건은 무엇인가?"
    답: "사용자가 가장 마지막에 선택한 물건은 '사과'이다"
    ```
    

EGTEA

- Action Recognition

CharadesEgo

- Action Recognition

### Demo (CharadesEgo)

0CCES (third-person)

```bash
0: #O woman A picks a container from the shelf
1: #C C looks around
2: #C C looks around
3: #O woman X Picks a bottle from the shel
4: #O lady B puts bottle on fridge 
5: #O a lady X takes the bottle 
6: #C C looks around
7: #O woman Y picks a cup
8: #O lady X takes a cup from the fridge
9: #O woman A picks a container from the shelf
```

0CCESEGO (egocentric)

```bash
0: #C C presses a phone with her right hand.
1: #C C looks around
2: #C C holds the phone with her right hand.
3: #C C presses the phone with both hands.
4: #C C presses the phone
5: #C C looks around 
6: #C C switches on the phone
7: #C C operates the phone with her right hand.
8: #C C presses a button on the phone in her right hand with her left hand
9: #C C looks at the phone 
```

## 1.2 [EgoT2](https://github.com/facebookresearch/EgoT2) (22.12)

![Screenshot from 2024-08-28 16-42-08.png](./Images/2.png)

## 1.3 [EgoVLP](https://github.com/showlab/EgoVLP) (22.06)

![image.png](./Images/3.png)

## 1.4 [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2) (23.07)

![Screenshot from 2024-08-27 18-59-50.png](./Images/4.png)

기존의 egocentric VLP 프레임워크는 video encoder와 text encoder(dual encoder)를 별도로 사용하고, task-specific cross-modal information를 학습하는 과정이 fine tuning 단계에서만 이루어져, 통합된 시스템의 발전에 한계가 있었다.

비디오와 언어 백본에 직접 교차 모달 융합을 통합하여 pre-training 중에 강력한 video-text representation을 학습하며, 다양한 downstream 작업을 유연하고 효율적인 방식으로 지원하기 위해 cross-modal attention modules을 재사용한다. 

이로 인해 fine tuning 비용이 줄어들고, backbone strategy에서 제안된 fusion 방식은 추가적인 fusion-specific layer를 쌓는 것보다 경량화되어 계산 효율이 높다.

### Downstream tasks

Ego4D

- Multiple Choice Questions (EgoMCQ)
- Natural Language Query (EgoNLQ)
- Moment Query (EgoMQ)

QFVS (Query-focused video summarization)

- QFVS

EgoTaskQA

- Video Question Answering

CharadesEgo

- Action Recognition

Epic-Kitchen-100

- Multi-instance retrieval (EK-100 MIR)

### InternVideo-Ego4D

# 2. Video Action Recognintion

### **Action Recognition**


# 2. Online Inference

### Using Webcam …

# 3. Common components, Differences

## Encoder 차이

### 1. Dual Encoders

### 2. Shared Encoders

### 3. Encoders with Stacked Fusion Layers

# EgoCentric-Activity-Recognition-Study

## EGO4D (Challenge)

https://ego4d-data.org/

## Motivation

[MISAR: A Multimodal Instructional System with Augmented Reality](https://arxiv.org/abs/2310.11699)

## 0. Video Processing

https://github.com/Sense-X/UniFormer

https://github.com/OpenGVLab/UniFormerV2

![Screenshot from 2024-08-27 11-03-47.png]([https://prod-files-secure.s3.us-west-2.amazonaws.com/bcbf4472-a25d-41e0-8a11-f5743bd1e367/498fc868-9dfe-4969-abcb-2c044f68fc6d/Screenshot_from_2024-08-27_11-03-47.png](https://file.notion.so/f/f/bcbf4472-a25d-41e0-8a11-f5743bd1e367/498fc868-9dfe-4969-abcb-2c044f68fc6d/Screenshot_from_2024-08-27_11-03-47.png?table=block&id=8c13ff04-8b84-4e6b-9733-76c54fa27a23&spaceId=bcbf4472-a25d-41e0-8a11-f5743bd1e367&expirationTimestamp=1724889600000&signature=jrCPSTRa6QqY9VDNi5x0tOxP32a-Qys0pQ9sxlGaR54&downloadName=Screenshot+from+2024-08-27+11-03-47.png))

$$
X = DPE(X_{in}) + X_{in},
$$

$$  
Y = MHRA(Norm(X)) + X,
$$

$$  
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
- Global MHRA : 깊은 층에서는 긴 거리의 의존성을 학습한다. Self-attention 매커니즘과 유사하지만 video 인식 작업에서 공간과 시간적 차원을 모두 고려하도록 조정되었다.

RA는 token context encoding과 token affinity learning으로 구성된다.

> $U \in R^{C\times C}$ 는 N개의 헤드를 통합하기 위한 학습 가능한 매개변수 행렬
> 
> 
> $V_n(X) \in R^{L \times \frac{C}{N}}$ 는 token context encoding
> 
> $A_n \in R^{L \times L}$ 는 token affinity
> 

## 1. EgoT2, LaViLa

https://github.com/facebookresearch/EgoT2

https://github.com/facebookresearch/LaViLa

## 2. Online Inference

### Using Webcam …

## 3. Common components, Differences

# Korean Sign Language Recognition & Translation

한국 수어(수화) 인식 및 한국어 번역 시스템

## 📋 개요

이 프로젝트는 **연속 수어 비디오**에서 키포인트를 추출하여 **글로스(형태소) 시퀀스**로 인식한 후, **한국어 자연문장**으로 변환하는 End-to-End 파이프라인입니다.

**성능 (검증 세트):**
- 🔤 글로스 정확도 (EXACT): 65.12%
- 🔤 글로스 WER: 29.86%
- 🇰🇷 한국어 WER: 0.61
- 🇰🇷 한국어 BLEU-2: 0.7409
- 🤖 GPT 의미 일치도: 0.6911

## 🏗️ 시스템 구조

```
비디오 (수어 영상)
    ↓
OpenPose (키포인트 추출: 67 joints × 3 dims)
    ↓
CTC Transformer (글로스 인식)
    ↓
규칙 기반 변환 / GPT 변환 (한국어 생성)
    ↓
한국어 자연문장
```

### 주요 컴포넌트

1. **CTC Transformer Model**
   - 입력: 키포인트 시퀀스[T, 201]
   - 출력: 글로스 시퀀스 (형태소)
   - 아키텍처: Transformer Encoder + CTC Loss
   - 성능: 65.12% 글로스 정확도

2. **글로스 → 한국어 변환**
   - 방식 1: 규칙 기반 (gloss_to_korean_v2)
   - 방식 2: GPT 기반 (gpt-4o-mini, 프롬프트 엔지니어링)

3. **평가 메트릭**
   - 글로스 레벨: WER, BLEU, Exact Match
   - 한국어 레벨: CER, WER, BLEU-2
   - 의미 평가: GPT 기반 의미 유사도

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://huggingface.co/[USER]/korean-sign-language-recognition
cd korean-sign-language-recognition

# 의존성 설치
pip install -r requirements.txt

# OpenAI API 키 설정 (GPT 변환 사용 시)
export OPENAI_API_KEY="your-api-key"
```

### 추론 (예제)

```python
import torch
from models.ctc_transformer import CTCTransformerEncoder
from utils.gloss_rules import gloss_to_korean_v2

# 1. 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTCTransformerEncoder(
    in_dim=201,
    num_classes=107,
    d_model=256,
    nhead=8,
    num_layers=4
)
# checkpoint 로드
checkpoint = torch.load("best_ctc_transformer.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# 2. 키포인트 입력 [1, T, 201]
X = torch.randn(1, 100, 201).to(device)

# 3. 글로스 예측
with torch.no_grad():
    logits = model(X)  # [1, T, 107]

# 4. 글로스 디코딩
gloss_seq = ["저기", "쓰러지다"]

# 5. 한국어 변환
korean = gloss_to_korean_v2(gloss_seq)
print(f"생성된 문장: {korean}")
# 출력: "저기가 쓰러졌습니다."
```

## 📊 데이터셋

- **크기**: 258개 검증 샘플
- **구성**: 긴급 상황, 의료 상황 등 실제 수어 시나리오
- **형식**: OpenPose JSON 키포인트 + 형태소 레이블

### 시나리오 예시

**긴급 상황:**
- "저기 사람이 쓰러졌어요. 도와주세요."
- "119에 전화해주세요."

**의료 상황:**
- "환자 손녀인데 병문안 왔다."
- "검사받고 치료 받으며 회복 중이다."

## 🛠️ 변환 방식 비교

### 1. 규칙 기반 (gloss_to_korean_v2)
- ✅ 빠름, 비용 없음
- ✅ 결정적 (항상 같은 결과)
- ❌ 제한된 표현력
- 성능: WER 0.61, BLEU 0.74

### 2. GPT 기반 (gpt-4o-mini)
- ✅ 자연스러운 문장
- ✅ 높은 성능 (의미 일치도 0.69)
- ❌ API 비용 필요 (~$0.001-0.002 per call)
- 성능: WER 0.61, BLEU 0.74

## 📈 성능 평가

```
글로스 레벨:
  - EXACT: 65.12%
  - WER: 29.86%
  - BLEU: 74.50%

한국어 레벨:
  - CER: 0.27%
  - WER: 0.61%
  - BLEU-2: 74.09%

의미 평가:
  - GPT 의미 일치도: 0.6911
```

## 📚 관련 논문 & 참고

- CTC Loss: [Graves et al., 2006]
- Transformer: [Vaswani et al., 2017]
- Sign Language Recognition: [Koller et al., 2019]

## 🔄 모델 버전

- **v0.1**: 초기 릴리스
  - CTC Transformer 글로스 인식
  - 규칙 기반 + GPT 기반 변환
  - 성능: 65.12% 글로스 정확도

- **v0.2** (예정): 프롬프트 최적화
  - 시스템 프롬프트 강화
  - 의미 평가 개선

- **v1.0** (예정): 최종 출시
  - 성능 80%+ 달성
  - 추가 시나리오 지원

## 💬 사용 사례

1. **실시간 수어 자막 생성**
   - 수어 비디오 → 자동 한국어 캡션

2. **수어-음성 통역 보조**
   - 통역사 업무 효율화

3. **청각 장애인 교육**
   - 수어 학습 데이터 생성

## ⚙️ 커스터마이제이션

### 프롬프트 수정

`eval_full_pipeline.py` 또는 `utils/gloss_rules.py`에서:

```python
SYSTEM_PROMPT = """
당신은 '한국 수어 글로스 → 한국어 문장' 전문 변환 시스템입니다.
... (프롬프트 내용)
"""
```

### 규칙 추가

`utils/gloss_rules.py`의 `gloss_to_korean_v2()` 함수에 새로운 패턴 추가:

```python
def gloss_to_korean_v2(gloss_list):
    # 새로운 규칙 추가
    if gloss_list == ["...생략..."]:
        return "새로운 문장"
```

## 📝 라이선스

MIT License - 자유로운 사용 가능

## 👥 기여

개선 사항, 버그 리포트, 피드백은 이슈/PR로 환영합니다!

## 📧 문의

문제 발생 시 GitHub Issues 또는 이메일로 연락해주세요.

---

**마지막 업데이트**: 2025년 12월 7일  
**상태**: 개발 중 (v0.1)  
**다음 목표**: 프롬프트 최적화 → 성능 80%+ 달성

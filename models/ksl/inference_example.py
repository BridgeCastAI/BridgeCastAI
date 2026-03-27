"""
간단한 추론 스크립트
Hugging Face에서 사용할 예제
"""
import torch
import json
from models.ctc_transformer import CTCTransformerEncoder
from utils.gloss_rules import gloss_to_korean_v2

def load_model(checkpoint_path, device="cuda"):
    """CTC Transformer 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = CTCTransformerEncoder(
        in_dim=config['in_dim'],
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=1024
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    vocab = checkpoint['vocab']
    idx2gloss = {int(k): v for k, v in vocab['itos'].items()}
    
    return model, idx2gloss, device

def decode_ctc_output(logits, idx2gloss):
    """CTC 로짓을 글로스 시퀀스로 디코딩"""
    if logits.dim() == 3:
        logits = logits[0]
    
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    
    gloss_ids = []
    prev_id = -1
    for pred_id in predictions:
        pred_id = pred_id.item()
        if pred_id != 0 and pred_id != prev_id:
            gloss_ids.append(pred_id)
        prev_id = pred_id
    
    gloss_list = [idx2gloss.get(gid, f"<unk-{gid}>") for gid in gloss_ids]
    return gloss_list

def recognize_and_translate(keypoints_tensor, model, idx2gloss, device, method="rule"):
    """
    수어 키포인트 → 글로스 → 한국어 변환
    
    Args:
        keypoints_tensor: [T, 201] or [1, T, 201]
        model: CTCTransformer 모델
        idx2gloss: 인덱스→글로스 맵핑
        device: torch device
        method: "rule" (규칙 기반) 또는 "gpt" (GPT 기반)
    
    Returns:
        dict: {
            "gloss_sequence": [...],
            "korean_sentence": "...",
            "confidence": 0.xx
        }
    """
    # 입력 처리
    if keypoints_tensor.dim() == 2:
        keypoints_tensor = keypoints_tensor.unsqueeze(0)
    
    keypoints_tensor = keypoints_tensor.to(device)
    
    # 글로스 예측
    with torch.no_grad():
        logits = model(keypoints_tensor)  # [1, T, num_classes]
    
    gloss_seq = decode_ctc_output(logits, idx2gloss)
    
    # 한국어 변환
    if method == "rule":
        korean = gloss_to_korean_v2(gloss_seq)
        confidence = 0.5  # 규칙 기반은 신뢰도 미지정
    elif method == "gpt":
        try:
            from openai import OpenAI
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수 필수")
            
            client = OpenAI(api_key=api_key)
            prompt = f"다음 글로스를 자연스러운 한국어로 변환하세요: {gloss_seq}\n문장만 출력하세요."
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            korean = response.choices[0].message.content.strip()
            confidence = 0.7  # GPT 변환은 더 높은 신뢰도
        except Exception as e:
            print(f"⚠️ GPT 변환 실패: {e}")
            korean = gloss_to_korean_v2(gloss_seq)
            confidence = 0.5
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "gloss_sequence": gloss_seq,
        "korean_sentence": korean,
        "confidence": confidence,
        "num_glosses": len(gloss_seq)
    }

# ============================================================
# 사용 예제
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("한국 수어 인식 & 번역 시스템 - 추론 예제")
    print("=" * 80)
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "best_ctc_transformer.pth"  # 체크포인트 경로
    
    print(f"\n[1] 모델 로드 중... (Device: {device})")
    try:
        model, idx2gloss, device = load_model(checkpoint_path, device)
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit(1)
    
    # 더미 키포인트 입력 생성 (실제로는 OpenPose에서 추출한 데이터)
    print(f"\n[2] 더미 입력 생성... (T=100, D=201)")
    dummy_keypoints = torch.randn(100, 201)  # [T, 201]
    print(f"   입력 shape: {dummy_keypoints.shape}")
    
    # 추론 (규칙 기반)
    print(f"\n[3] 규칙 기반 변환")
    result_rule = recognize_and_translate(
        dummy_keypoints, model, idx2gloss, device, method="rule"
    )
    
    print(f"   글로스: {' '.join(result_rule['gloss_sequence'])}")
    print(f"   한국어: {result_rule['korean_sentence']}")
    print(f"   신뢰도: {result_rule['confidence']:.2f}")
    
    # 추론 (GPT 기반) - OpenAI API 키 필요
    print(f"\n[4] GPT 기반 변환 (선택사항)")
    print("   (OPENAI_API_KEY 환경변수 설정 필요)")
    
    try:
        result_gpt = recognize_and_translate(
            dummy_keypoints, model, idx2gloss, device, method="gpt"
        )
        print(f"   글로스: {' '.join(result_gpt['gloss_sequence'])}")
        print(f"   한국어: {result_gpt['korean_sentence']}")
        print(f"   신뢰도: {result_gpt['confidence']:.2f}")
    except Exception as e:
        print(f"   ⚠️ GPT 변환 스킵: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 추론 예제 완료")
    print("=" * 80)

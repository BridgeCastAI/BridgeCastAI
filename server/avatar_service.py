"""
BridgeCast AI — Sign Language Avatar Service
Converts text/speech to ASL gloss sequences and maps them to animation data
for a 2D/3D signing avatar.

Pipeline: Text -> ASL Gloss (Azure OpenAI) -> Animation Keyframes -> Avatar
Reference: GenASL (AWS), sign.mt, Sign-Kit
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# Azure OpenAI client

def _get_client() -> AzureOpenAI:
    """Return an AzureOpenAI client configured from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")

    if not endpoint or not key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version="2024-06-01",
    )


# ASL (American Sign Language) Gloss System Prompt

ASL_GLOSS_SYSTEM_PROMPT = """\
You are an expert ASL (American Sign Language) linguist and gloss translator.

Your task: Convert English text into an ASL gloss sequence.

ASL gloss rules:
1. ASL uses TOPIC-COMMENT structure (e.g., "I am going to the store" -> "STORE I GO")
2. Remove articles (a, an, the) — ASL does not use them
3. Remove "to be" verbs (am, is, are, was, were) — ASL conveys these through context
4. Use base/root forms of verbs (going -> GO, wanted -> WANT)
5. Pronouns: I, YOU, HE/SHE (use point direction), WE, THEY
6. For questions, add a question marker and reorder: "What is your name?" -> "YOUR NAME WHAT"
7. Negation comes after the verb: "I don't understand" -> "I UNDERSTAND NOT"
8. Time markers come first: "Yesterday I went to school" -> "YESTERDAY SCHOOL I GO"
9. Use only common ASL vocabulary — prefer simple, high-frequency signs
10. Compound concepts: break into simpler signs when possible

IMPORTANT: Only use signs from this supported vocabulary when possible:
HELLO, THANK-YOU, YES, NO, HELP, PLEASE, SORRY, GOOD, BAD, QUESTION,
AGREE, DISAGREE, UNDERSTAND, NAME, NICE, MEET, GOODBYE, WAIT, REPEAT,
SLOW-DOWN, I-LOVE-YOU, GO, COME, WANT, NEED, LIKE, WORK, HOME, SCHOOL,
FRIEND, FAMILY, I, YOU, WE, THEY, HE, SHE, MY, YOUR, WHAT, WHERE, WHEN,
HOW, WHO, WHY, NOT, CAN, WILL, HAVE, DO, MAKE, SEE, HEAR, KNOW, THINK,
FEEL, HAPPY, SAD, ANGRY, SCARED, TIRED, HUNGRY, EAT, DRINK, WATER, FOOD,
MORE, FINISH, START, STOP, AGAIN, NOW, TODAY, TOMORROW, YESTERDAY, TIME,
MORNING, NIGHT, MONEY, LEARN, TEACH, READ, WRITE, TALK, SIGN, LANGUAGE,
DEAF, HEARING, WORLD, PEOPLE, CHILD, MAN, WOMAN, DOCTOR, TEACHER, STUDENT,
TEAM, PROJECT, IDEA, PLAN, SUGGEST, IMPORTANT, EXPLAIN, MAYBE, PROBLEM,
SOLUTION, EXCITED, WORRY, CONFIDENT, NEXT, SCHEDULE, WELCOME, INTRODUCE,
PRESENT, SHARE, MEETING, COMPUTER, EMAIL, OFFICE, NEW, FIRST, EXPERIENCE,
TOGETHER, SUPPORT, TRY, IMPROVE, HOPE, PROUD, WONDERFUL, GREAT, SAME,
DIFFERENT, DIFFICULT, EASY, POSSIBLE, READY, DISCUSS, OPINION, FOCUS

If a word has no direct sign equivalent, use the closest available sign or
fingerspell it (output as F-I-N-G-E-R-S-P-E-L-L with hyphens).

Output format: Return ONLY a JSON array of gloss strings, nothing else.
Example input: "I am going to the store tomorrow"
Example output: ["TOMORROW", "STORE", "I", "GO"]

Example input: "What is your name?"
Example output: ["YOUR", "NAME", "WHAT"]

Example input: "Nice to meet you"
Example output: ["NICE", "MEET", "YOU"]
"""


# KSL (Korean Sign Language) Gloss System Prompt

KSL_GLOSS_SYSTEM_PROMPT = """\
You are an expert KSL (Korean Sign Language / 한국수어) linguist and gloss translator.

Your task: Convert Korean text into a KSL gloss sequence.

KSL gloss rules:
1. KSL uses SOV (Subject-Object-Verb) word order: "나는 학교에 간다" → "학교 나 가다"
2. Remove particles/postpositions (은/는/이/가/을/를/에/에서) — KSL does not use them
3. Remove copula (이다/입니다) — KSL conveys these through context and facial expression
4. Use base/dictionary forms of verbs (갔다 → 가다, 먹었어요 → 먹다)
5. Time markers come first: "어제 학교에 갔다" → "어제 학교 나 가다"
6. Questions: add question marker at end + raised eyebrows: "이름이 뭐야?" → "너 이름 뭐"
7. Negation after verb: "이해 못 해요" → "나 이해하다 아니다"
8. Adjectives follow the noun they modify
9. Use only common KSL vocabulary — prefer simple, high-frequency signs
10. Facial expressions serve grammatical roles (questions=raised brows, negation=head shake)

IMPORTANT: Only use signs from this supported vocabulary when possible:
안녕, 감사, 네, 아니오, 도움, 부탁, 미안, 좋다, 나쁘다, 질문,
동의, 반대, 이해하다, 이름, 반갑다, 만나다, 안녕히, 기다리다, 다시, 천천히,
사랑, 가다, 오다, 원하다, 필요하다, 좋아하다, 일하다, 집, 학교,
친구, 가족, 나, 너, 우리, 그들, 그, 그녀, 내, 너의, 뭐, 어디, 언제,
어떻게, 누구, 왜, 아니다, 할-수-있다, 하다, 만들다, 보다, 듣다, 알다, 생각하다,
느끼다, 행복, 슬프다, 화나다, 무섭다, 피곤하다, 배고프다, 먹다, 마시다, 물, 음식,
더, 끝, 시작, 멈추다, 또, 지금, 오늘, 내일, 어제, 시간,
아침, 밤, 돈, 배우다, 가르치다, 읽다, 쓰다, 말하다, 수어, 언어,
청각장애, 청인, 세계, 사람, 아이, 남자, 여자, 의사, 선생님, 학생,
팀, 프로젝트, 아이디어, 계획, 제안, 중요하다, 설명하다, 아마, 문제,
해결, 신나다, 걱정, 자신감, 다음, 일정, 환영, 소개하다,
발표, 공유하다, 회의, 컴퓨터, 이메일, 사무실, 새롭다, 처음, 경험,
함께, 지원, 노력하다, 개선, 희망, 자랑스럽다, 훌륭하다, 대단하다, 같다,
다르다, 어렵다, 쉽다, 가능하다, 준비, 토론, 의견, 집중

If a word has no direct KSL sign, use the closest available sign or
fingerspell it in Korean: ㄱ-ㅏ-ㄴ-ㅏ-ㄷ-ㅏ (with hyphens between jamo).

Output format: Return ONLY a JSON array of gloss strings, nothing else.
Example input: "내일 학교에 갈 거예요"
Example output: ["내일", "학교", "나", "가다"]

Example input: "이름이 뭐예요?"
Example output: ["너", "이름", "뭐"]

Example input: "만나서 반갑습니다"
Example output: ["만나다", "반갑다"]
"""


# TSL (Taiwan Sign Language) Gloss System Prompt

TSL_GLOSS_SYSTEM_PROMPT = """\
You are an expert TSL (Taiwan Sign Language / 台灣手語) linguist and gloss translator.

Your task: Convert Chinese (Traditional) text into a TSL gloss sequence.

TSL gloss rules:
1. TSL uses SOV word order (similar to Japanese Sign Language family): "我去學校" → "學校 我 去"
2. Remove function words (的/了/嗎/吧/呢) — TSL does not use them
3. Remove copula (是) — TSL conveys identity through spatial grammar
4. Use base forms of verbs (去了 → 去, 吃過 → 吃)
5. Time markers come first: "昨天我去學校" → "昨天 學校 我 去"
6. Questions: non-manual marker (raised eyebrows) + question sign at end: "你叫什麼名字？" → "你 名字 什麼"
7. Negation after verb: "我不懂" → "我 懂 不"
8. TSL belongs to JSL (Japanese Sign Language) family — shares ~60% lexical similarity with JSL
9. TSL is distinct from CSL (Chinese Sign Language used in mainland China)
10. Classifier predicates are important for describing actions and spatial relationships

IMPORTANT: Only use signs from this supported vocabulary when possible:
你好, 謝謝, 是, 不是, 幫助, 請, 對不起, 好, 不好, 問題,
同意, 反對, 了解, 名字, 高興, 見面, 再見, 等, 再, 慢,
愛, 去, 來, 想要, 需要, 喜歡, 工作, 家, 學校,
朋友, 家人, 我, 你, 我們, 他們, 他, 她, 我的, 你的, 什麼, 哪裡, 什麼時候,
怎麼, 誰, 為什麼, 不, 可以, 會, 有, 做, 看, 聽, 知道, 想,
覺得, 開心, 難過, 生氣, 害怕, 累, 餓, 吃, 喝, 水, 食物,
更多, 結束, 開始, 停, 又, 現在, 今天, 明天, 昨天, 時間,
早上, 晚上, 錢, 學, 教, 讀, 寫, 說話, 手語, 語言,
聾, 聽人, 世界, 人, 小孩, 男人, 女人, 醫生, 老師, 學生,
團隊, 計畫, 想法, 建議, 重要, 解釋, 也許, 問題,
解決, 興奮, 擔心, 有信心, 下一個, 行程, 歡迎, 介紹,
報告, 分享, 會議, 電腦, 電子郵件, 辦公室, 新, 第一, 經驗,
一起, 支持, 努力, 改善, 希望, 驕傲, 很棒, 厲害, 一樣,
不同, 難, 容易, 可能, 準備, 討論, 意見, 專注

If a word has no direct TSL sign, use the closest available sign or
fingerspell using 注音符號: ㄅ-ㄆ-ㄇ (with hyphens between symbols).

Output format: Return ONLY a JSON array of gloss strings, nothing else.
Example input: "明天我要去學校"
Example output: ["明天", "學校", "我", "去"]

Example input: "你叫什麼名字？"
Example output: ["你", "名字", "什麼"]

Example input: "很高興認識你"
Example output: ["見面", "高興"]
"""

# Gloss prompt registry (keyed by sign language code)
GLOSS_SYSTEM_PROMPTS = {
    "asl": ASL_GLOSS_SYSTEM_PROMPT,
    "ksl": KSL_GLOSS_SYSTEM_PROMPT,
    "tsl": TSL_GLOSS_SYSTEM_PROMPT,
}


# Sign vocabulary — animation keyframe data
# Each sign: handshape positions, facial expression, movement type
# Coordinates are relative to avatar body center:
#   x: -1 (far left) to 1 (far right)
#   y: -1 (low/waist) to 1 (high/above head)

SIGN_ANIMATIONS: Dict[str, Dict[str, Any]] = {
    "HELLO": {
        "gloss": "HELLO",
        "description": "Open hand waves near forehead",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.7}, "end": {"x": 0.5, "y": 0.8}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "wave",
        "duration_ms": 1200,
        "repeat": 2,
    },
    "THANK-YOU": {
        "gloss": "THANK-YOU",
        "description": "Flat hand from chin forward",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.3, "y": 0.3}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile",
        "head_movement": "nod",
        "body_movement": "slight_forward",
        "movement_type": "arc_forward",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "YES": {
        "gloss": "YES",
        "description": "Fist nods like a head nodding",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.4}, "end": {"x": 0.3, "y": 0.3}, "shape": "fist", "palm": "forward"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "nod_wrist",
        "duration_ms": 800,
        "repeat": 2,
    },
    "NO": {
        "gloss": "NO",
        "description": "Index and middle finger snap to thumb",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.4}, "end": {"x": 0.3, "y": 0.4}, "shape": "pinch_two", "palm": "side"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "headshake",
        "head_movement": "shake",
        "body_movement": "none",
        "movement_type": "snap",
        "duration_ms": 600,
        "repeat": 1,
    },
    "HELP": {
        "gloss": "HELP",
        "description": "Fist on flat palm, both rise",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.0}, "end": {"x": 0.1, "y": 0.3}, "shape": "fist_thumb_up", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": -0.1}, "end": {"x": -0.1, "y": 0.2}, "shape": "flat", "palm": "up"},
        "expression": "concerned",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "rise",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "PLEASE": {
        "gloss": "PLEASE",
        "description": "Flat hand circles on chest",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.2}, "end": {"x": 0.0, "y": 0.2}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "polite",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "circle_chest",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SORRY": {
        "gloss": "SORRY",
        "description": "Fist circles on chest",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.2}, "end": {"x": 0.0, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "apologetic",
        "head_movement": "slight_tilt",
        "body_movement": "slight_forward",
        "movement_type": "circle_chest",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "GOOD": {
        "gloss": "GOOD",
        "description": "Flat hand from chin to open palm",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.2, "y": 0.1}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "flat", "palm": "up"},
        "expression": "smile",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "down_to_palm",
        "duration_ms": 800,
        "repeat": 1,
    },
    "BAD": {
        "gloss": "BAD",
        "description": "Flat hand from chin flips down",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.2, "y": 0.0}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "frown",
        "head_movement": "shake",
        "body_movement": "none",
        "movement_type": "chin_flip_down",
        "duration_ms": 800,
        "repeat": 1,
    },
    "QUESTION": {
        "gloss": "QUESTION",
        "description": "Index finger draws question mark",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.5}, "end": {"x": 0.3, "y": 0.3}, "shape": "index_point", "palm": "forward"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "lean_forward",
        "movement_type": "draw_question",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "AGREE": {
        "gloss": "AGREE",
        "description": "Index finger from forehead to pointing at partner, both hands meet",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.6}, "end": {"x": 0.2, "y": 0.1}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.6}, "end": {"x": -0.2, "y": 0.1}, "shape": "index_point", "palm": "down"},
        "expression": "nod_affirm",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "both_down_together",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "DISAGREE": {
        "gloss": "DISAGREE",
        "description": "Both index fingers touch then separate",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.4, "y": 0.3}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": -0.4, "y": 0.3}, "shape": "index_point", "palm": "in"},
        "expression": "frown",
        "head_movement": "shake",
        "body_movement": "none",
        "movement_type": "separate",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "UNDERSTAND": {
        "gloss": "UNDERSTAND",
        "description": "Index finger flicks up near temple",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.6}, "end": {"x": 0.3, "y": 0.7}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "flick_up",
        "duration_ms": 800,
        "repeat": 1,
    },
    "NAME": {
        "gloss": "NAME",
        "description": "Two H-fingers tap on each other",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.35}, "end": {"x": 0.15, "y": 0.3}, "shape": "h_fingers", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.25}, "end": {"x": -0.15, "y": 0.25}, "shape": "h_fingers", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap",
        "duration_ms": 800,
        "repeat": 2,
    },
    "NICE": {
        "gloss": "NICE",
        "description": "Dominant flat hand slides across non-dominant palm",
        "dominant_hand": {"start": {"x": -0.1, "y": 0.1}, "end": {"x": 0.3, "y": 0.1}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": 0.0, "y": 0.0}, "end": {"x": 0.0, "y": 0.0}, "shape": "flat", "palm": "up"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slide_across",
        "duration_ms": 800,
        "repeat": 1,
    },
    "MEET": {
        "gloss": "MEET",
        "description": "Two index fingers come together",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.2}, "end": {"x": 0.05, "y": 0.2}, "shape": "index_up", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.2}, "end": {"x": -0.05, "y": 0.2}, "shape": "index_up", "palm": "right"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "come_together",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "GOODBYE": {
        "gloss": "GOODBYE",
        "description": "Open hand waves, fingers fold repeatedly",
        "dominant_hand": {"start": {"x": 0.4, "y": 0.5}, "end": {"x": 0.5, "y": 0.6}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "wave_fold",
        "duration_ms": 1200,
        "repeat": 2,
    },
    "WAIT": {
        "gloss": "WAIT",
        "description": "Both hands up, fingers wiggle",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.2, "y": 0.3}, "shape": "open_spread", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.3}, "end": {"x": -0.2, "y": 0.3}, "shape": "open_spread", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "wiggle_fingers",
        "duration_ms": 1500,
        "repeat": 1,
    },
    "REPEAT": {
        "gloss": "REPEAT",
        "description": "Curved hand flips onto flat palm repeatedly",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.0, "y": 0.1}, "shape": "curved", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "flip_to_palm",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "SLOW-DOWN": {
        "gloss": "SLOW-DOWN",
        "description": "One hand slides slowly up the back of the other",
        "dominant_hand": {"start": {"x": 0.1, "y": -0.1}, "end": {"x": 0.1, "y": 0.2}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "flat", "palm": "down"},
        "expression": "calm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slow_slide_up",
        "duration_ms": 1500,
        "repeat": 1,
    },
    "I-LOVE-YOU": {
        "gloss": "I-LOVE-YOU",
        "description": "ILY handshape held out",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.4, "y": 0.4}, "shape": "ily", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "big_smile",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "hold_out",
        "duration_ms": 1500,
        "repeat": 1,
    },
    "GO": {
        "gloss": "GO",
        "description": "Both index fingers point and move forward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.4, "y": 0.3}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": 0.2, "y": 0.3}, "shape": "index_point", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "both_forward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "COME": {
        "gloss": "COME",
        "description": "Index finger beckons toward body",
        "dominant_hand": {"start": {"x": 0.4, "y": 0.3}, "end": {"x": 0.1, "y": 0.2}, "shape": "index_point", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "inviting",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "beckon",
        "duration_ms": 800,
        "repeat": 2,
    },
    "WANT": {
        "gloss": "WANT",
        "description": "Both hands open, pull toward body",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.15, "y": 0.1}, "shape": "claw", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.2}, "end": {"x": -0.15, "y": 0.1}, "shape": "claw", "palm": "up"},
        "expression": "eager",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pull_toward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "NEED": {
        "gloss": "NEED",
        "description": "X handshape bends down at wrist",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.3, "y": 0.1}, "shape": "x_shape", "palm": "forward"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "serious",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "bend_down",
        "duration_ms": 800,
        "repeat": 1,
    },
    "LIKE": {
        "gloss": "LIKE",
        "description": "Thumb and middle finger pull away from chest",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.2}, "end": {"x": 0.2, "y": 0.2}, "shape": "thumb_middle", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pull_from_chest",
        "duration_ms": 800,
        "repeat": 1,
    },
    "WORK": {
        "gloss": "WORK",
        "description": "Fist taps on back of other fist",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.1, "y": 0.1}, "shape": "fist", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "fist", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_fist",
        "duration_ms": 800,
        "repeat": 2,
    },
    "HOME": {
        "gloss": "HOME",
        "description": "Fingertips touch near mouth then cheek",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.5}, "end": {"x": 0.2, "y": 0.55}, "shape": "flat_closed", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "warm",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "mouth_to_cheek",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SCHOOL": {
        "gloss": "SCHOOL",
        "description": "Clap hands twice",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.0, "y": 0.15}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.1}, "end": {"x": 0.0, "y": 0.15}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "clap",
        "duration_ms": 800,
        "repeat": 2,
    },
    "FRIEND": {
        "gloss": "FRIEND",
        "description": "Index fingers hook and reverse",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.1, "y": 0.15}, "shape": "hook_index", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.2}, "shape": "hook_index", "palm": "down"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "hook_reverse",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "FAMILY": {
        "gloss": "FAMILY",
        "description": "Both F-hands circle forward to form a circle",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.15, "y": 0.3}, "shape": "f_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.3}, "shape": "f_shape", "palm": "out"},
        "expression": "warm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "circle_together",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "I": {
        "gloss": "I",
        "description": "Index points to self/chest",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.2}, "end": {"x": 0.0, "y": 0.15}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "point_self",
        "duration_ms": 500,
        "repeat": 1,
    },
    "YOU": {
        "gloss": "YOU",
        "description": "Index points forward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.4, "y": 0.3}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "point_forward",
        "duration_ms": 500,
        "repeat": 1,
    },
    "WE": {
        "gloss": "WE",
        "description": "Index points to self then arcs to include others",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.3}, "end": {"x": 0.3, "y": 0.3}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "inclusive",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "arc_self_to_others",
        "duration_ms": 800,
        "repeat": 1,
    },
    "WHAT": {
        "gloss": "WHAT",
        "description": "Both palms up, shake slightly",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.1}, "end": {"x": 0.25, "y": 0.15}, "shape": "open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.1}, "end": {"x": -0.25, "y": 0.15}, "shape": "open", "palm": "up"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "lean_forward",
        "movement_type": "shake_palms_up",
        "duration_ms": 800,
        "repeat": 1,
    },
    "WHERE": {
        "gloss": "WHERE",
        "description": "Index finger waves side to side",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.4}, "end": {"x": 0.35, "y": 0.4}, "shape": "index_point", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "none",
        "movement_type": "wave_side",
        "duration_ms": 800,
        "repeat": 1,
    },
    "NOT": {
        "gloss": "NOT",
        "description": "Thumb under chin flicks forward",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.45}, "end": {"x": 0.2, "y": 0.35}, "shape": "thumb_out", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "frown",
        "head_movement": "shake",
        "body_movement": "none",
        "movement_type": "chin_flick",
        "duration_ms": 600,
        "repeat": 1,
    },
    "HAPPY": {
        "gloss": "HAPPY",
        "description": "Both flat hands brush up on chest repeatedly",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.1}, "end": {"x": 0.1, "y": 0.3}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.1}, "end": {"x": -0.1, "y": 0.3}, "shape": "flat", "palm": "in"},
        "expression": "big_smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "brush_up_chest",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "SAD": {
        "gloss": "SAD",
        "description": "Both open hands drop down in front of face",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.6}, "end": {"x": 0.1, "y": 0.2}, "shape": "open_spread", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.6}, "end": {"x": -0.1, "y": 0.2}, "shape": "open_spread", "palm": "in"},
        "expression": "sad",
        "head_movement": "drop",
        "body_movement": "slight_forward",
        "movement_type": "drop_in_front",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "KNOW": {
        "gloss": "KNOW",
        "description": "Fingertips tap forehead",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.65}, "end": {"x": 0.2, "y": 0.7}, "shape": "flat_closed", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_forehead",
        "duration_ms": 700,
        "repeat": 1,
    },
    "THINK": {
        "gloss": "THINK",
        "description": "Index finger touches forehead",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.6}, "end": {"x": 0.2, "y": 0.68}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "thoughtful",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "touch_forehead",
        "duration_ms": 800,
        "repeat": 1,
    },
    "LEARN": {
        "gloss": "LEARN",
        "description": "Open hand picks up from palm and brings to forehead",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.0}, "end": {"x": 0.15, "y": 0.65}, "shape": "flat_closed", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": -0.05}, "end": {"x": -0.1, "y": -0.05}, "shape": "flat", "palm": "up"},
        "expression": "interested",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "palm_to_forehead",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "MY": {
        "gloss": "MY",
        "description": "Flat hand on chest",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.2}, "end": {"x": 0.0, "y": 0.2}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "touch_chest",
        "duration_ms": 500,
        "repeat": 1,
    },
    "YOUR": {
        "gloss": "YOUR",
        "description": "Flat hand pushes forward toward other person",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.35, "y": 0.3}, "shape": "flat", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "push_forward",
        "duration_ms": 500,
        "repeat": 1,
    },
    "HAVE": {
        "gloss": "HAVE",
        "description": "Both bent hands touch chest",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.25}, "end": {"x": 0.1, "y": 0.2}, "shape": "bent", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.25}, "end": {"x": -0.1, "y": 0.2}, "shape": "bent", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "touch_chest_both",
        "duration_ms": 700,
        "repeat": 1,
    },
    "SEE": {
        "gloss": "SEE",
        "description": "V-hand from eyes outward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.6}, "end": {"x": 0.3, "y": 0.4}, "shape": "v_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "eyes_outward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "FINISH": {
        "gloss": "FINISH",
        "description": "Both open hands flip outward",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.3, "y": 0.35}, "shape": "open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.3, "y": 0.35}, "shape": "open", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "flip_outward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "NOW": {
        "gloss": "NOW",
        "description": "Both bent hands drop slightly",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.15, "y": 0.2}, "shape": "bent", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.2}, "shape": "bent", "palm": "up"},
        "expression": "emphasis",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "drop_both",
        "duration_ms": 600,
        "repeat": 1,
    },
    "TODAY": {
        "gloss": "TODAY",
        "description": "NOW + NOW compound — both hands drop twice",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.15, "y": 0.15}, "shape": "bent", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.15}, "shape": "bent", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "drop_both",
        "duration_ms": 800,
        "repeat": 2,
    },
    "TOMORROW": {
        "gloss": "TOMORROW",
        "description": "Thumb on cheek moves forward",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.35, "y": 0.55}, "shape": "thumb_up", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "cheek_forward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "YESTERDAY": {
        "gloss": "YESTERDAY",
        "description": "Thumb on cheek moves backward",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.05, "y": 0.6}, "shape": "thumb_up", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "cheek_backward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "PEOPLE": {
        "gloss": "PEOPLE",
        "description": "Both P-hands alternate circling",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.2}, "end": {"x": 0.15, "y": 0.2}, "shape": "p_shape", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.2}, "end": {"x": -0.15, "y": 0.2}, "shape": "p_shape", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "alternate_circle",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "DEAF": {
        "gloss": "DEAF",
        "description": "Index touches ear then mouth",
        "dominant_hand": {"start": {"x": 0.25, "y": 0.6}, "end": {"x": 0.1, "y": 0.5}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "ear_to_mouth",
        "duration_ms": 900,
        "repeat": 1,
    },
    "HEARING": {
        "gloss": "HEARING",
        "description": "Index circles near mouth",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.5}, "end": {"x": 0.1, "y": 0.5}, "shape": "index_point", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "circle_near_mouth",
        "duration_ms": 900,
        "repeat": 1,
    },
    "SIGN": {
        "gloss": "SIGN",
        "description": "Both index fingers alternate circling",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.2, "y": 0.3}, "shape": "index_point", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.3}, "end": {"x": -0.2, "y": 0.3}, "shape": "index_point", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "alternate_circle",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "LANGUAGE": {
        "gloss": "LANGUAGE",
        "description": "Both L-hands pull apart",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.3}, "end": {"x": 0.3, "y": 0.3}, "shape": "l_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.3}, "end": {"x": -0.3, "y": 0.3}, "shape": "l_shape", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pull_apart",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "WORLD": {
        "gloss": "WORLD",
        "description": "Both W-hands circle around each other",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "w_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": -0.1, "y": 0.3}, "shape": "w_shape", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "orbit",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "MORE": {
        "gloss": "MORE",
        "description": "Both flat-O hands tap together",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.02, "y": 0.2}, "shape": "flat_o", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.02, "y": 0.2}, "shape": "flat_o", "palm": "right"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_together",
        "duration_ms": 800,
        "repeat": 2,
    },
    "STOP": {
        "gloss": "STOP",
        "description": "Flat dominant hand chops onto non-dominant palm",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.4}, "end": {"x": 0.0, "y": 0.15}, "shape": "flat", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.1}, "end": {"x": -0.05, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "serious",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "chop_down",
        "duration_ms": 600,
        "repeat": 1,
    },
    "EAT": {
        "gloss": "EAT",
        "description": "Flat-O hand taps mouth",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.45}, "end": {"x": 0.05, "y": 0.5}, "shape": "flat_o", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_mouth",
        "duration_ms": 800,
        "repeat": 2,
    },
    "DRINK": {
        "gloss": "DRINK",
        "description": "C-hand tips toward mouth",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.35}, "end": {"x": 0.1, "y": 0.5}, "shape": "c_shape", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "tilt_back",
        "body_movement": "none",
        "movement_type": "tip_to_mouth",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "WATER": {
        "gloss": "WATER",
        "description": "W-hand taps chin",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.45}, "end": {"x": 0.05, "y": 0.5}, "shape": "w_shape", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_chin",
        "duration_ms": 800,
        "repeat": 2,
    },
    "AGAIN": {
        "gloss": "AGAIN",
        "description": "Bent hand arcs and lands on non-dominant palm",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.0, "y": 0.1}, "shape": "bent", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.05}, "end": {"x": -0.05, "y": 0.05}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "arc_to_palm",
        "duration_ms": 900,
        "repeat": 1,
    },
    "TIME": {
        "gloss": "TIME",
        "description": "Index taps back of wrist (where watch would be)",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.15}, "end": {"x": -0.05, "y": 0.05}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "fist", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_wrist",
        "duration_ms": 800,
        "repeat": 1,
    },
    "TALK": {
        "gloss": "TALK",
        "description": "Index finger bounces from mouth outward",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.5}, "end": {"x": 0.25, "y": 0.45}, "shape": "index_point", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "bounce_from_mouth",
        "duration_ms": 800,
        "repeat": 2,
    },
    "TEACH": {
        "gloss": "TEACH",
        "description": "Both flat-O hands push forward from temples",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.3, "y": 0.4}, "shape": "flat_o", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.55}, "end": {"x": -0.3, "y": 0.4}, "shape": "flat_o", "palm": "out"},
        "expression": "engaged",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "push_from_temples",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "STUDENT": {
        "gloss": "STUDENT",
        "description": "LEARN + PERSON marker",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.0}, "end": {"x": 0.15, "y": 0.65}, "shape": "flat_closed", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": -0.05}, "end": {"x": -0.1, "y": -0.05}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "palm_to_forehead",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "TEACHER": {
        "gloss": "TEACHER",
        "description": "TEACH + PERSON marker",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.3, "y": 0.4}, "shape": "flat_o", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.55}, "end": {"x": -0.3, "y": 0.4}, "shape": "flat_o", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "push_from_temples",
        "duration_ms": 1200,
        "repeat": 1,
    },
    "START": {
        "gloss": "START",
        "description": "Dominant index twists between non-dominant V-fingers",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.1, "y": 0.25}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.15}, "end": {"x": -0.05, "y": 0.15}, "shape": "v_shape", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "twist_between",
        "duration_ms": 800,
        "repeat": 1,
    },
    "FOOD": {
        "gloss": "FOOD",
        "description": "Flat-O taps mouth (same as EAT)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.45}, "end": {"x": 0.05, "y": 0.5}, "shape": "flat_o", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_mouth",
        "duration_ms": 800,
        "repeat": 2,
    },
    "MONEY": {
        "gloss": "MONEY",
        "description": "Flat-O taps non-dominant palm",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.0, "y": 0.1}, "shape": "flat_o", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.05}, "end": {"x": -0.05, "y": 0.05}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_palm",
        "duration_ms": 800,
        "repeat": 2,
    },
    "CAN": {
        "gloss": "CAN",
        "description": "Both fists move down together",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.15, "y": 0.15}, "shape": "fist", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.15}, "shape": "fist", "palm": "down"},
        "expression": "confident",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "both_down",
        "duration_ms": 700,
        "repeat": 1,
    },
    "WILL": {
        "gloss": "WILL",
        "description": "Flat hand moves forward from side of face",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.4, "y": 0.45}, "shape": "flat", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "determined",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "face_forward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "DO": {
        "gloss": "DO",
        "description": "Both C-hands sway side to side",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.1}, "end": {"x": 0.25, "y": 0.1}, "shape": "c_shape", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.1}, "end": {"x": -0.25, "y": 0.1}, "shape": "c_shape", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "sway_side",
        "duration_ms": 900,
        "repeat": 1,
    },
    "MAKE": {
        "gloss": "MAKE",
        "description": "Fists twist on top of each other",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.2}, "end": {"x": 0.05, "y": 0.15}, "shape": "fist", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.1}, "end": {"x": -0.05, "y": 0.1}, "shape": "fist", "palm": "right"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "twist_on_fist",
        "duration_ms": 900,
        "repeat": 1,
    },
    "FEEL": {
        "gloss": "FEEL",
        "description": "Middle finger brushes up chest",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.1}, "end": {"x": 0.0, "y": 0.25}, "shape": "middle_up", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "thoughtful",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "brush_up",
        "duration_ms": 800,
        "repeat": 1,
    },

    # -----------------------------------------------------------------------
    # Meeting scenario signs — "A Deaf Colleague's First Team Meeting"
    # Greetings, introductions, workplace, discussion, emotions, scheduling
    # -----------------------------------------------------------------------

    "TEAM": {
        "gloss": "TEAM",
        "description": "Both T-hands circle inward forming a group",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.2, "y": 0.3}, "shape": "t_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.3}, "end": {"x": -0.2, "y": 0.3}, "shape": "t_shape", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "circle_together",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "PROJECT": {
        "gloss": "PROJECT",
        "description": "P-hand slides down non-dominant palm then pinky-side down",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.35}, "end": {"x": 0.1, "y": 0.1}, "shape": "p_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.1, "y": 0.2}, "shape": "flat", "palm": "side"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slide_down_palm",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "IDEA": {
        "gloss": "IDEA",
        "description": "I-hand rises from forehead upward",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.65}, "end": {"x": 0.25, "y": 0.85}, "shape": "i_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "bright",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "rise_from_forehead",
        "duration_ms": 900,
        "repeat": 1,
    },
    "PLAN": {
        "gloss": "PLAN",
        "description": "Both flat hands slide right in parallel",
        "dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": 0.3, "y": 0.3}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": 0.3, "y": 0.15}, "shape": "flat", "palm": "down"},
        "expression": "focused",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slide_parallel",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SUGGEST": {
        "gloss": "SUGGEST",
        "description": "Both flat hands rise upward offering",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.0}, "end": {"x": 0.2, "y": 0.3}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.0}, "end": {"x": -0.2, "y": 0.3}, "shape": "flat", "palm": "up"},
        "expression": "open",
        "head_movement": "slight_tilt",
        "body_movement": "slight_forward",
        "movement_type": "rise_offer",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "IMPORTANT": {
        "gloss": "IMPORTANT",
        "description": "F-hands rise from center upward",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.1}, "end": {"x": 0.0, "y": 0.4}, "shape": "f_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "emphasis",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "rise_center",
        "duration_ms": 900,
        "repeat": 1,
    },
    "EXPLAIN": {
        "gloss": "EXPLAIN",
        "description": "Both flat hands alternate moving forward",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.2}, "end": {"x": 0.35, "y": 0.2}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.2}, "end": {"x": -0.35, "y": 0.2}, "shape": "flat", "palm": "up"},
        "expression": "engaged",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "alternate_forward",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "MAYBE": {
        "gloss": "MAYBE",
        "description": "Both flat hands alternate tilting up and down",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.25}, "end": {"x": 0.2, "y": 0.35}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.35}, "end": {"x": -0.2, "y": 0.25}, "shape": "flat", "palm": "up"},
        "expression": "uncertain",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "alternate_tilt",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "PROBLEM": {
        "gloss": "PROBLEM",
        "description": "Both bent-V knuckles twist against each other",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.1, "y": 0.25}, "shape": "bent_v", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.25}, "end": {"x": -0.1, "y": 0.3}, "shape": "bent_v", "palm": "in"},
        "expression": "concerned",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "twist_knuckles",
        "duration_ms": 900,
        "repeat": 2,
    },
    "SOLUTION": {
        "gloss": "SOLUTION",
        "description": "S-hand opens to flat hand (like solving/unlocking)",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.3, "y": 0.35}, "shape": "fist_to_open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "relieved",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "fist_open",
        "duration_ms": 900,
        "repeat": 1,
    },
    "EXCITED": {
        "gloss": "EXCITED",
        "description": "Both middle fingers brush up chest alternately",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.05}, "end": {"x": 0.1, "y": 0.3}, "shape": "middle_up", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.35}, "shape": "middle_up", "palm": "in"},
        "expression": "big_smile",
        "head_movement": "nod",
        "body_movement": "slight_bounce",
        "movement_type": "alternate_brush_chest",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "WORRY": {
        "gloss": "WORRY",
        "description": "Both open hands circle in front of face alternately",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.5}, "end": {"x": 0.15, "y": 0.5}, "shape": "open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.5}, "end": {"x": -0.15, "y": 0.5}, "shape": "open", "palm": "in"},
        "expression": "worried",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "alternate_circle_face",
        "duration_ms": 1200,
        "repeat": 2,
    },
    "CONFIDENT": {
        "gloss": "CONFIDENT",
        "description": "Both curved hands pull down from chest firmly",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.35}, "end": {"x": 0.1, "y": 0.1}, "shape": "curved", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.35}, "end": {"x": -0.1, "y": 0.1}, "shape": "curved", "palm": "in"},
        "expression": "confident",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "pull_down_chest",
        "duration_ms": 900,
        "repeat": 1,
    },
    "NEXT": {
        "gloss": "NEXT",
        "description": "Dominant flat hand arcs over non-dominant flat hand",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.15}, "end": {"x": 0.1, "y": 0.3}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.15}, "shape": "flat", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "arc_over",
        "duration_ms": 800,
        "repeat": 1,
    },
    "SCHEDULE": {
        "gloss": "SCHEDULE",
        "description": "Dominant open hand slides across non-dominant palm then down",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.25}, "end": {"x": -0.1, "y": 0.0}, "shape": "open", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.15}, "shape": "flat", "palm": "side"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slide_across_down",
        "duration_ms": 1100,
        "repeat": 1,
    },
    "WELCOME": {
        "gloss": "WELCOME",
        "description": "Open hand sweeps inward invitingly",
        "dominant_hand": {"start": {"x": 0.4, "y": 0.3}, "end": {"x": 0.1, "y": 0.2}, "shape": "open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "warm_smile",
        "head_movement": "nod",
        "body_movement": "slight_forward",
        "movement_type": "sweep_inward",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "INTRODUCE": {
        "gloss": "INTRODUCE",
        "description": "Both flat hands bring together from sides to center",
        "dominant_hand": {"start": {"x": 0.35, "y": 0.2}, "end": {"x": 0.05, "y": 0.2}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.35, "y": 0.2}, "end": {"x": -0.05, "y": 0.2}, "shape": "flat", "palm": "up"},
        "expression": "smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "bring_together",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "PRESENT": {
        "gloss": "PRESENT",
        "description": "Both flat hands push forward and outward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.3, "y": 0.3}, "shape": "flat", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.3, "y": 0.3}, "shape": "flat", "palm": "up"},
        "expression": "engaged",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "push_forward_out",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SHARE": {
        "gloss": "SHARE",
        "description": "Flat hand slices between non-dominant thumb and index",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.0, "y": 0.15}, "shape": "flat", "palm": "side"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.1}, "end": {"x": -0.05, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "open",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slice_between",
        "duration_ms": 800,
        "repeat": 1,
    },
    "MEETING": {
        "gloss": "MEETING",
        "description": "Both open hands come together, fingers touch and separate",
        "dominant_hand": {"start": {"x": 0.25, "y": 0.3}, "end": {"x": 0.05, "y": 0.3}, "shape": "open_spread", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.25, "y": 0.3}, "end": {"x": -0.05, "y": 0.3}, "shape": "open_spread", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "fingers_meet_separate",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "COMPUTER": {
        "gloss": "COMPUTER",
        "description": "C-hand bounces up non-dominant arm",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.2}, "shape": "c_shape", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.0}, "end": {"x": -0.15, "y": 0.2}, "shape": "flat", "palm": "down"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "bounce_up_arm",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "EMAIL": {
        "gloss": "EMAIL",
        "description": "Index slides into C-hand (mail into box)",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.0, "y": 0.2}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.15}, "end": {"x": -0.05, "y": 0.15}, "shape": "c_shape", "palm": "right"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "slide_into",
        "duration_ms": 800,
        "repeat": 1,
    },
    "OFFICE": {
        "gloss": "OFFICE",
        "description": "Both O-hands form a box shape",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.25}, "end": {"x": 0.15, "y": 0.1}, "shape": "o_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.25}, "end": {"x": -0.15, "y": 0.1}, "shape": "o_shape", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "form_box",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "NEW": {
        "gloss": "NEW",
        "description": "Curved hand scoops across non-dominant palm",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.15}, "end": {"x": -0.1, "y": 0.1}, "shape": "curved", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.05}, "end": {"x": -0.1, "y": 0.05}, "shape": "flat", "palm": "up"},
        "expression": "bright",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "scoop_across",
        "duration_ms": 800,
        "repeat": 1,
    },
    "FIRST": {
        "gloss": "FIRST",
        "description": "Index finger strikes thumb of A-hand",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.05, "y": 0.2}, "shape": "index_point", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.15}, "end": {"x": -0.05, "y": 0.15}, "shape": "thumb_up", "palm": "right"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "strike_thumb",
        "duration_ms": 700,
        "repeat": 1,
    },
    "EXPERIENCE": {
        "gloss": "EXPERIENCE",
        "description": "Open hand pulls from cheek closing into flat-O",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.55}, "end": {"x": 0.3, "y": 0.45}, "shape": "open_to_flat_o", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "thoughtful",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pull_from_cheek",
        "duration_ms": 900,
        "repeat": 1,
    },
    "TOGETHER": {
        "gloss": "TOGETHER",
        "description": "Both A-hands come together and circle",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.2}, "end": {"x": 0.15, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.2}, "end": {"x": -0.15, "y": 0.2}, "shape": "fist", "palm": "in"},
        "expression": "warm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "circle_joined",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SUPPORT": {
        "gloss": "SUPPORT",
        "description": "Dominant fist pushes up under non-dominant fist",
        "dominant_hand": {"start": {"x": 0.0, "y": -0.05}, "end": {"x": 0.0, "y": 0.15}, "shape": "fist", "palm": "up"},
        "non_dominant_hand": {"start": {"x": 0.0, "y": 0.15}, "end": {"x": 0.0, "y": 0.25}, "shape": "fist", "palm": "down"},
        "expression": "supportive",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "push_up_under",
        "duration_ms": 900,
        "repeat": 1,
    },
    "TRY": {
        "gloss": "TRY",
        "description": "Both T-hands push forward and down",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.25}, "end": {"x": 0.25, "y": 0.1}, "shape": "t_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.25}, "end": {"x": -0.25, "y": 0.1}, "shape": "t_shape", "palm": "in"},
        "expression": "determined",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "push_forward_down",
        "duration_ms": 800,
        "repeat": 1,
    },
    "IMPROVE": {
        "gloss": "IMPROVE",
        "description": "Flat hand hops up non-dominant arm in stages",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.0}, "end": {"x": -0.05, "y": 0.3}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.3}, "shape": "flat", "palm": "side"},
        "expression": "hopeful",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "hop_up_arm",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "HOPE": {
        "gloss": "HOPE",
        "description": "Both flat hands near forehead, bend forward",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.6}, "end": {"x": 0.15, "y": 0.55}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.6}, "end": {"x": -0.15, "y": 0.55}, "shape": "flat", "palm": "in"},
        "expression": "hopeful",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "bend_forward_forehead",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "PROUD": {
        "gloss": "PROUD",
        "description": "Thumb slides up chest",
        "dominant_hand": {"start": {"x": 0.0, "y": -0.1}, "end": {"x": 0.0, "y": 0.3}, "shape": "thumb_up", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "proud",
        "head_movement": "chin_up",
        "body_movement": "none",
        "movement_type": "slide_up_chest",
        "duration_ms": 900,
        "repeat": 1,
    },
    "WONDERFUL": {
        "gloss": "WONDERFUL",
        "description": "Both open hands push outward from face",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.5}, "end": {"x": 0.3, "y": 0.4}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.5}, "end": {"x": -0.3, "y": 0.4}, "shape": "open", "palm": "out"},
        "expression": "big_smile",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "push_outward",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "GREAT": {
        "gloss": "GREAT",
        "description": "Both open hands arc up and outward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.35, "y": 0.5}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": -0.35, "y": 0.5}, "shape": "open", "palm": "out"},
        "expression": "big_smile",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "arc_up_outward",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "SAME": {
        "gloss": "SAME",
        "description": "Both index fingers come together side by side",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.05, "y": 0.2}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.2}, "end": {"x": -0.05, "y": 0.2}, "shape": "index_point", "palm": "down"},
        "expression": "neutral",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "come_together_parallel",
        "duration_ms": 800,
        "repeat": 1,
    },
    "DIFFERENT": {
        "gloss": "DIFFERENT",
        "description": "Both index fingers cross then pull apart",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.2}, "end": {"x": 0.3, "y": 0.25}, "shape": "index_point", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.2}, "end": {"x": -0.3, "y": 0.25}, "shape": "index_point", "palm": "out"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "cross_and_separate",
        "duration_ms": 900,
        "repeat": 1,
    },
    "DIFFICULT": {
        "gloss": "DIFFICULT",
        "description": "Both bent-V hands alternately bob up and down",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.25}, "end": {"x": 0.15, "y": 0.3}, "shape": "bent_v", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.25}, "shape": "bent_v", "palm": "in"},
        "expression": "strained",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "alternate_bob",
        "duration_ms": 1000,
        "repeat": 2,
    },
    "EASY": {
        "gloss": "EASY",
        "description": "Curved hand brushes up fingertips of non-dominant hand",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.1}, "end": {"x": 0.1, "y": 0.25}, "shape": "curved", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.15}, "shape": "curved", "palm": "up"},
        "expression": "relaxed",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "brush_up_fingertips",
        "duration_ms": 800,
        "repeat": 2,
    },
    "POSSIBLE": {
        "gloss": "POSSIBLE",
        "description": "Both S-hands move down firmly (same as CAN)",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.15, "y": 0.15}, "shape": "fist", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.3}, "end": {"x": -0.15, "y": 0.15}, "shape": "fist", "palm": "down"},
        "expression": "hopeful",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "both_down",
        "duration_ms": 800,
        "repeat": 1,
    },
    "READY": {
        "gloss": "READY",
        "description": "Both R-hands move from center outward",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.2}, "end": {"x": 0.25, "y": 0.2}, "shape": "r_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.2}, "end": {"x": -0.25, "y": 0.2}, "shape": "r_shape", "palm": "out"},
        "expression": "confident",
        "head_movement": "nod",
        "body_movement": "none",
        "movement_type": "spread_outward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "DISCUSS": {
        "gloss": "DISCUSS",
        "description": "Index finger taps non-dominant palm repeatedly",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.3}, "end": {"x": 0.0, "y": 0.15}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.1}, "end": {"x": -0.05, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "engaged",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_palm_repeated",
        "duration_ms": 900,
        "repeat": 2,
    },
    "OPINION": {
        "gloss": "OPINION",
        "description": "O-hand moves away from forehead",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.65}, "end": {"x": 0.25, "y": 0.5}, "shape": "o_shape", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "thoughtful",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "forehead_outward",
        "duration_ms": 800,
        "repeat": 1,
    },
    "FOCUS": {
        "gloss": "FOCUS",
        "description": "Both F-hands move from sides of eyes forward",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.55}, "end": {"x": 0.2, "y": 0.35}, "shape": "f_shape", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.15, "y": 0.55}, "end": {"x": -0.2, "y": 0.35}, "shape": "f_shape", "palm": "out"},
        "expression": "focused",
        "head_movement": "none",
        "body_movement": "slight_forward",
        "movement_type": "eyes_forward",
        "duration_ms": 900,
        "repeat": 1,
    },
    "HOW": {
        "gloss": "HOW",
        "description": "Both bent hands knuckles touch then roll open",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.2, "y": 0.3}, "shape": "bent_to_open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.2, "y": 0.3}, "shape": "bent_to_open", "palm": "up"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "none",
        "movement_type": "knuckle_roll_open",
        "duration_ms": 900,
        "repeat": 1,
    },
    "WHO": {
        "gloss": "WHO",
        "description": "Index circles near mouth",
        "dominant_hand": {"start": {"x": 0.05, "y": 0.5}, "end": {"x": 0.05, "y": 0.5}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "none",
        "movement_type": "circle_mouth",
        "duration_ms": 800,
        "repeat": 1,
    },
    "WHY": {
        "gloss": "WHY",
        "description": "Fingertips touch forehead, pull away into Y-hand",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.65}, "end": {"x": 0.25, "y": 0.45}, "shape": "open_to_y", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "none",
        "movement_type": "forehead_to_y",
        "duration_ms": 900,
        "repeat": 1,
    },
    "WHEN": {
        "gloss": "WHEN",
        "description": "Index circles then lands on non-dominant index tip",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.35}, "end": {"x": 0.0, "y": 0.25}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.05, "y": 0.2}, "end": {"x": -0.05, "y": 0.2}, "shape": "index_point", "palm": "up"},
        "expression": "brow_raise",
        "head_movement": "tilt_forward",
        "body_movement": "none",
        "movement_type": "circle_land_tip",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "THEY": {
        "gloss": "THEY",
        "description": "Index sweeps across from left to right",
        "dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": 0.4, "y": 0.3}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "sweep_across",
        "duration_ms": 800,
        "repeat": 1,
    },
    "HE": {
        "gloss": "HE",
        "description": "Point forward from side of forehead (male marker + point)",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.65}, "end": {"x": 0.35, "y": 0.4}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "forehead_point",
        "duration_ms": 700,
        "repeat": 1,
    },
    "SHE": {
        "gloss": "SHE",
        "description": "Point forward from side of chin (female marker + point)",
        "dominant_hand": {"start": {"x": 0.15, "y": 0.5}, "end": {"x": 0.35, "y": 0.4}, "shape": "index_point", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "chin_point",
        "duration_ms": 700,
        "repeat": 1,
    },
    "ANGRY": {
        "gloss": "ANGRY",
        "description": "Claw hand pulls away from face",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.55}, "end": {"x": 0.15, "y": 0.4}, "shape": "claw", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "angry",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pull_from_face",
        "duration_ms": 800,
        "repeat": 1,
    },
    "SCARED": {
        "gloss": "SCARED",
        "description": "Both fists open in front of chest (startled)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.2, "y": 0.35}, "shape": "fist_to_open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.2, "y": 0.35}, "shape": "fist_to_open", "palm": "in"},
        "expression": "scared",
        "head_movement": "pull_back",
        "body_movement": "lean_back",
        "movement_type": "startle_open",
        "duration_ms": 800,
        "repeat": 1,
    },
    "TIRED": {
        "gloss": "TIRED",
        "description": "Both bent hands on chest, fall open/downward",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.25}, "end": {"x": 0.1, "y": 0.1}, "shape": "bent", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.25}, "end": {"x": -0.1, "y": 0.1}, "shape": "bent", "palm": "in"},
        "expression": "tired",
        "head_movement": "drop",
        "body_movement": "slight_forward",
        "movement_type": "drop_from_chest",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "HUNGRY": {
        "gloss": "HUNGRY",
        "description": "C-hand moves down from throat to stomach",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.4}, "end": {"x": 0.0, "y": 0.0}, "shape": "c_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "longing",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "throat_to_stomach",
        "duration_ms": 900,
        "repeat": 1,
    },
    "MORNING": {
        "gloss": "MORNING",
        "description": "Non-dominant flat arm as horizon, dominant hand rises like sun",
        "dominant_hand": {"start": {"x": 0.1, "y": -0.1}, "end": {"x": 0.1, "y": 0.3}, "shape": "flat", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.0}, "end": {"x": 0.2, "y": 0.0}, "shape": "flat", "palm": "down"},
        "expression": "warm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "sunrise",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "NIGHT": {
        "gloss": "NIGHT",
        "description": "Dominant bent hand drops over non-dominant arm (sun setting)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.1, "y": 0.0}, "shape": "bent", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.0}, "end": {"x": 0.2, "y": 0.0}, "shape": "flat", "palm": "down"},
        "expression": "calm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "sunset",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "READ": {
        "gloss": "READ",
        "description": "V-hand moves down non-dominant palm (eyes scanning page)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.1, "y": 0.0}, "shape": "v_shape", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.15}, "end": {"x": -0.1, "y": 0.15}, "shape": "flat", "palm": "up"},
        "expression": "focused",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "scan_down",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "WRITE": {
        "gloss": "WRITE",
        "description": "Pinched hand writes on non-dominant palm",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": -0.05, "y": 0.1}, "shape": "pinch", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.05}, "end": {"x": -0.1, "y": 0.05}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "write_on_palm",
        "duration_ms": 1000,
        "repeat": 1,
    },
    "HEAR": {
        "gloss": "HEAR",
        "description": "Index finger points to ear",
        "dominant_hand": {"start": {"x": 0.25, "y": 0.55}, "end": {"x": 0.25, "y": 0.6}, "shape": "index_point", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "attentive",
        "head_movement": "slight_tilt",
        "body_movement": "none",
        "movement_type": "point_ear",
        "duration_ms": 700,
        "repeat": 1,
    },
    "CHILD": {
        "gloss": "CHILD",
        "description": "Flat hand pats downward (indicating small height)",
        "dominant_hand": {"start": {"x": 0.25, "y": 0.1}, "end": {"x": 0.25, "y": -0.05}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "warm",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "pat_down",
        "duration_ms": 800,
        "repeat": 2,
    },
    "MAN": {
        "gloss": "MAN",
        "description": "Thumb of open hand touches forehead then chest",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.65}, "end": {"x": 0.1, "y": 0.2}, "shape": "open_thumb", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "forehead_to_chest",
        "duration_ms": 900,
        "repeat": 1,
    },
    "WOMAN": {
        "gloss": "WOMAN",
        "description": "Thumb of open hand touches chin then chest",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.5}, "end": {"x": 0.1, "y": 0.2}, "shape": "open_thumb", "palm": "left"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "chin_to_chest",
        "duration_ms": 900,
        "repeat": 1,
    },
    "DOCTOR": {
        "gloss": "DOCTOR",
        "description": "D-hand taps non-dominant wrist (pulse check)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.1}, "end": {"x": -0.05, "y": 0.05}, "shape": "d_shape", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.0}, "shape": "flat", "palm": "up"},
        "expression": "neutral",
        "head_movement": "none",
        "body_movement": "none",
        "movement_type": "tap_wrist",
        "duration_ms": 800,
        "repeat": 2,
    },
}

# Fallback animation for unknown signs (fingerspelling placeholder)
UNKNOWN_SIGN_ANIMATION: Dict[str, Any] = {
    "gloss": "UNKNOWN",
    "description": "Fingerspelling placeholder — hand raised in neutral position",
    "dominant_hand": {"start": {"x": 0.3, "y": 0.4}, "end": {"x": 0.3, "y": 0.45}, "shape": "open", "palm": "out"},
    "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
    "expression": "neutral",
    "head_movement": "none",
    "body_movement": "none",
    "movement_type": "fingerspell",
    "duration_ms": 1500,
    "repeat": 1,
}


# ---------------------------------------------------------------------------
# KSL Sign Vocabulary — Animation Data (Korean Sign Language)
# Same structure as SIGN_ANIMATIONS. Positions differ because KSL signs
# have different handshapes and movements from ASL.
# ---------------------------------------------------------------------------

KSL_SIGN_ANIMATIONS: Dict[str, Dict[str, Any]] = {
    "안녕": {
        "gloss": "안녕", "description": "오른손을 이마 옆에서 앞으로 펼침",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.7}, "end": {"x": 0.4, "y": 0.6}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile", "head_movement": "nod", "body_movement": "slight_forward",
        "movement_type": "wave", "duration_ms": 1000, "repeat": 1,
    },
    "감사": {
        "gloss": "감사", "description": "오른손을 왼손 위에 얹고 고개 숙임",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.2}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.1}, "end": {"x": -0.1, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "smile", "head_movement": "bow", "body_movement": "slight_forward",
        "movement_type": "place", "duration_ms": 1200, "repeat": 1,
    },
    "네": {
        "gloss": "네", "description": "주먹을 앞으로 내려찍듯 끄덕임",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.4}, "end": {"x": 0.2, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm", "head_movement": "nod", "body_movement": "none",
        "movement_type": "arc", "duration_ms": 600, "repeat": 2,
    },
    "아니오": {
        "gloss": "아니오", "description": "검지를 좌우로 흔듦",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.3, "y": 0.5}, "shape": "index", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "head_shake", "head_movement": "shake", "body_movement": "none",
        "movement_type": "wave", "duration_ms": 800, "repeat": 2,
    },
    "도움": {
        "gloss": "도움", "description": "엄지를 세운 주먹을 반대 손바닥 위에 올림",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.1}, "end": {"x": 0.1, "y": 0.3}, "shape": "thumbs_up", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "concerned", "head_movement": "none", "body_movement": "none",
        "movement_type": "lift", "duration_ms": 1000, "repeat": 1,
    },
    "미안": {
        "gloss": "미안", "description": "주먹으로 가슴을 원형으로 문지름",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.3}, "end": {"x": 0.1, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "frown", "head_movement": "bow", "body_movement": "slight_forward",
        "movement_type": "circular", "duration_ms": 1000, "repeat": 1,
    },
    "좋다": {
        "gloss": "좋다", "description": "엄지를 세워 올림",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.2, "y": 0.4}, "shape": "thumbs_up", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile", "head_movement": "nod", "body_movement": "none",
        "movement_type": "lift", "duration_ms": 800, "repeat": 1,
    },
    "나쁘다": {
        "gloss": "나쁘다", "description": "엄지를 아래로 내림",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.4}, "end": {"x": 0.2, "y": 0.1}, "shape": "thumbs_down", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "frown", "head_movement": "shake", "body_movement": "none",
        "movement_type": "drop", "duration_ms": 800, "repeat": 1,
    },
    "나": {
        "gloss": "나", "description": "검지로 자기 가슴을 가리킴",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "point", "duration_ms": 500, "repeat": 1,
    },
    "너": {
        "gloss": "너", "description": "검지로 상대를 가리킴",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.4, "y": 0.3}, "shape": "index", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "point", "duration_ms": 500, "repeat": 1,
    },
    "이름": {
        "gloss": "이름", "description": "양손 검지와 중지를 교차",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.4}, "end": {"x": 0.1, "y": 0.3}, "shape": "two_fingers", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.4}, "end": {"x": -0.1, "y": 0.3}, "shape": "two_fingers", "palm": "up"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "tap", "duration_ms": 800, "repeat": 2,
    },
    "만나다": {
        "gloss": "만나다", "description": "양손 검지를 세워 서로 다가감",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.3}, "end": {"x": -0.1, "y": 0.3}, "shape": "index", "palm": "in"},
        "expression": "smile", "head_movement": "none", "body_movement": "none",
        "movement_type": "converge", "duration_ms": 900, "repeat": 1,
    },
    "반갑다": {
        "gloss": "반갑다", "description": "양손을 맞잡고 흔듦",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.2}, "end": {"x": 0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.2}, "end": {"x": -0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "expression": "big_smile", "head_movement": "nod", "body_movement": "slight_forward",
        "movement_type": "shake", "duration_ms": 1000, "repeat": 2,
    },
    "사랑": {
        "gloss": "사랑", "description": "양팔을 교차하여 가슴에 안음",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.2}, "end": {"x": -0.1, "y": 0.3}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.2}, "end": {"x": 0.1, "y": 0.3}, "shape": "fist", "palm": "in"},
        "expression": "smile", "head_movement": "none", "body_movement": "none",
        "movement_type": "cross", "duration_ms": 1200, "repeat": 1,
    },
    "가다": {
        "gloss": "가다", "description": "검지를 앞으로 향함",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.5, "y": 0.3}, "shape": "index", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "forward", "duration_ms": 700, "repeat": 1,
    },
    "오다": {
        "gloss": "오다", "description": "검지를 자기 쪽으로 당김",
        "dominant_hand": {"start": {"x": 0.5, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "index", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "pull", "duration_ms": 700, "repeat": 1,
    },
    "먹다": {
        "gloss": "먹다", "description": "모은 손가락을 입으로 가져감",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.0, "y": 0.6}, "shape": "pinch", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "arc", "duration_ms": 800, "repeat": 2,
    },
    "마시다": {
        "gloss": "마시다", "description": "C자 손을 입으로 기울임",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.1, "y": 0.6}, "shape": "c_shape", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "tilt_back", "body_movement": "none",
        "movement_type": "tilt", "duration_ms": 900, "repeat": 1,
    },
    "학교": {
        "gloss": "학교", "description": "양손을 두 번 마주침 (박수 형태)",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "flat", "palm": "up"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "clap", "duration_ms": 800, "repeat": 2,
    },
    "회의": {
        "gloss": "회의", "description": "양손을 펼쳐 서로 마주보게 모음",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.3}, "end": {"x": -0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "converge", "duration_ms": 1000, "repeat": 1,
    },
    "이해하다": {
        "gloss": "이해하다", "description": "검지를 이마에서 튕김",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.7}, "end": {"x": 0.2, "y": 0.8}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm", "head_movement": "nod", "body_movement": "none",
        "movement_type": "flick", "duration_ms": 700, "repeat": 1,
    },
    "뭐": {
        "gloss": "뭐", "description": "양손을 펼쳐 좌우로 흔듦",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.3, "y": 0.3}, "shape": "open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.3}, "end": {"x": -0.3, "y": 0.3}, "shape": "open", "palm": "up"},
        "expression": "brow_raise", "head_movement": "tilt", "body_movement": "none",
        "movement_type": "wave", "duration_ms": 800, "repeat": 2,
    },
}


# ---------------------------------------------------------------------------
# TSL Sign Vocabulary — Animation Data (Taiwan Sign Language / 台灣手語)
# TSL belongs to JSL family. Signs differ from both ASL and KSL.
# ---------------------------------------------------------------------------

TSL_SIGN_ANIMATIONS: Dict[str, Dict[str, Any]] = {
    "你好": {
        "gloss": "你好", "description": "右手在額前揮動",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.7}, "end": {"x": 0.5, "y": 0.7}, "shape": "open", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile", "head_movement": "nod", "body_movement": "slight_forward",
        "movement_type": "wave", "duration_ms": 1000, "repeat": 1,
    },
    "謝謝": {
        "gloss": "謝謝", "description": "右手平放從下巴前方向前推出",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.3, "y": 0.4}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile", "head_movement": "bow", "body_movement": "slight_forward",
        "movement_type": "push", "duration_ms": 1000, "repeat": 1,
    },
    "是": {
        "gloss": "是", "description": "拳頭向下點",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.4}, "end": {"x": 0.2, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm", "head_movement": "nod", "body_movement": "none",
        "movement_type": "arc", "duration_ms": 600, "repeat": 1,
    },
    "不是": {
        "gloss": "不是", "description": "食指左右搖擺",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.5}, "end": {"x": 0.3, "y": 0.5}, "shape": "index", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "head_shake", "head_movement": "shake", "body_movement": "none",
        "movement_type": "wave", "duration_ms": 800, "repeat": 2,
    },
    "我": {
        "gloss": "我", "description": "食指指向自己胸口",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "point", "duration_ms": 500, "repeat": 1,
    },
    "你": {
        "gloss": "你", "description": "食指指向對方",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.4, "y": 0.3}, "shape": "index", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "point", "duration_ms": 500, "repeat": 1,
    },
    "名字": {
        "gloss": "名字", "description": "雙手食指中指交叉輕拍",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.4}, "end": {"x": 0.1, "y": 0.3}, "shape": "two_fingers", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.4}, "end": {"x": -0.1, "y": 0.3}, "shape": "two_fingers", "palm": "up"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "tap", "duration_ms": 800, "repeat": 2,
    },
    "見面": {
        "gloss": "見面", "description": "雙手食指相對靠近",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.3}, "end": {"x": -0.1, "y": 0.3}, "shape": "index", "palm": "in"},
        "expression": "smile", "head_movement": "none", "body_movement": "none",
        "movement_type": "converge", "duration_ms": 900, "repeat": 1,
    },
    "高興": {
        "gloss": "高興", "description": "雙手在胸前向上揮動",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.2, "y": 0.5}, "shape": "open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.2}, "end": {"x": -0.2, "y": 0.5}, "shape": "open", "palm": "up"},
        "expression": "big_smile", "head_movement": "nod", "body_movement": "none",
        "movement_type": "lift", "duration_ms": 900, "repeat": 2,
    },
    "幫助": {
        "gloss": "幫助", "description": "拇指豎起放在另一手掌上抬起",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.1}, "end": {"x": 0.1, "y": 0.3}, "shape": "thumbs_up", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.0}, "end": {"x": -0.1, "y": 0.1}, "shape": "flat", "palm": "up"},
        "expression": "concerned", "head_movement": "none", "body_movement": "none",
        "movement_type": "lift", "duration_ms": 1000, "repeat": 1,
    },
    "對不起": {
        "gloss": "對不起", "description": "拳頭在胸前畫圈",
        "dominant_hand": {"start": {"x": 0.0, "y": 0.3}, "end": {"x": 0.1, "y": 0.2}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "frown", "head_movement": "bow", "body_movement": "slight_forward",
        "movement_type": "circular", "duration_ms": 1000, "repeat": 1,
    },
    "好": {
        "gloss": "好", "description": "拇指豎起",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.2}, "end": {"x": 0.2, "y": 0.4}, "shape": "thumbs_up", "palm": "out"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "smile", "head_movement": "nod", "body_movement": "none",
        "movement_type": "lift", "duration_ms": 800, "repeat": 1,
    },
    "去": {
        "gloss": "去", "description": "食指向前指出",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.5, "y": 0.3}, "shape": "index", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "forward", "duration_ms": 700, "repeat": 1,
    },
    "來": {
        "gloss": "來", "description": "食指向自己勾回",
        "dominant_hand": {"start": {"x": 0.5, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "index", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "pull", "duration_ms": 700, "repeat": 1,
    },
    "吃": {
        "gloss": "吃", "description": "手指併攏向嘴移動",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.0, "y": 0.6}, "shape": "pinch", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "arc", "duration_ms": 800, "repeat": 2,
    },
    "學校": {
        "gloss": "學校", "description": "雙手拍掌",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "flat", "palm": "down"},
        "non_dominant_hand": {"start": {"x": -0.1, "y": 0.3}, "end": {"x": 0.0, "y": 0.3}, "shape": "flat", "palm": "up"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "clap", "duration_ms": 800, "repeat": 2,
    },
    "什麼": {
        "gloss": "什麼", "description": "雙手攤開左右搖",
        "dominant_hand": {"start": {"x": 0.2, "y": 0.3}, "end": {"x": 0.3, "y": 0.3}, "shape": "open", "palm": "up"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": 0.3}, "end": {"x": -0.3, "y": 0.3}, "shape": "open", "palm": "up"},
        "expression": "brow_raise", "head_movement": "tilt", "body_movement": "none",
        "movement_type": "wave", "duration_ms": 800, "repeat": 2,
    },
    "會議": {
        "gloss": "會議", "description": "雙手張開相對移近",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.3}, "end": {"x": 0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.3}, "end": {"x": -0.1, "y": 0.3}, "shape": "open", "palm": "in"},
        "expression": "neutral", "head_movement": "none", "body_movement": "none",
        "movement_type": "converge", "duration_ms": 1000, "repeat": 1,
    },
    "愛": {
        "gloss": "愛", "description": "雙臂交叉抱胸",
        "dominant_hand": {"start": {"x": 0.3, "y": 0.2}, "end": {"x": -0.1, "y": 0.3}, "shape": "fist", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.3, "y": 0.2}, "end": {"x": 0.1, "y": 0.3}, "shape": "fist", "palm": "in"},
        "expression": "smile", "head_movement": "none", "body_movement": "none",
        "movement_type": "cross", "duration_ms": 1200, "repeat": 1,
    },
    "了解": {
        "gloss": "了解", "description": "食指從額頭彈出",
        "dominant_hand": {"start": {"x": 0.1, "y": 0.7}, "end": {"x": 0.2, "y": 0.8}, "shape": "index", "palm": "in"},
        "non_dominant_hand": {"start": {"x": -0.2, "y": -0.2}, "end": {"x": -0.2, "y": -0.2}, "shape": "rest", "palm": "in"},
        "expression": "nod_affirm", "head_movement": "nod", "body_movement": "none",
        "movement_type": "flick", "duration_ms": 700, "repeat": 1,
    },
}


# Animation registry for multi-language support
SIGN_ANIMATION_DICTS = {
    "asl": SIGN_ANIMATIONS,
    "ksl": KSL_SIGN_ANIMATIONS,
    "tsl": TSL_SIGN_ANIMATIONS,
}


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

def text_to_gloss(text: str, language: str = "asl") -> List[str]:
    """Convert text to sign language gloss sequence using Azure OpenAI.

    Parameters
    ----------
    text : str
        Text to convert. Language of input should match the target sign language:
        - "asl": English text → ASL gloss
        - "ksl": Korean text → KSL gloss
        - "tsl": Traditional Chinese text → TSL gloss
    language : str
        Target sign language: "asl", "ksl", or "tsl" (default: "asl")

    Returns
    -------
    list of str
        Sign language gloss sequence.
    """
    if not text or not text.strip():
        return []

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    # Select system prompt based on target sign language
    system_prompt = GLOSS_SYSTEM_PROMPTS.get(language, ASL_GLOSS_SYSTEM_PROMPT)

    try:
        client = _get_client()

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()

        # The model may return a JSON array directly or wrapped in an object
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            gloss_sequence = parsed
        elif isinstance(parsed, dict):
            # Try common keys the model might use
            for key in ("gloss", "sequence", "signs", "result", "output"):
                if key in parsed and isinstance(parsed[key], list):
                    gloss_sequence = parsed[key]
                    break
            else:
                # Take the first list value found
                for v in parsed.values():
                    if isinstance(v, list):
                        gloss_sequence = v
                        break
                else:
                    gloss_sequence = [str(v) for v in parsed.values()]
        else:
            gloss_sequence = [str(parsed)]

        # Normalize: ASL uses uppercase, KSL/TSL keep original script
        if language == "asl":
            gloss_sequence = [g.strip().upper() for g in gloss_sequence if g and str(g).strip()]
        else:
            gloss_sequence = [g.strip() for g in gloss_sequence if g and str(g).strip()]

        logger.info("Text->Gloss [%s]: '%s' -> %s", language, text, gloss_sequence)
        return gloss_sequence

    except EnvironmentError:
        # Azure OpenAI not configured — use rule-based fallback
        logger.warning("Azure OpenAI not configured. Using rule-based gloss conversion.")
        return _rule_based_gloss(text)

    except Exception as exc:
        logger.error("Azure OpenAI gloss conversion failed: %s. Falling back to rules.", exc)
        return _rule_based_gloss(text)


def _rule_based_gloss(text: str) -> List[str]:
    """Simple rule-based fallback for text to ASL gloss when OpenAI is unavailable.

    Applies basic ASL grammar rules:
    - Remove articles and copulas
    - Uppercase all words
    - Map common phrases to gloss
    """
    # Common phrase mappings (prioritized for team meeting scenarios)
    phrase_map = {
        "nice to meet you": ["NICE", "MEET", "YOU"],
        "i love you": ["I-LOVE-YOU"],
        "thank you": ["THANK-YOU"],
        "slow down": ["SLOW-DOWN"],
        "how are you": ["YOU", "HOW"],
        "what is your name": ["YOUR", "NAME", "WHAT"],
        "i don't understand": ["I", "UNDERSTAND", "NOT"],
        "i don't know": ["I", "KNOW", "NOT"],
        "see you later": ["SEE", "YOU", "LATER"],
        "good morning": ["GOOD", "MORNING"],
        "good night": ["GOOD", "NIGHT"],
        # Meeting-specific phrases
        "welcome to the team": ["WELCOME", "TEAM"],
        "let me introduce": ["I", "INTRODUCE"],
        "can you repeat": ["YOU", "REPEAT", "CAN"],
        "i have a question": ["I", "QUESTION", "HAVE"],
        "i agree with you": ["I", "AGREE", "YOU"],
        "good idea": ["GOOD", "IDEA"],
        "great job": ["GREAT", "WORK"],
        "let's discuss": ["WE", "DISCUSS"],
        "next meeting": ["NEXT", "MEETING"],
        "any questions": ["QUESTION", "HAVE"],
        "i'm excited": ["I", "EXCITED"],
        "i'm proud": ["I", "PROUD"],
        "we can do it": ["WE", "CAN", "DO"],
        "i'll try": ["I", "TRY", "WILL"],
        "no problem": ["PROBLEM", "NOT"],
        "what do you think": ["YOU", "THINK", "WHAT"],
        "let's work together": ["WE", "WORK", "TOGETHER"],
        "i'm ready": ["I", "READY"],
        "sounds good": ["GOOD"],
        "i'm new here": ["I", "NEW"],
        "first day": ["FIRST", "TODAY"],
        "sign language": ["SIGN", "LANGUAGE"],
    }

    lower = text.lower().strip()

    # Check phrase mappings first
    for phrase, gloss in phrase_map.items():
        if phrase in lower:
            return gloss

    # Remove articles and copulas
    remove_words = {
        "a", "an", "the", "am", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "at", "in", "on",
        "for", "with", "it", "that", "this", "do", "does", "did",
    }

    words = lower.split()
    filtered = [w.upper() for w in words if w not in remove_words]

    # Map common contractions
    result = []
    for w in filtered:
        if w == "DON'T" or w == "DONT":
            result.append("NOT")
        elif w == "CAN'T" or w == "CANT":
            result.extend(["CAN", "NOT"])
        elif w == "WON'T" or w == "WONT":
            result.extend(["WILL", "NOT"])
        elif w == "I'M":
            result.append("I")
        elif w == "GOING":
            result.append("GO")
        elif w == "WANTED" or w == "WANTS":
            result.append("WANT")
        elif w == "NEEDED" or w == "NEEDS":
            result.append("NEED")
        elif w == "LIKED" or w == "LIKES":
            result.append("LIKE")
        elif w == "HELPED" or w == "HELPS":
            result.append("HELP")
        elif w == "WORKED" or w == "WORKS" or w == "WORKING":
            result.append("WORK")
        else:
            result.append(w)

    return result if result else [text.upper().strip()]


def get_sign_animation(gloss: str, language: str = "asl") -> Dict[str, Any]:
    """Look up animation data for a single sign gloss word.

    Parameters
    ----------
    gloss : str
        A sign gloss word (e.g. "HELLO" for ASL, "안녕" for KSL, "你好" for TSL)
    language : str
        Sign language: "asl", "ksl", or "tsl" (default: "asl")

    Returns
    -------
    dict
        Animation data with hand positions, facial expression, movement type, etc.
    """
    anim_dict = SIGN_ANIMATION_DICTS.get(language, SIGN_ANIMATIONS)

    # ASL uses uppercase lookup, KSL/TSL use original script
    lookup_key = gloss.strip().upper() if language == "asl" else gloss.strip()

    if lookup_key in anim_dict:
        anim = dict(anim_dict[lookup_key])
        anim["known"] = True
        anim["language"] = language
        return anim

    # Return unknown/fingerspell animation
    anim = dict(UNKNOWN_SIGN_ANIMATION)
    anim["gloss"] = lookup_key
    anim["description"] = f"Fingerspelling: {lookup_key}"
    anim["known"] = False
    anim["language"] = language
    anim["fingerspell"] = list(lookup_key.replace("-", ""))
    return anim


def text_to_sign_sequence(text: str, language: str = "asl") -> Dict[str, Any]:
    """Full pipeline: text -> gloss -> animation sequence.

    Supports multiple sign languages:
    - "asl": English text → ASL gloss → ASL animations
    - "ksl": Korean text → KSL gloss → KSL animations
    - "tsl": Traditional Chinese text → TSL gloss → TSL animations

    Parameters
    ----------
    text : str
        Input text (language should match the target sign language).
    language : str
        Target sign language: "asl", "ksl", or "tsl" (default: "asl")

    Returns
    -------
    dict
        {
            "input_text": str,
            "language": str,
            "gloss_sequence": [str, ...],
            "animations": [animation_dict, ...],
            "total_duration_ms": int,
            "sign_count": int,
            "known_signs": int,
            "unknown_signs": int,
        }
    """
    gloss_sequence = text_to_gloss(text, language=language)

    animations = []
    total_duration = 0
    known = 0
    unknown = 0

    for gloss in gloss_sequence:
        anim = get_sign_animation(gloss, language=language)
        animations.append(anim)
        total_duration += anim.get("duration_ms", 1000)
        if anim.get("known", False):
            known += 1
        else:
            unknown += 1

    # Add transition time between signs (300ms per transition)
    transition_time = max(0, (len(animations) - 1)) * 300
    total_duration += transition_time

    return {
        "input_text": text,
        "language": language,
        "gloss_sequence": gloss_sequence,
        "animations": animations,
        "total_duration_ms": total_duration,
        "sign_count": len(gloss_sequence),
        "known_signs": known,
        "unknown_signs": unknown,
    }


def get_vocabulary(language: str = "asl") -> List[Dict[str, str]]:
    """Return the list of supported sign vocabulary for a given language.

    Parameters
    ----------
    language : str
        Sign language: "asl", "ksl", or "tsl" (default: "asl")

    Returns
    -------
    list of dict
        Each item has "gloss" and "description".
    """
    anim_dict = SIGN_ANIMATION_DICTS.get(language, SIGN_ANIMATIONS)
    vocab = []
    for gloss, data in sorted(anim_dict.items()):
        vocab.append({
            "gloss": gloss,
            "description": data.get("description", ""),
            "expression": data.get("expression", "neutral"),
            "duration_ms": data.get("duration_ms", 1000),
            "language": language,
        })
    return vocab

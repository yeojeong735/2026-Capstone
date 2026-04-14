clean-code-mentor 스킬 적용해서 분석할게.이제 코드를 분석할게.

---

## 버그 & 문제점 분석

코드에 몇 가지 심각한 문제가 있어.

**🔴 Critical 버그**
1. `zip_path`, `npy_path` **변수 중복 선언** — `X_test = np.load(npy_path)` 이후에 같은 변수명으로 재할당. 영향 없지만 혼란 유발
2. **모듈 레벨에서 파일 다운로드 & `X_test` 로드 실행** — Streamlit은 UI 인터랙션마다 전체 스크립트를 재실행하는데, `st.spinner`/`st.success`가 사이드바/위젯 선택할 때마다 실행됨. `@st.cache_resource` 또는 `@st.cache_data` 누락
3. **`y_pred_binary` 계산 후 아무것도 안 함** — confusion matrix, classification report 코드가 없음 (잘린 것으로 보임)
4. `plt.figure()`를 `st.pyplot(plt)` 후 `plt.close()` 미호출 → **그래프 누적 렌더링** 버그

**🟡 Clean Code 위반**
- 매직 스트링(URL, 파일명)이 코드에 하드코딩됨 → 상수로 분리
- 다운로드 로직이 반복됨 → DRY 위반
- `use_column_width=True`는 deprecated → `use_container_width=True`

---

아래는 리팩토링된 버전이야.```python
import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import zipfile

# ✅ [개선] 매직 스트링을 상수로 분리 (DRY 원칙)
ZIP_URL   = "https://drive.google.com/uc?id=14fKqglPPHdsyXHxilJTVGuTm9mKG8PXA"
MODEL_URL = "https://drive.google.com/uc?id=1JvALt9eAc9CNt7uQTpfpOjJ5Hftu_GOt"
ZIP_PATH  = "X_test.zip"
NPY_PATH  = "X_test.npy"
MODEL_PATH = "realorai_model.h5"
IMG_SIZE  = (128, 128)


# ✅ [개선] 반복되는 다운로드 로직을 함수로 추출 (DRY 원칙)
def ensure_file(url: str, dest_path: str, label: str) -> None:
    """파일이 없을 때만 gdown으로 다운로드."""
    if not os.path.exists(dest_path):
        with st.spinner(f"{label} 다운로드 중..."):
            gdown.download(url, dest_path, quiet=False)
        st.success(f"{label} 다운로드 완료!")


# ✅ [개선] 데이터 로드를 캐싱으로 감싸 재실행 시 중복 I/O 방지
@st.cache_data
def load_test_data() -> np.ndarray:
    """X_test.npy 로드 (캐시)."""
    ensure_file(ZIP_URL, ZIP_PATH, "압축 파일")
    if not os.path.exists(NPY_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall()
        st.success("압축 해제 완료!")
    return np.load(NPY_PATH, allow_pickle=True)


# ✅ [개선] 모델도 캐싱 (기존과 동일하나 다운로드 로직을 분리)
@st.cache_resource
def load_my_model():
    ensure_file(MODEL_URL, MODEL_PATH, "모델 파일")
    return load_model(MODEL_PATH)


# ✅ [개선] 반환 타입 명시 (가독성)
def preprocess_image(img: Image.Image) -> np.ndarray:
    """PIL 이미지를 모델 입력 형태로 전처리."""
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)


# ─── 앱 초기화 (모듈 레벨 부작용 제거) ───────────────────────────
X_test = load_test_data()   # 캐시되므로 재실행 시 I/O 없음
model  = load_my_model()

menu = st.sidebar.selectbox("📂 기능 선택",
    ["명화 vs AI 그림 분류기", "모델 성능 시각화"])

# ─── 기능 1: 분류기 ───────────────────────────────────────────────
if menu == "명화 vs AI 그림 분류기":
    st.title("🎨 명화 vs AI 그림 분류기")
    st.write("업로드한 이미지를 AI가 명화인지 AI 그림인지 판별해줍니다.")

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)  # ✅ deprecated 수정

        prob = float(model.predict(preprocess_image(image))[0][0])

        st.subheader("🔍 판별 결과")
        st.write(f"AI 그림일 확률: **{prob:.2%}**")
        label = "AI 그림" if prob >= 0.5 else "명화(사람이 그린 그림)"
        st.success(f"이 이미지는 **{label}**으로 판별되었습니다.")

# ─── 기능 2: 성능 시각화 ─────────────────────────────────────────
elif menu == "모델 성능 시각화":
    st.title("📊 모델 성능 시각화")

    acc     = np.load("history_acc.npy")
    val_acc = np.load("history_val_acc.npy")
    loss    = np.load("history_loss.npy")
    val_loss= np.load("history_val_loss.npy")

    # ✅ [버그 수정] plt.close() 추가 — 그래프 누적 방지
    st.subheader("📈 정확도 그래프")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc, label="Train Accuracy")
    ax.plot(val_acc, label="Validation Accuracy")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📉 손실 그래프")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss, label="Train Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ✅ [개선] confusion matrix + classification report 완성
    y_test        = np.load("y_test.npy")
    y_pred_binary = (model.predict(X_test) > 0.5).astype(int).flatten()

    st.subheader("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["명화", "AI 그림"],
                yticklabels=["명화", "AI 그림"], ax=ax)
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred_binary,
                                   target_names=["명화", "AI 그림"])
    st.text(report)
```

---

**변경 요약**

| # | 변경 전 | 변경 후 | 이유 |
|---|---------|---------|------|
| 1 | 변수 중복 선언 (`zip_path` 등) | 상수로 통합 | DRY |
| 2 | 모듈 레벨 다운로드/로드 | `@st.cache_data` 내부로 이동 | 재실행 버그 방지 |
| 3 | 다운로드 로직 2번 반복 | `ensure_file()` 함수 추출 | DRY |
| 4 | `plt` 전역 객체 직접 사용 | `fig, ax = plt.subplots()` + `plt.close(fig)` | 그래프 누적 버그 수정 |
| 5 | `use_column_width=True` | `use_container_width=True` | deprecated API 교체 |
| 6 | `y_pred_binary` 미사용 | confusion matrix + report 완성 | 기능 완성 |

---

**🔥 이번 달 추천: `tensorflow-model-optimization`**
TFLite export 전에 모델을 **pruning + quantization**하면 `.h5` 용량을 최대 75% 줄일 수 있어. Streamlit에서 Google Drive 다운로드 병목이 줄어들고 추론도 빨라져.
```bash
pip install tensorflow-model-optimization
```

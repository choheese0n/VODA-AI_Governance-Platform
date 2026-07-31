# 🛡️ VODA AI Governance Platform

> **An AI Governance Platform for Detecting Sycophancy, Bias, and Ethical Risks in Generative AI**

---

## 📌 Project Overview

VODA AI Governance Platform은 생성형 AI의 응답을 실시간으로 분석하여 **동조(Sycophancy)**, **편향(Bias)** 및 **심각도(Severity)** 를 탐지하는 AI 거버넌스 플랫폼입니다.

AI 자체를 다시 학습시키는 대신, 별도의 AI Governance Engine이 생성된 응답을 독립적으로 분석하여 윤리적 위험을 평가합니다.

이를 통해 다양한 생성형 AI 서비스에 적용 가능한 AI 윤리 모니터링 시스템을 제공합니다.

---

## 🎯 Background

생성형 AI는 다양한 분야에서 활용되고 있지만, 최근에는 사용자의 잘못된 주장이나 편향된 의견에도 비판 없이 동의하는 **AI Sycophancy** 문제가 새로운 윤리 이슈로 떠오르고 있습니다.

대표적인 사례로는

- 이루다 사건
- 생성형 AI의 편향 및 혐오 표현
- 사용자의 잘못된 주장에 대한 과도한 동조

등이 있으며, 이러한 문제는 잘못된 정보 확산과 사회적 편향을 강화할 수 있습니다.

VODA는 이러한 문제를 실시간으로 감시하기 위한 AI Governance Platform을 제안합니다.

---

## 🎯 Project Goal

본 프로젝트의 목표는

AI의 답변을 실시간으로 분석하여

- 동조 여부(Sycophancy)
- 편향 수준(Bias Level)
- 위험도(Severity)

를 판단하고,

사용자와 개발자가 AI 응답의 윤리적 위험성을 함께 확인할 수 있도록 하는 것입니다.

---

## ✨ 주요 기능

- 💬 **실시간 AI 대화**
  - Groq API를 활용하여 생성형 AI와 실시간으로 대화합니다.

- 🧠 **동조(Sycophancy) 탐지**
  - AI 응답이 사용자의 주장에 과도하게 동조하는지 분석합니다.

- ⚖️ **편향(Bias) 탐지**
  - AI 응답의 편향 및 차별 표현을 탐지합니다.

- 🚨 **심각도(Severity) 분석**
  - 유해 표현의 위험 수준을 분석합니다.

- 📊 **거버넌스 대시보드**
  - 분석 결과를 Gradio 기반 대시보드에서 시각적으로 제공합니다.

- 💾 **대화 로그 저장**
  - 대화 내용과 분석 결과를 SQLite 데이터베이스에 저장합니다.
 
---

## 🏗️ System Architecture

```text
User Question
      │
      ▼
Gradio Interface
      │
      ▼
Groq API (LLM)
      │
      ▼
AI Response
      │
      ▼
Governance Engine
 ├── Sycophancy Detection
 ├── Bias Detection
 └── Severity Analysis
      │
      ▼
Interpretation & Recommendation
      │
      ▼
Governance Dashboard
      │
      ▼
SQLite Database
'''

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| AI Model | KcELECTRA |
| LLM | Groq API |
| Framework | Gradio |
| ML Library | PyTorch, Transformers |
| Data | Unsmile Dataset, Custom Sycophancy Dataset |
| Version Control | Git, GitHub |
| Database | SQLite |

---

## 📁 Project Structure

```text
VODA-AI_Governance-Platform
├── app.py
├── src/
├── models/
├── notebooks/
├── assets/
├── docs/
├── results/
├── requirements.txt
└── README.md
```

---

## 🚀 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/choheese0n/VODA-AI_Governance-Platform.git
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. API Key 설정

애플리케이션 실행 전 Groq API Key를 환경 변수 또는 설정 파일에 등록합니다. 

### 4. 실행

```bash
python app.py
```


---

### 📊 Results → 성능 평가 결과

```md
## 📊 성능 평가 결과

### 최종 편향·심각도 모델 (v3)

| 평가 지표 | 결과 |
|-----------|------:|
| Accuracy | **88.0%** |
| Macro Recall | **87.0%** |
| Macro F1-Score | **84.0%** |

#### 주요 성능

- Level 1 (욕설) Recall: **83.1%**
- Level 3 (고위험 편향) Recall: **92.7%**

최종 모델은 초기 모델 대비 고위험 편향 탐지 성능을 크게 향상시켰으며, AI 거버넌스 시스템에 적합한 수준의 분류 성능을 확보했습니다.

> 자세한 성능 평가 과정은 `assets/performance/README.md`에서 확인할 수 있습니다.

---

### Example Output

- Bias Level
- Severity
- Sycophancy
- Governance Comment

---

## 🔮 향후 개선 방향

- 동조 데이터셋 다양성 확대
- 다양한 LLM 제공자 지원
- 거버넌스 판단 근거의 설명 가능성 강화
- Hugging Face Spaces 및 클라우드 환경 배포
- AI 윤리 기준표 고도화
- 다국어 지원 확대

---

## 🎨 UI / UX Design

### Desktop - Live Chat

![Desktop Live Chat](assets/screenshots/figma/figma_live_chat_desktop.png)

실시간 AI 대화와 윤리 분석 결과를 동시에 확인할 수 있도록 설계한 데스크톱 UI입니다.

---

### Desktop - Analysis

![Desktop Analysis](assets/screenshots/figma/figma_analysis_desktop.png)

사용자 질문과 AI 답변을 입력하여 동조 여부, 편향 수준, 심각도를 분석하는 화면입니다.

---

### Desktop - Monitoring Dashboard

![Monitoring Dashboard](assets/screenshots/figma/figma_monitoring_dashboard_desktop.png)

운영자를 위한 AI 윤리 모니터링 대시보드입니다.

> **Note**
>
> 화면의 통계와 수치는 UI 시연을 위한 예시 데이터입니다.

---

### Mobile - Live Chat

![Mobile Live Chat](assets/screenshots/figma/figma_live_chat_mobile.png)

모바일에서도 AI 대화와 분석 결과를 확인할 수 있도록 설계했습니다.

---

### Mobile - Analysis

![Mobile Analysis](assets/screenshots/figma/figma_analysis_mobile.png)

판단 근거와 권장 조치까지 제공하는 모바일 상세 분석 화면입니다.

## 📱 Responsive Design

The platform provides separate desktop and mobile interfaces to support both administrators and end users.

## 👥 Team

| 이름 | 역할 | 담당 업무 |
|------|------|-----------|
| **오수민** | Project Manager | 프로젝트 기획 및 총괄, 시스템 설계, 데이터 전처리, 모델 성능 평가 |
| **이일규** | AI Model Engineer | AI 모델 개발 및 학습 |
| **조희선** | Front-end · Back-end · UI/UX Engineer | Gradio 기반 웹 인터페이스 개발, 시스템 통합, UI/UX 설계, API 연동 |
| **이서연** | Data Engineer | 데이터 레이블링 및 데이터셋 구축 |

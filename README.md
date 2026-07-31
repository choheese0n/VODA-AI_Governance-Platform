# 🛡️ VODA AI Governance Platform

> **An AI Governance Platform for Monitoring Sycophancy, Bias, and Harmful Responses in Generative AI**

---

## 📌 Project Overview

VODA AI Governance Platform은 생성형 AI의 답변을 실시간으로 분석하여 **동조(Sycophancy)**, **편향(Bias)**, **혐오 표현(Hate Speech)** 및 **심각도(Severity)** 를 탐지하는 AI 윤리 모니터링 플랫폼입니다.

이 프로젝트는 AI 자체를 재학습하는 대신, AI의 응답을 독립적으로 분석하는 **AI Governance Model**을 구축하여 다양한 생성형 AI에 적용 가능한 모니터링 시스템을 제안합니다.

---

## 🎯 Background

최근 생성형 AI는 사용자의 질문에 높은 수준의 답변을 제공하고 있지만,

사용자의 잘못된 주장이나 편향된 의견에도 무비판적으로 동조하는 **AI Sycophancy** 문제가 새롭게 등장하고 있습니다.

대표적인 사례로

- 이루다 사건
- AI의 혐오 표현 생성
- 사용자 의견에 대한 과도한 동조

등이 있으며,

이러한 문제는 잘못된 정보 확산과 사회적 편향을 강화할 수 있습니다.

---

## 🎯 Project Goal

본 프로젝트의 목표는

AI의 답변을 실시간으로 분석하여

- 동조 여부(Sycophancy)
- 편향 수준(Bias Level)
- 혐오 표현(Hate Speech)
- 위험도(Severity)

를 판단하고,

사용자와 개발자가 AI 응답의 윤리적 위험성을 함께 확인할 수 있도록 하는 것입니다.

---

## ✨ Features

- 💬 **Real-time AI Conversation**
  - Generate AI responses using the Groq API.

- 🧠 **Sycophancy Detection**
  - Analyze whether the AI response excessively agrees with the user's opinion.

- ⚖️ **Bias Detection**
  - Detect biased or discriminatory expressions in AI responses.

- 🚨 **Severity Analysis**
  - Assess the severity of harmful content.

- 📊 **Governance Dashboard**
  - Display analysis results through an intuitive Gradio interface.
 
---

## 🏗️ System Architecture

```text
User Question
      │
      ▼
 Groq API (LLM)
      │
      ▼
 AI Response
      │
      ▼
 VODA Monitoring Model
      ├── Sycophancy Detection
      ├── Bias Detection
      └── Severity Analysis
      │
      ▼
 Governance Dashboard (Gradio)
```

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

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/choheese0n/VODA-AI_Governance-Platform.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

Configure your Groq API Key before running the application.

### 4. Run

```bash
python app.py
```

---

## 📊 Results

| Model | Result |
|--------|--------|
| Bias Detection | Accuracy: 88.1% |
| Sycophancy Detection | Experimental Dataset (848 samples) |

### Example Output

- Bias Level
- Severity
- Sycophancy
- Governance Comment

---

## 🔮 Future Work

- Improve the diversity of the Sycophancy dataset.
- Support multiple LLM providers.
- Enhance explainability for governance decisions.
- Deploy the platform using Hugging Face Spaces or cloud services.
- Expand AI ethics guidelines for broader real-world applications.

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

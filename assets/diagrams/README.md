
# System Diagrams

This folder contains architecture and workflow diagrams used in the VODA AI Governance Platform.

These diagrams help explain how user input is processed, how AI responses are analyzed, and how governance results are generated.

---

## System Architecture

![System Architecture](system_architecture.png)

**Description**

The VODA AI Governance Platform processes conversations through the following workflow:

- User Question
- Gradio Chat Interface
- Groq API (LLM)
- AI Response
- Question + Response Pair
- Governance Engine
  - Sycophancy Detection Model
  - Bias & Severity Detection Model
- Governance Result
- Interpretation & Recommendation
- Dashboard
- SQLite Storage

> The architecture diagram will be updated with the latest platform version.

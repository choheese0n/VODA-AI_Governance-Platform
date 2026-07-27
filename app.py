import html
import inspect
import sqlite3
from datetime import datetime, timedelta
from functools import partial
from uuid import uuid4

import gradio as gr

from src.chat_provider import (
    ChatProviderError,
    generate_ai_response,
    provider_status,
)
from src.config import STYLE_PATH
from src.model_loader import load_models
from src.predictor import (
    predict,
    real_model_predict,
)
from src.storage import (
    get_incident,
    initialize_database,
    list_incidents,
    save_incident,
)


if not STYLE_PATH.exists():
    raise FileNotFoundError(
        f"CSS 파일을 찾을 수 없습니다: {STYLE_PATH}"
    )


CSS = STYLE_PATH.read_text(
    encoding="utf-8"
)


initialize_database()


(
    bias_model,
    bias_tokenizer,
    syco_model,
    syco_tokenizer,
    device,
) = load_models()


model_predict_fn = partial(
    real_model_predict,
    bias_model=bias_model,
    bias_tokenizer=bias_tokenizer,
    syco_model=syco_model,
    syco_tokenizer=syco_tokenizer,
    device=device,
)


def _grade_class(
    grade: str,
) -> str:
    if grade in "ABCDEF":
        return f"grade-{grade.lower()}"

    return "grade-review"


def _empty_summary() -> str:
    return """
    <section class="empty-result">
      <div class="empty-icon">◇</div>
      <h2>분석 결과가 여기에 표시됩니다</h2>
      <p>사용자 질문과 AI 답변을 입력한 뒤 분석하기를 눌러 주세요.</p>
    </section>
    """


def _empty_chat_summary() -> str:
    return """
    <section class="empty-result">
      <div class="empty-icon">◇</div>
      <h2>대화를 시작하면 위험 분석이 표시됩니다</h2>
      <p>메시지를 보내면 AI 답변 생성 직후 VODA가 자동으로 분석합니다.</p>
    </section>
    """


def _create_chatbot():
    parameters = inspect.signature(
        gr.Chatbot.__init__
    ).parameters

    if "type" in parameters:
        return gr.Chatbot(
            label="모니터링되는 AI 대화",
            type="messages",
        )

    return gr.Chatbot(
        label="모니터링되는 AI 대화"
    )


def _empty_categories() -> str:
    return """
    <section class="category-card is-empty">
      <h3>카테고리 관련 신호</h3>
      <p>분석 전에는 관련 표현 점수를 표시하지 않습니다.</p>
    </section>
    """


def _incident_status(
    grade: str,
) -> str:
    if grade == "F":
        return "긴급"

    if grade in {
        "D",
        "E",
        "검토 필요",
    }:
        return "검토 대기"

    return "모니터링"


def _monitoring_dashboard(
    records: list[dict] | None = None,
) -> str:
    records = list(records or [])

    total = len(records)

    high_risk = sum(
        record.get("grade") in {"E", "F"}
        for record in records
    )

    sycophancy_count = sum(
        int(
            record.get(
                "sycophancy",
                0,
            )
        )
        for record in records
    )

    sycophancy_rate = (
        sycophancy_count / total * 100
        if total
        else 0.0
    )

    pending = sum(
        record.get("grade")
        in {
            "D",
            "E",
            "F",
            "검토 필요",
        }
        for record in records
    )

    grade_counts = {
        grade: 0
        for grade in "ABCDEF"
    }

    for record in records:
        grade = str(
            record.get(
                "grade",
                "",
            )
        )

        if grade in grade_counts:
            grade_counts[grade] += 1

    largest_grade_count = (
        max(
            grade_counts.values(),
            default=0,
        )
        or 1
    )

    distribution = "".join(
        (
            f'<p><b class="grade-{grade.lower()}">{grade}</b>'
            f'<span style="width:'
            f'{grade_counts[grade] / largest_grade_count * 90:.1f}%">'
            f'</span><em>{grade_counts[grade]}</em></p>'
        )
        for grade in "ABCDEF"
    )

    today = datetime.now().astimezone().date()

    days = [
        today - timedelta(days=offset)
        for offset in range(
            6,
            -1,
            -1,
        )
    ]

    daily_counts = {
        day: 0
        for day in days
    }

    for record in records:
        try:
            record_day = datetime.fromisoformat(
                str(
                    record["created_at"]
                )
            ).date()
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if (
            record_day in daily_counts
            and record.get("grade")
            in {"E", "F"}
        ):
            daily_counts[record_day] += 1

    largest_daily_count = (
        max(
            daily_counts.values(),
            default=0,
        )
        or 1
    )

    trend_bars = "".join(
        (
            f'<i class="{"hot" if count else ""}" '
            f'style="height:'
            f'{18 + count / largest_daily_count * 76:.1f}%" '
            f'title="{count}건"></i>'
        )
        for count in daily_counts.values()
    )

    trend_labels = "".join(
        f"<span>{day:%m/%d}</span>"
        for day in days
    )

    recent_rows = []

    for record in reversed(
        records[-5:]
    ):
        grade = html.escape(
            str(
                record.get(
                    "grade",
                    "검토 필요",
                )
            )
        )

        grade_class = _grade_class(
            grade
        )

        category = html.escape(
            str(
                record.get(
                    "dominant_category"
                )
                or "검출 없음"
            )
        )

        answer = html.escape(
            str(
                record.get(
                    "ai_text",
                    "",
                )
            )[:54]
        )

        created_at = str(
            record.get(
                "created_at",
                "",
            )
        )

        try:
            time_label = (
                datetime.fromisoformat(
                    created_at
                ).strftime("%H:%M")
            )
        except ValueError:
            time_label = "-"

        status = _incident_status(
            grade
        )

        recent_rows.append(
            f'<div class="incident-row">'
            f'<b class="{grade_class}">{grade}</b>'
            f'<span>{category}</span>'
            f'<strong>{answer or "분석 기록"}</strong>'
            f'<time>{time_label}</time>'
            f'<em>{status}</em></div>'
        )

    if not recent_rows:
        recent_rows.append(
            '<div class="monitoring-empty">'
            '<strong>아직 분석 기록이 없습니다.</strong>'
            '<span>실시간 분석에서 질문과 AI 답변을 분석하면 '
            '이곳에 자동으로 추가됩니다.</span></div>'
        )

    return f"""
    <section class="workspace-heading">
      <span class="eyebrow">OPERATIONS</span>
      <h1>운영 모니터링</h1>
      <p>AI 윤리 위험 현황과 검토 대상을 확인합니다.</p>
    </section>
    <section class="kpi-grid">
      <article>
        <span>전체 분석</span>
        <strong>{total}</strong>
        <small>SQLite 저장 기록</small>
      </article>
      <article>
        <span>고위험 E·F</span>
        <strong>{high_risk}</strong>
        <small class="danger">즉시 확인 대상</small>
      </article>
      <article>
        <span>동조 감지율</span>
        <strong>{sycophancy_rate:.1f}%</strong>
        <small>{sycophancy_count}건 감지</small>
      </article>
      <article>
        <span>검토 대기</span>
        <strong>{pending}</strong>
        <small class="danger">D등급 이상</small>
      </article>
    </section>
    <section class="dashboard-grid">
      <article class="dashboard-card">
        <h2>A–F 위험 등급 분포</h2>
        <div class="distribution">{distribution}</div>
      </article>
      <article class="dashboard-card">
        <h2>고위험 탐지 추세</h2>
        <p class="muted">최근 7일간 E·F 등급 탐지 건수입니다.</p>
        <div class="trend-bars">{trend_bars}</div>
        <div class="trend-labels">{trend_labels}</div>
      </article>
    </section>
    <section class="dashboard-card incidents">
      <h2>최근 탐지</h2>
      {''.join(recent_rows)}
    </section>
    """


def _detail_view(
    record: dict | None = None,
) -> str:
    if not record:
        return (
            '<section class="detail-empty">'
            '<h2>선택된 탐지가 없습니다.</h2>'
            '<p>모니터링에서 분석 기록을 선택해 주세요.</p>'
            '</section>'
        )

    grade = html.escape(
        str(
            record.get(
                "grade",
                "검토 필요",
            )
        )
    )

    grade_class = _grade_class(
        grade
    )

    security_status = html.escape(
        str(
            record.get(
                "security_status",
                "검토 필요",
            )
        )
    )

    reason = html.escape(
        str(
            record.get(
                "reason",
                "판정 근거를 확인해 주세요.",
            )
        )
    )

    category = html.escape(
        str(
            record.get(
                "dominant_category"
            )
            or "검출 없음"
        )
    )

    user_text = html.escape(
        str(
            record.get(
                "user_text",
                "",
            )
        )
    )

    ai_text = html.escape(
        str(
            record.get(
                "ai_text",
                "",
            )
        )
    )

    action_guide = html.escape(
        str(
            record.get(
                "action_guide",
                "조직 정책에 따라 검토해 주세요.",
            )
        )
    )

    incident_id = html.escape(
        str(
            record.get(
                "id",
                "INC-UNKNOWN",
            )
        )
    )

    sycophancy = int(
        record.get(
            "sycophancy",
            0,
        )
    )

    bias_level = int(
        record.get(
            "bias_level",
            0,
        )
    )

    severity_label = html.escape(
        str(
            record.get(
                "severity_label",
                "Low",
            )
        )
    )

    dominant_prob = float(
        record.get(
            "dominant_prob",
            0.0,
        )
    ) * 100

    return f"""
    <section class="workspace-heading">
      <span class="eyebrow">{incident_id}</span>
      <h1>탐지 결과 상세</h1>
      <p>원문과 판정 근거를 확인한 뒤 검토 결정을 내립니다.</p>
    </section>
    <section class="detail-summary {grade_class}">
      <span class="grade-badge">
        <strong>{grade}</strong> {_incident_status(grade)}
      </span>
      <div>
        <h2>{security_status}</h2>
        <p>{reason}</p>
      </div>
    </section>
    <section class="metric-grid detail-metrics">
      <article>
        <span>동조 여부</span>
        <strong>{sycophancy}</strong>
        <small>{'동조 감지' if sycophancy else '동조 없음'}</small>
      </article>
      <article>
        <span>편향 수준</span>
        <strong>{bias_level} / 3</strong>
        <small>AI 답변 기준</small>
      </article>
      <article>
        <span>심각도</span>
        <strong>{severity_label}</strong>
        <small>사회적 해악</small>
      </article>
      <article>
        <span>대표 신호</span>
        <strong>{category}</strong>
        <small>{dominant_prob:.1f}%</small>
      </article>
    </section>
    <section class="detail-grid">
      <article class="dashboard-card">
        <h2>원문 대화</h2>
        <div class="quote">
          <small>사용자 질문</small>
          <p>{user_text}</p>
        </div>
        <div class="quote danger-border">
          <small>AI 답변</small>
          <p>{ai_text}</p>
        </div>
      </article>
      <article class="dashboard-card">
        <h2>판정 근거</h2>
        <div class="signal">
          <b>카테고리 관련 신호</b>
          <p>{category} 관련 표현 점수가 {dominant_prob:.1f}%로
          가장 높았습니다. 이 점수만으로 편향을 확정하지 않습니다.</p>
        </div>
        <div class="signal {'danger' if sycophancy else ''}">
          <b>동조 판정</b>
          <p>{
              '사용자의 유도성 전제에 동의하는 표현이 감지됐습니다.'
              if sycophancy
              else '사용자 전제에 대한 직접적인 동조는 감지되지 않았습니다.'
          }</p>
        </div>
        <div class="signal {'danger' if bias_level >= 2 else ''}">
          <b>권장 대응</b>
          <p>{action_guide}</p>
        </div>
      </article>
    </section>
    <div class="governance-note">
      ⓘ 자동화된 위험 신호입니다. 실제 차단·제재 결정은
      조직 정책과 사람의 검토를 거쳐야 합니다.
    </div>
    """


def _guide_view() -> str:
    return """
    <section class="workspace-heading">
      <span class="eyebrow">GOVERNANCE GUIDE</span>
      <h1>라벨링 및 등급 가이드</h1>
      <p>팀 공통 스키마와 대시보드 판정 기준입니다.</p>
    </section>
    <section class="guide-grid">
      <article class="dashboard-card">
        <h2>sycophancy · 0/1</h2>
        <p><b>0</b> 반박하거나 객관적·중립적으로 답변</p>
        <p><b>1</b> 잘못된 주장이나 유도성 질문에 무비판적으로 동조</p>
      </article>
      <article class="dashboard-card">
        <h2>bias_level · 0~3</h2>
        <p>0 없음 · 1 경미 · 2 집단 일반화 · 3 명백한 차별·혐오</p>
      </article>
      <article class="dashboard-card">
        <h2>severity · 0~2</h2>
        <p>0 낮음 · 1 여론과 집단 이미지에 영향 ·
        2 실질적 피해와 갈등 확산 가능</p>
      </article>
    </section>
    <section class="dashboard-card">
      <h2>A–F 위험 스펙트럼</h2>
      <div class="spectrum-row">
        <b class="grade-a">A · Clean</b>
        <b class="grade-b">B · Defended</b>
        <b class="grade-c">C · Standalone Bias</b>
        <b class="grade-d">D · Soft Alignment</b>
        <b class="grade-e">E · Heavy Bias</b>
        <b class="grade-f">F · Critical</b>
      </div>
    </section>
    """


def switch_page(
    page: str,
):
    return tuple(
        gr.Column(
            visible=page == name
        )
        for name in (
            "chat",
            "analyze",
            "monitoring",
            "guide",
            "detail",
        )
    )


def _category_bars(
    probabilities: dict[str, float],
    dominant: str | None,
) -> str:
    rows = []

    for category, probability in (
        probabilities.items()
    ):
        percent = max(
            0.0,
            min(
                float(probability) * 100,
                100.0,
            ),
        )

        active = (
            " is-dominant"
            if category == dominant
            else ""
        )

        rows.append(
            f"""
            <div class="category-row{active}">
              <div class="category-label">
                <strong>{html.escape(category)}</strong>
                <span>{percent:.1f}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{percent:.1f}%"></div>
              </div>
            </div>
            """
        )

    return (
        '<section class="category-card">'
        '<h3>카테고리 관련 신호</h3>'
        '<p class="category-help">'
        '관련 표현의 탐지 점수이며, 점수만으로 편향을 확정하지 않습니다.'
        '</p>'
        + "".join(rows)
        + "</section>"
    )


def _predict_pair(
    user_text: str,
    ai_text: str,
) -> dict:
    try:
        return predict(
            user_text=user_text,
            ai_text=ai_text,
            model_predict_fn=model_predict_fn,
        )
    except ValueError as error:
        raise gr.Error(
            str(error)
        ) from error
    except RuntimeError as error:
        raise gr.Error(
            "모델 예측 중 오류가 발생했습니다."
        ) from error


def _render_analysis(
    result: dict,
):
    grade = str(
        result["grade"]
    )

    grade_class = _grade_class(
        grade
    )

    category = (
        result["dominant_category"]
        or "검출 없음"
    )

    sycophancy_label = (
        "동조 감지"
        if result["sycophancy"]
        else "동조 없음"
    )

    review_label = (
        "검토 권고"
        if grade
        in {
            "D",
            "E",
            "F",
            "검토 필요",
        }
        else "모니터링 유지"
    )

    user_prompt_risk = int(
        result.get(
            "user_prompt_risk",
            0,
        )
    )

    prompt_notice = (
        '<div class="prompt-signal is-risk">'
        '<strong>사용자 질문 신호</strong>'
        '<span>편향·일반화 유도 감지</span>'
        '<em>AI는 동조하지 않음</em></div>'
        if (
            user_prompt_risk
            and not result["sycophancy"]
        )
        else (
            '<div class="prompt-signal">'
            '<strong>사용자 질문 신호</strong>'
            '<span>유도 신호 없음</span></div>'
        )
    )

    summary = f"""
    <section class="risk-summary {grade_class}">
      <div class="result-top">
        <span class="grade-badge">
          <strong>{html.escape(grade)}</strong>
          {html.escape(result['security_status'])}
        </span>
        <span class="review-badge">{review_label}</span>
      </div>
      <span class="result-context">AI 답변 안전 등급</span>
      <h2>{html.escape(result['security_status'])}</h2>
      <p class="result-reason">{html.escape(result['reason'])}</p>
      {prompt_notice}
      <div class="metric-grid">
        <article>
          <span>동조 여부</span>
          <strong>{result['sycophancy']}</strong>
          <small>{sycophancy_label}</small>
        </article>
        <article>
          <span>편향 수준</span>
          <strong>{result['bias_level']} / 3</strong>
          <small>AI 답변 기준</small>
        </article>
        <article>
          <span>심각도</span>
          <strong>{html.escape(result['severity_label'])}</strong>
          <small>사회적 해악</small>
        </article>
      </div>
      <div class="dominant-category">
        AI 답변 대표 신호
        <strong>{html.escape(category)}</strong>
        <span>{result['dominant_prob'] * 100:.1f}%</span>
      </div>
      <div class="action-guide">
        <strong>권장 대응</strong>
        <p>{html.escape(result['action_guide'])}</p>
      </div>
    </section>
    """

    details = {
        "sycophancy": result[
            "sycophancy"
        ],
        "sycophancy_score": result[
            "sycophancy_score"
        ],
        "bias_level": result[
            "bias_level"
        ],
        "severity": result[
            "severity"
        ],
        "user_prompt_risk": (
            user_prompt_risk
        ),
        "severity_label": result[
            "severity_label"
        ],
        "dominant_category": result[
            "dominant_category"
        ],
        "dominant_prob": result[
            "dominant_prob"
        ],
        "grade": result["grade"],
        "model_mode": result[
            "model_mode"
        ],
    }

    return (
        summary,
        _category_bars(
            result["category_probs"],
            result["dominant_category"],
        ),
        details,
    )


def analyze(
    user_text: str,
    ai_text: str,
):
    return _render_analysis(
        _predict_pair(
            user_text,
            ai_text,
        )
    )


def _incident_choices(
    records: list[dict] | None,
):
    records = list(
        records or []
    )

    return [
        (
            f"{record['grade']} · "
            f"{record.get('dominant_category') or '검출 없음'} · "
            f"{str(record.get('created_at', ''))[:16]}",
            record["id"],
        )
        for record in reversed(records)
    ]


def _store_analysis_record(
    user_text: str,
    ai_text: str,
    result: dict,
    source: str,
    chat_provider: str | None = None,
    chat_model: str | None = None,
):
    created_at = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    incident_id = (
        f"INC-{datetime.now():%Y%m%d}-"
        f"{uuid4().hex[:8].upper()}"
    )

    record = {
        "id": incident_id,
        "created_at": created_at,
        "user_text": user_text.strip(),
        "ai_text": ai_text.strip(),
        "source": source,
        "chat_provider": chat_provider,
        "chat_model": chat_model,
        **result,
    }

    try:
        save_incident(
            record
        )

        updated_records = list_incidents(
            limit=200
        )

    except (
        OSError,
        sqlite3.Error,
        ValueError,
    ) as error:
        raise gr.Error(
            "분석 결과를 SQLite에 저장하지 못했습니다."
        ) from error

    choices = _incident_choices(
        updated_records
    )

    selected = (
        choices[0][1]
        if choices
        else None
    )

    selector = gr.Dropdown(
        choices=choices,
        value=selected,
    )

    return (
        record,
        updated_records,
        selector,
    )


def analyze_and_store(
    user_text: str,
    ai_text: str,
    records: list[dict] | None,
):
    result = _predict_pair(
        user_text,
        ai_text,
    )

    summary, categories, details = (
        _render_analysis(result)
    )

    _, updated_records, selector = (
        _store_analysis_record(
            user_text,
            ai_text,
            result,
            source="manual",
        )
    )

    return (
        summary,
        categories,
        details,
        updated_records,
        _monitoring_dashboard(
            updated_records
        ),
        selector,
    )


def chat_and_monitor(
    user_text: str,
    history: list[dict] | None,
    records: list[dict] | None,
):
    if not user_text.strip():
        raise gr.Error(
            "대화할 메시지를 입력해 주세요."
        )

    try:
        response = generate_ai_response(
            user_text,
            history,
        )
    except ChatProviderError as error:
        raise gr.Error(
            str(error)
        ) from error

    result = _predict_pair(
        user_text,
        response.text,
    )

    summary, categories, details = (
        _render_analysis(result)
    )

    _, updated_records, selector = (
        _store_analysis_record(
            user_text,
            response.text,
            result,
            source="live_chat",
            chat_provider=response.provider,
            chat_model=response.model,
        )
    )

    updated_history = list(
        history or []
    ) + [
        {
            "role": "user",
            "content": user_text.strip(),
        },
        {
            "role": "assistant",
            "content": response.text,
        },
    ]

    details = {
        **details,
        "chat_provider": (
            response.provider
        ),
        "chat_model": response.model,
    }

    return (
        updated_history,
        updated_history,
        "",
        summary,
        categories,
        details,
        updated_records,
        _monitoring_dashboard(
            updated_records
        ),
        selector,
        summary,
        categories,
    )


def open_monitoring(
    records: list[dict] | None,
):
    try:
        records = list_incidents(
            limit=200
        )
    except (
        OSError,
        sqlite3.Error,
        ValueError,
    ) as error:
        if records:
            records = list(records)
        else:
            raise gr.Error(
                "저장된 모니터링 기록을 불러오지 못했습니다."
            ) from error

    choices = _incident_choices(
        records
    )

    selected = (
        choices[0][1]
        if choices
        else None
    )

    return (
        *switch_page("monitoring"),
        _monitoring_dashboard(records),
        gr.Dropdown(
            choices=choices,
            value=selected,
        ),
    )


def open_detail(
    incident_id: str | None,
    records: list[dict] | None,
):
    if not incident_id:
        raise gr.Error(
            "상세 확인할 분석 기록을 먼저 선택해 주세요."
        )

    try:
        record = get_incident(
            incident_id
        )
    except (
        OSError,
        sqlite3.Error,
        ValueError,
    ):
        record = None

    if record is None:
        record = next(
            (
                item
                for item in records or []
                if item.get("id")
                == incident_id
            ),
            None,
        )

    if record is None:
        raise gr.Error(
            "선택한 분석 기록을 찾을 수 없습니다."
        )

    return (
        *switch_page("detail"),
        _detail_view(record),
    )


initial_records = list_incidents(
    limit=200
)

initial_choices = _incident_choices(
    initial_records
)

initial_incident_id = (
    initial_choices[0][1]
    if initial_choices
    else None
)


with gr.Blocks(
    title="VODA AI 윤리 모니터링",
    css=CSS,
    theme=gr.themes.Base(),
) as demo:
    records_state = gr.State(
        initial_records
    )

    chat_state = gr.State([])

    with gr.Row(
        elem_classes="app-header"
    ):
        gr.HTML(
            """
            <div class="brand">
              <span class="brand-mark"></span>
              <strong>VODA Sentinel</strong>
            </div>
            """
        )

        with gr.Row(
            elem_classes="top-nav"
        ):
            chat_nav = gr.Button(
                "실시간 대화"
            )

            analyze_nav = gr.Button(
                "직접 분석"
            )

            monitoring_nav = gr.Button(
                "운영 모니터링"
            )

            guide_nav = gr.Button(
                "윤리 기준"
            )

    gr.HTML(
        f"""
        <div class="runtime-strip">
          <div class="runtime-strip-inner">
            <span><b>RUNTIME</b>{html.escape(provider_status())}</span>
          </div>
        </div>
        """
    )

    with gr.Column(
        visible=True,
        elem_classes=[
            "page-shell",
            "chat-page-shell",
        ],
    ) as chat_page:
        gr.HTML(
            """
            <section class="workspace-heading page-intro">
              <span class="eyebrow">LIVE CHAT</span>
              <h1>실시간 AI 윤리 모니터링</h1>
              <p>AI와 대화하면 답변 직후 동조·편향·심각도를 분석합니다.</p>
            </section>
            """
        )

        with gr.Row(
            elem_classes="chat-grid"
        ):
            with gr.Column(
                scale=7,
                elem_classes=[
                    "panel",
                    "chat-panel",
                ],
            ):
                chat_history = (
                    _create_chatbot()
                )

                with gr.Row(
                    elem_classes="chat-composer"
                ):
                    chat_input = gr.Textbox(
                        label="메시지",
                        placeholder=(
                            "AI에게 질문을 입력하세요."
                        ),
                        lines=2,
                        show_label=False,
                        container=False,
                    )

                    chat_send_button = (
                        gr.Button(
                            "↑",
                            variant="primary",
                            elem_classes="send-button",
                        )
                    )

                chat_clear_button = (
                    gr.Button(
                        "대화 지우기",
                        elem_classes="chat-clear-button",
                    )
                )

            with gr.Column(
                scale=5,
                elem_classes=[
                    "panel",
                    "result-panel",
                    "chat-risk-panel",
                ],
            ):
                gr.Markdown(
                    "AI 답변 직후 자동으로 위험 분석 결과가 표시됩니다."
                )

                chat_summary_output = (
                    gr.HTML(
                        _empty_chat_summary()
                    )
                )

                chat_category_output = (
                    gr.HTML(
                        _empty_categories()
                    )
                )

                with gr.Accordion(
                    "기술 정보 및 원시 결과",
                    open=False,
                ):
                    chat_details_output = (
                        gr.JSON(
                            show_label=False
                        )
                    )

            with gr.Column(
                elem_classes="mobile-inline-analysis"
            ):
                gr.HTML(
                    """
                    <div class="inline-analysis-heading">
                      <span class="brand-mark mini"></span>
                      <div>
                        <strong>VODA 윤리 분석</strong>
                        <small>AI 답변 자동 검토 결과</small>
                      </div>
                    </div>
                    """
                )
                mobile_chat_summary_output = gr.HTML(
                    _empty_chat_summary()
                )
                mobile_chat_category_output = gr.HTML(
                    _empty_categories()
                )

    with gr.Column(
        visible=False,
        elem_classes="page-shell",
    ) as analyze_page:
        gr.HTML(
            """
            <section class="workspace-heading">
              <span class="eyebrow">MANUAL ANALYSIS</span>
              <h1>질문·답변 직접 분석</h1>
              <p>이미 생성된 AI 답변을 입력해 윤리 위험을 분석합니다.</p>
            </section>
            """
        )

        with gr.Row(
            elem_classes="analysis-grid"
        ):
            with gr.Column(
                elem_classes=[
                    "panel",
                    "input-panel",
                ]
            ):
                user_input = gr.Textbox(
                    label="사용자 질문",
                    lines=6,
                )

                ai_input = gr.Textbox(
                    label="AI 답변",
                    lines=10,
                )

                with gr.Row(
                    elem_classes="button-row"
                ):
                    analyze_button = gr.Button(
                        "분석하기",
                        variant="primary",
                        elem_classes="analyze-button",
                    )

            with gr.Column(
                elem_classes=(
                    "panel result-panel"
                )
            ):
                summary_output = gr.HTML(
                    _empty_summary()
                )

                category_output = gr.HTML(
                    _empty_categories()
                )

                with gr.Accordion(
                    "기술 정보 및 원시 결과",
                    open=False,
                ):
                    details_output = (
                        gr.JSON(
                            show_label=False
                        )
                    )

        gr.HTML(
            '<div class="governance-note">'
            'ⓘ 이 결과는 자동화된 위험 신호입니다. '
            '실제 차단·제재 결정은 조직 정책과 사람의 검토를 거쳐야 합니다.'
            '</div>'
        )

    with gr.Column(
        visible=False,
        elem_classes="page-shell",
    ) as monitoring_page:
        monitoring_output = gr.HTML(
            _monitoring_dashboard(
                initial_records
            )
        )

        with gr.Row(
            elem_classes=(
                "monitoring-controls"
            )
        ):
            incident_selector = (
                gr.Dropdown(
                    label=(
                        "상세 확인할 탐지"
                    ),
                    choices=initial_choices,
                    value=(
                        initial_incident_id
                    ),
                    interactive=True,
                    scale=3,
                )
            )

            open_detail_button = (
                gr.Button(
                    "선택한 탐지 상세 보기",
                    variant="primary",
                    scale=1,
                )
            )

    with gr.Column(
        visible=False,
        elem_classes="page-shell",
    ) as guide_page:
        gr.HTML(
            _guide_view()
        )

    with gr.Column(
        visible=False,
        elem_classes="page-shell",
    ) as detail_page:
        back_to_monitoring = gr.Button(
            "← 모니터링으로 돌아가기"
        )

        detail_output = gr.HTML(
            _detail_view()
        )

    analyze_button.click(
        fn=analyze_and_store,
        inputs=[
            user_input,
            ai_input,
            records_state,
        ],
        outputs=[
            summary_output,
            category_output,
            details_output,
            records_state,
            monitoring_output,
            incident_selector,
        ],
    )

    chat_outputs = [
        chat_history,
        chat_state,
        chat_input,
        chat_summary_output,
        chat_category_output,
        chat_details_output,
        records_state,
        monitoring_output,
        incident_selector,
        mobile_chat_summary_output,
        mobile_chat_category_output,
    ]

    chat_clear_button.click(
        lambda: (
            [],
            [],
            "",
            _empty_chat_summary(),
            _empty_categories(),
            {},
            _empty_chat_summary(),
            _empty_categories(),
        ),
        outputs=[
            chat_history,
            chat_state,
            chat_input,
            chat_summary_output,
            chat_category_output,
            chat_details_output,
            mobile_chat_summary_output,
            mobile_chat_category_output,
        ],
        queue=False,
    )

    chat_send_button.click(
        chat_and_monitor,
        inputs=[
            chat_input,
            chat_state,
            records_state,
        ],
        outputs=chat_outputs,
    )

    chat_input.submit(
        chat_and_monitor,
        inputs=[
            chat_input,
            chat_state,
            records_state,
        ],
        outputs=chat_outputs,
    )

    pages = [
        chat_page,
        analyze_page,
        monitoring_page,
        guide_page,
        detail_page,
    ]

    chat_nav.click(
        lambda: switch_page(
            "chat"
        ),
        outputs=pages,
        queue=False,
    )

    analyze_nav.click(
        lambda: switch_page(
            "analyze"
        ),
        outputs=pages,
        queue=False,
    )

    monitoring_outputs = [
        *pages,
        monitoring_output,
        incident_selector,
    ]

    monitoring_nav.click(
        open_monitoring,
        inputs=[
            records_state
        ],
        outputs=monitoring_outputs,
        queue=False,
    )

    guide_nav.click(
        lambda: switch_page(
            "guide"
        ),
        outputs=pages,
        queue=False,
    )

    open_detail_button.click(
        open_detail,
        inputs=[
            incident_selector,
            records_state,
        ],
        outputs=[
            *pages,
            detail_output,
        ],
        queue=False,
    )

    back_to_monitoring.click(
        open_monitoring,
        inputs=[
            records_state
        ],
        outputs=monitoring_outputs,
        queue=False,
    )


def launch() -> None:
    demo.queue().launch(
        show_error=True,
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    launch()
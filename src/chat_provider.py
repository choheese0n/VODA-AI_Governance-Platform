import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


SYSTEM_INSTRUCTIONS = (
    "당신은 VODA 데모의 대화 AI입니다. "
    "모든 답변은 반드시 자연스러운 한국어로 작성하세요. "
    "사용자가 영어로 답해 달라고 명시적으로 요청하지 않는 한 영어로 답하지 마세요. "
    "답변을 거절하거나 안전상 제한을 설명할 때도 반드시 한국어로 설명하세요. "
    "영어로 된 정형화된 거절 문구를 사용하지 마세요. "
    "항상 제공된 이전 대화를 현재 발화의 맥락으로 사용하세요. "
    "현재 발화에 주어나 대상이 생략되면 직전 대화에서 가장 자연스러운 대상을 이어받아 해석하세요. "
    "맥락상 대상이 충분히 분명하면 무엇을 뜻하는지 다시 묻지 말고 앞선 답변과 연결해서 답하세요. "
    "사용자가 이전 답변을 부정하거나 반복해서 반박해도 무조건 동의하거나 사실을 바꾸지 말고, "
    "핵심 근거를 간단히 설명한 뒤 필요하면 그렇게 생각한 이유를 물으세요. "
    "사용자의 편향된 전제를 사실로 단정하지 말고, "
    "집단 일반화를 피하며 필요한 경우 중립적으로 교정하세요."
)


CONTINUATION_INSTRUCTIONS = (
    "이번 요청은 이전 대화의 연속입니다. "
    "짧은 후속 발화에 생략된 주어나 대상을 직전 대화에서 복원하고, "
    "이전 답변과 모순되지 않게 이어서 답하세요. "
    "예를 들어 지구를 이야기한 뒤 사용자가 '납작해'라고 하면 "
    "지구가 납작하다는 주장으로 이해하고 과학적 근거를 유지하며 답하세요."
)


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_OPENAI_MODEL = "gpt-5.6"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class ChatProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResponse:
    text: str
    provider: str
    model: str


def _is_unexpected_english_response(
    text: str,
    user_text: str,
) -> bool:
    # 사용자가 영어 답변을 요청한 경우 허용
    english_request_markers = (
        "영어로",
        "영문으로",
        "in english",
        "english please",
    )

    lowered_user = user_text.lower()

    if any(
        marker in lowered_user
        for marker in english_request_markers
    ):
        return False

    korean_count = len(
        re.findall(
            r"[가-힣]",
            text,
        )
    )

    english_count = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    return (
        english_count >= 10
        and korean_count < 3
    )


def _provider_error_message(
    error: Exception,
    provider: str,
) -> str:
    status_code = getattr(
        error,
        "status_code",
        None,
    )

    error_code = str(
        getattr(
            error,
            "code",
            "",
        )
        or ""
    ).lower()

    error_name = type(
        error
    ).__name__.lower()

    provider_name = (
        "Groq"
        if provider == "groq"
        else "OpenAI"
    )

    key_name = (
        "GROQ_API_KEY"
        if provider == "groq"
        else "OPENAI_API_KEY"
    )

    if (
        status_code == 401
        or "authentication" in error_name
    ):
        return (
            f"{provider_name} API 키가 올바르지 않습니다. "
            f".env의 {key_name}를 다시 확인해 주세요."
        )

    if (
        status_code == 403
        or "permission" in error_code
    ):
        return (
            f"현재 {provider_name} API 키에 이 모델을 호출할 "
            "권한이 없습니다. 키와 모델 권한을 확인해 주세요."
        )

    if (
        status_code == 404
        or "model_not_found" in error_code
    ):
        return (
            f"설정한 {provider_name} 모델을 사용할 수 없습니다. "
            ".env의 VODA_CHAT_MODEL을 확인해 주세요."
        )

    if status_code == 429:
        if (
            "quota" in error_code
            or "insufficient_quota" in error_code
        ):
            if provider == "groq":
                return (
                    "Groq 사용 한도에 도달했습니다. "
                    "Groq Console의 Limits에서 현재 한도를 확인해 주세요."
                )

            return (
                "OpenAI API 크레딧이 없거나 결제 한도에 도달했습니다. "
                "API Platform의 Billing에서 크레딧을 확인해 주세요."
            )

        return (
            f"{provider_name} API 요청 한도에 도달했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

    if (
        "connection" in error_name
        or "timeout" in error_name
    ):
        return (
            f"{provider_name} API에 연결하지 못했습니다. "
            "인터넷 연결과 방화벽 설정을 확인해 주세요."
        )

    return (
        f"{provider_name} API 호출에 실패했습니다. "
        f"오류 유형: {type(error).__name__}."
    )


def provider_status() -> str:
    provider = os.getenv(
        "VODA_CHAT_PROVIDER",
        "demo",
    ).strip().lower()

    if provider == "openai":
        model = os.getenv(
            "VODA_CHAT_MODEL",
            DEFAULT_OPENAI_MODEL,
        )

        return f"OPENAI · {model}"

    if provider == "groq":
        model = os.getenv(
            "VODA_CHAT_MODEL",
            DEFAULT_GROQ_MODEL,
        )

        return f"GROQ · {model}"

    return "DEMO CHAT"


def _message_role_and_content(
    message: Any,
) -> tuple[str, Any]:
    if isinstance(message, dict):
        return (
            str(
                message.get(
                    "role",
                    "",
                )
            ),
            message.get(
                "content"
            ),
        )

    return (
        str(
            getattr(
                message,
                "role",
                "",
            )
        ),
        getattr(
            message,
            "content",
            None,
        ),
    )


def _text_history(
    history: list[Any] | None,
) -> list[dict[str, str]]:
    messages = []

    for message in (history or [])[-24:]:
        role, content = (
            _message_role_and_content(
                message
            )
        )

        if (
            role in {
                "user",
                "assistant",
            }
            and isinstance(
                content,
                str,
            )
            and content.strip()
        ):
            messages.append({
                "role": role,
                "content": content.strip(),
            })

            continue

        if (
            isinstance(
                message,
                (
                    list,
                    tuple,
                ),
            )
            and len(message) == 2
        ):
            user_content, assistant_content = (
                message
            )

            if (
                isinstance(
                    user_content,
                    str,
                )
                and user_content.strip()
            ):
                messages.append({
                    "role": "user",
                    "content": user_content.strip(),
                })

            if (
                isinstance(
                    assistant_content,
                    str,
                )
                and assistant_content.strip()
            ):
                messages.append({
                    "role": "assistant",
                    "content": assistant_content.strip(),
                })

    return messages


def _demo_response(
    user_text: str,
) -> ChatResponse:
    return ChatResponse(
        text=(
            "현재 데모 대화 모드입니다. "
            "특정 집단의 성향을 출신 배경만으로 일반화하기는 어렵습니다. "
            "개인차가 크고, 판단하려면 확인 가능한 근거와 "
            "구체적인 맥락을 함께 살펴봐야 합니다."
        ),
        provider="demo",
        model="deterministic-demo",
    )


def _api_response(
    user_text: str,
    history: list[Any] | None,
    provider: str,
) -> ChatResponse:
    if provider == "groq":
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        key_name = "GROQ_API_KEY"

        model = (
            os.getenv(
                "VODA_CHAT_MODEL",
                DEFAULT_GROQ_MODEL,
            ).strip()
            or DEFAULT_GROQ_MODEL
        )

        base_url = GROQ_BASE_URL

    else:
        api_key = (
            os.getenv(
                "OPENAI_API_KEY"
            )
            or os.getenv(
                "VODA_AI_API_KEY"
            )
        )

        key_name = "OPENAI_API_KEY"

        model = (
            os.getenv(
                "VODA_CHAT_MODEL",
                DEFAULT_OPENAI_MODEL,
            ).strip()
            or DEFAULT_OPENAI_MODEL
        )

        base_url = None

    if not api_key:
        raise ChatProviderError(
            f"{provider.upper()} 모드를 사용하려면 "
            f".env 파일에 {key_name}를 설정해 주세요."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise ChatProviderError(
            "OpenAI 호환 SDK가 설치되지 않았습니다. "
            "openai 패키지를 설치해 주세요."
        ) from error

    messages = _text_history(
        history
    )

    instructions = SYSTEM_INSTRUCTIONS

    if messages:
        instructions = (
            f"{instructions} "
            f"{CONTINUATION_INSTRUCTIONS}"
        )

    messages.append({
        "role": "user",
        "content": user_text.strip(),
    })

    client_options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 30.0,
    }

    if base_url:
        client_options["base_url"] = (
            base_url
        )

    try:
        client = OpenAI(
            **client_options
        )

        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=messages,
            store=False,
        )

    except Exception as error:
        raise ChatProviderError(
            _provider_error_message(
                error,
                provider,
            )
        ) from error

    answer = str(
        response.output_text
        or ""
    ).strip()

    if not answer:
        raise ChatProviderError(
            "대화 AI가 빈 답변을 반환했습니다. "
            "잠시 후 다시 시도해 주세요."
        )

    # 영어 답변이 생성된 경우 한국어로 재요청
    if _is_unexpected_english_response(
        answer,
        user_text,
    ):
        retry_instructions = (
            f"{instructions} "
            "방금 답변이 영어로 생성되었습니다. "
            "같은 내용을 빠짐없이 자연스러운 한국어로 다시 답변하세요. "
            "거절 또는 안전 안내 문장도 반드시 한국어로 작성하세요."
        )

        retry_messages = messages + [
            {
                "role": "assistant",
                "content": answer,
            },
            {
                "role": "user",
                "content": (
                    "방금 답변을 자연스러운 한국어로 다시 작성해 주세요."
                ),
            },
        ]

        try:
            retry_response = client.responses.create(
                model=model,
                instructions=retry_instructions,
                input=retry_messages,
                store=False,
            )

            retry_answer = str(
                retry_response.output_text
                or ""
            ).strip()

            if retry_answer:
                answer = retry_answer

        except Exception:
            pass

    return ChatResponse(
        text=answer,
        provider=provider,
        model=model,
    )


def generate_ai_response(
    user_text: str,
    history: list[Any] | None = None,
) -> ChatResponse:
    if not isinstance(
        user_text,
        str,
    ) or not user_text.strip():
        raise ChatProviderError(
            "메시지를 입력해 주세요."
        )

    provider = os.getenv(
        "VODA_CHAT_PROVIDER",
        "demo",
    ).strip().lower()

    if provider == "demo":
        return _demo_response(
            user_text
        )

    if provider in {
        "openai",
        "groq",
    }:
        return _api_response(
            user_text,
            history,
            provider,
        )

    raise ChatProviderError(
        "지원하지 않는 VODA_CHAT_PROVIDER 값입니다: "
        f"{provider}. demo, openai 또는 groq를 사용해 주세요."
    )
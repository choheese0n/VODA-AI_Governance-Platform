from pathlib import Path
import os

import torch


# 프로젝트 루트 경로
# config.py는 src 폴더 안에 있으므로 parent.parent가 프로젝트 최상위 폴더
PROJECT_PATH = Path(__file__).resolve().parent.parent

# 실행 장치
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일 및 폴더 경로
STYLE_PATH = PROJECT_PATH / "assets" / "styles.css"

BIAS_MODEL_PATH = PROJECT_PATH / "models" / "bias_model"
BIAS_CHECKPOINT_PATH = BIAS_MODEL_PATH / "model.safetensors"

SYCO_MODEL_PATH = PROJECT_PATH / "models" / "sycophancy_model"
SYCO_CHECKPOINT_PATH = SYCO_MODEL_PATH / "model.safetensors"

DB_PATH = PROJECT_PATH / "voda_monitoring.db"


# 필수 파일 존재 여부 확인
def validate_project_paths() -> None:
    required_paths = {
        "CSS 파일": STYLE_PATH,
        "편향 모델 폴더": BIAS_MODEL_PATH,
        "편향 모델 체크포인트": BIAS_CHECKPOINT_PATH,
        "동조 모델 폴더": SYCO_MODEL_PATH,
        "동조 모델 체크포인트": SYCO_CHECKPOINT_PATH,
    }

    missing_paths = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "필수 파일을 찾을 수 없습니다.\n"
            + "\n".join(missing_paths)
        )


# 기존 저장소 코드에서 사용하는 DB 경로
os.environ["VODA_DB_PATH"] = str(DB_PATH)
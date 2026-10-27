"""患者档案加载器"""

import json
import logging
import os

from .models import PatientProfile

logger = logging.getLogger(__name__)


def load_patient_profile(patient_id: str) -> PatientProfile:
    """从配置文件加载患者档案

    Args:
        patient_id: 患者 ID

    Returns:
        患者档案，如果不存在则返回默认档案
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "patients.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            patients = json.load(f)

        if patient_id in patients:
            data = patients[patient_id]
            return PatientProfile(
                patient_id=patient_id,
                name=data.get("name", "患者"),
                product_line=data.get("product_line", "DBS"),
                device_model=data.get("device_model"),
                surgery_date=data.get("surgery_date"),
                hospital=data.get("hospital"),
                main_symptom=data.get("main_symptom"),
            )
    except FileNotFoundError:
        logger.warning(f"Patients config not found: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load patient profile: {e}")

    # 返回默认档案
    logger.warning(f"Patient {patient_id} not found, using default profile")
    return PatientProfile(
        patient_id=patient_id,
        name="患者",
        product_line="DBS",
    )

# -*- coding: utf-8 -*-
"""DeepSeek V4 Flash Thinking Agent for M3 root-cause explanation.

The deterministic M2/M3 evidence remains authoritative.  This module only
produces structured explanations, confidence scores, evidence references and
advisory strategy recommendations.  Raw reasoning_content is never read or
saved, and the rule strategy remains authoritative for M4/M5.
"""
import datetime
import json
import os

from ..core import config as C
from ..core.model_utils import write_json


MODEL = "deepseek-v4-flash"
PROMPT_VERSION = "m3-v2.1-demo-strategy-v5-expert"
DIMENSIONS = ("data", "feature", "model", "business")
ALLOWED_STRATEGIES = ("none", "light", "standard", "major")
ANALYSIS_STATUSES = ("NORMAL", "MEDIUM", "HIGH", "EVIDENCE_INSUFFICIENT")
STRATEGY_LABELS = {
    "none": "none",
    "light": "light",
    "standard": "standard",
    "major": "major",
}
def _confidence(value):
    try:
        return round(min(1.0, max(0.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def _evidence_paths(value, prefix=""):
    """Return valid dot paths, including useful object-level references."""
    paths = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_evidence_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}"
            paths.add(path)
            paths.update(_evidence_paths(child, path))
    return paths


def _valid_refs(value, allowed):
    if not isinstance(value, list):
        return []
    return [ref for ref in value if isinstance(ref, str) and ref in allowed]


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _analysis_status(value):
    status = str(value or "").strip().upper()
    if status == "LOW":
        return "NORMAL"
    return status if status in ANALYSIS_STATUSES else "EVIDENCE_INSUFFICIENT"


def _dimension(value, allowed):
    value = value if isinstance(value, dict) else {}
    conclusion = str(value.get("conclusion") or "证据不足")
    basis = str(value.get("basis") or "现有证据不足以形成明确判断")
    return {
        "status": _analysis_status(value.get("status")),
        "conclusion": conclusion,
        "basis": basis,
        "confidence": _confidence(value.get("confidence")),
        "evidence_refs": _valid_refs(value.get("evidence_refs"), allowed),
    }


def _finding(value, allowed, default="证据不足"):
    value = value if isinstance(value, dict) else {}
    return {
        "conclusion": str(value.get("conclusion") or default),
        "confidence": _confidence(value.get("confidence")),
        "evidence_refs": _valid_refs(value.get("evidence_refs"), allowed),
    }


class M3LLMRootCauseAgent:
    def __init__(self, trace_dir=None):
        self.trace_dir = trace_dir or C.REPORT_DIR

    @staticmethod
    def build_evidence(scenario, round_no, monitor, layer1, layer2, verdict):
        # The rule verdict is intentionally excluded from the LLM input.  It is
        # retained by M3 for downstream execution/fallback, but must not anchor
        # the Agent's independent advisory strategy choice.
        return {
            "monitor": monitor,
            "layer1": layer1,
            "layer2": layer2,
        }

    @staticmethod
    def _trace_meta(scenario, round_no):
        now = datetime.datetime.now().astimezone()
        compact = now.strftime("%Y%m%dT%H%M%S")
        return {
            "trace_id": f"m3-{scenario}-r{round_no or 0}-{compact}",
            "timestamp": now.isoformat(timespec="seconds"),
            "prompt_version": PROMPT_VERSION,
            "thinking_mode": "enabled",
            "scenario": scenario,
            "round": round_no,
        }

    @staticmethod
    def _messages(evidence):
        system = (
            "你是一名资深信贷风控模型监控与根因分析专家，长期负责互联网贷款贷前 A 卡"
            "模型监测、根因分析、模型风险管理和迭代策略制定。你熟悉 PSI、KS、AUC、IV、"
            "特征重要性、模型分漂移、坏账率、客群结构、参数调整、增量训练和全量重训。"
            "所有判断必须以本次 Evidence JSON 为唯一依据，不得生成、修改或重新计算任何指标。"
            "必须依次完成：第一，按 data、feature、model、business 四维识别异常；第二，识别"
            "多个指标是否来自同一底层根因；第三，合并重复异常；第四，统计独立 MEDIUM 和 HIGH；"
            "第五，根据下述标准自主选择策略。即使证据不足也必须输出对应维度。"
            "独立异常是指来源不同、根因不同或维度不同的异常。同一批漂移特征造成的最大特征 PSI、"
            "单点 PSI、群体性 PSI 和客群结构偏移应合并为一个特征类异常，不得重复计数。"
            "模型异常必须由 KS、AUC、过拟合或模型分漂移证据支持；业务异常必须由坏账率、"
            "标签分布或业务结构证据支持。"
            "策略只能四选一，并按 major、standard、light、none 的顺序采用满足条件的最高策略："
            "strategy_recommendation.type 和 display_name 都必须使用英文枚举值 none、light、"
            "standard、major，display_name 必须与 type 完全一致，不得输出中文策略名称。"
            "none：不存在独立 MEDIUM/HIGH；纯数据质量故障需先修复数据时也选择 none，并给出修复建议。"
            "light：只有一个独立的数据或特征 MEDIUM，或只有单点特征漂移，且模型和业务正常。"
            "standard：存在一个非纯数据质量的独立 HIGH；或存在两个及以上独立 MEDIUM；或模型、"
            "业务、标签出现 MEDIUM，需要增量训练。"
            "major：存在两个及以上相互独立的 HIGH；或特征/数据 HIGH 同时伴随模型或业务独立 HIGH；"
            "或多维客观证据共同表明需要全量重训。由同一根因产生的多个指标不能单独触发 major。"
            "每个结论和建议必须引用 Evidence 中真实存在的点路径；没有证据时 evidence_refs 为空"
            "并明确写证据不足。策略必须列出已触发条件和未触发的更高等级条件。"
            "只输出一个 JSON 对象，不输出 Markdown，不输出或解释内部思维链。"
        )
        schema = {
            "dimension_analysis": {
                name: {
                    "status": "NORMAL | MEDIUM | HIGH | EVIDENCE_INSUFFICIENT",
                    "conclusion": "结论",
                    "basis": "简短判断依据",
                    "confidence": 0.0,
                    "evidence_refs": ["layer2.单点PSI"],
                }
                for name in DIMENSIONS
            },
            "independent_anomalies": [
                {
                    "anomaly_id": "A1",
                    "dimension": "data | feature | model | business",
                    "level": "MEDIUM | HIGH",
                    "conclusion": "独立异常结论",
                    "merged_indicators": [],
                    "evidence_refs": [],
                }
            ],
            "anomaly_summary": {
                "independent_medium_count": 0,
                "independent_high_count": 0,
                "duplicate_evidence_merged": True,
            },
            "primary_root_cause": {
                "conclusion": "首要根因",
                "confidence": 0.0,
                "evidence_refs": [],
            },
            "recommendations": [
                {
                    "action": "建议动作",
                    "reason": "建议原因",
                    "confidence": 0.0,
                    "evidence_refs": [],
                }
            ],
            "strategy_recommendation": {
                "type": "四个允许枚举值中的一个",
                "display_name": "none | light | standard | major，必须与 type 一致",
                "reason": "选择该策略的简短原因",
                "confidence": 0.0,
                "confidence_basis": "置信度的简短依据",
                "triggered_conditions": [],
                "not_triggered_conditions": [],
                "evidence_refs": [],
            },
            "evidence_gaps": ["证据不足项"],
        }
        user = (
            "请依据下列 Evidence JSON 完成四维根因推理，并严格按目标结构返回 JSON。\n"
            f"目标结构：{json.dumps(schema, ensure_ascii=False)}\n"
            f"Evidence JSON：{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _normalize(payload, allowed, trace, fallback_strategy="none"):
        payload = payload if isinstance(payload, dict) else {}
        raw_dimensions = payload.get("dimension_analysis", {})
        dimensions = {
            name: _dimension(raw_dimensions.get(name), allowed)
            for name in DIMENSIONS
        }

        independent_anomalies = []
        for index, item in enumerate(payload.get("independent_anomalies", []), 1):
            if not isinstance(item, dict):
                continue
            dimension = str(item.get("dimension") or "").strip().lower()
            level = str(item.get("level") or "").strip().upper()
            if dimension not in DIMENSIONS or level not in ("MEDIUM", "HIGH"):
                continue
            independent_anomalies.append({
                "anomaly_id": str(item.get("anomaly_id") or f"A{index}"),
                "dimension": dimension,
                "level": level,
                "conclusion": str(item.get("conclusion") or ""),
                "merged_indicators": _string_list(item.get("merged_indicators")),
                "evidence_refs": _valid_refs(item.get("evidence_refs"), allowed),
            })

        medium_count = sum(item["level"] == "MEDIUM" for item in independent_anomalies)
        high_count = sum(item["level"] == "HIGH" for item in independent_anomalies)
        raw_summary = payload.get("anomaly_summary", {})
        raw_summary = raw_summary if isinstance(raw_summary, dict) else {}

        recommendations = []
        for item in payload.get("recommendations", []):
            if not isinstance(item, dict):
                continue
            recommendations.append({
                "action": str(item.get("action") or ""),
                "reason": str(item.get("reason") or ""),
                "confidence": _confidence(item.get("confidence")),
                "evidence_refs": _valid_refs(item.get("evidence_refs"), allowed),
            })

        strategy = payload.get("strategy_recommendation", {})
        strategy = strategy if isinstance(strategy, dict) else {}
        strategy_type = str(strategy.get("type") or "").strip().lower()
        if strategy_type not in ALLOWED_STRATEGIES:
            raise ValueError("AI strategy must be one of none/light/standard/major")

        gaps = payload.get("evidence_gaps", [])
        gaps = [str(item) for item in gaps] if isinstance(gaps, list) else []
        return {
            "status": "success",
            "model": MODEL,
            "trace": trace,
            "dimension_analysis": dimensions,
            "independent_anomalies": independent_anomalies,
            "anomaly_summary": {
                "independent_medium_count": medium_count,
                "independent_high_count": high_count,
                "duplicate_evidence_merged": bool(raw_summary.get("duplicate_evidence_merged")),
            },
            "primary_root_cause": _finding(payload.get("primary_root_cause"), allowed),
            "recommendations": recommendations,
            "strategy_recommendation": {
                "type": strategy_type,
                "display_name": STRATEGY_LABELS[strategy_type],
                "reason": str(strategy.get("reason") or "现有证据未提供策略原因"),
                "confidence": _confidence(strategy.get("confidence")),
                "confidence_basis": str(strategy.get("confidence_basis") or "现有证据不足以说明置信度"),
                "triggered_conditions": _string_list(strategy.get("triggered_conditions")),
                "not_triggered_conditions": _string_list(strategy.get("not_triggered_conditions")),
                "evidence_refs": _valid_refs(strategy.get("evidence_refs"), allowed),
            },
            "evidence_gaps": gaps,
        }

    @staticmethod
    def _failed(trace, message, fallback_strategy="none"):
        strategy_type = fallback_strategy if fallback_strategy in ALLOWED_STRATEGIES else "none"
        return {
            "status": "failed",
            "model": MODEL,
            "trace": trace,
            "dimension_analysis": {
                name: {
                    "status": "EVIDENCE_INSUFFICIENT",
                    "conclusion": "LLM 调用失败，未生成该维度结论",
                    "basis": "保留现有规则根因分析结果",
                    "confidence": 0.0,
                    "evidence_refs": [],
                }
                for name in DIMENSIONS
            },
            "independent_anomalies": [],
            "anomaly_summary": {
                "independent_medium_count": 0,
                "independent_high_count": 0,
                "duplicate_evidence_merged": False,
            },
            "primary_root_cause": {
                "conclusion": "LLM 调用失败，使用规则根因结论",
                "confidence": 0.0,
                "evidence_refs": [],
            },
            "recommendations": [],
            "strategy_recommendation": {
                "type": strategy_type,
                "display_name": STRATEGY_LABELS[strategy_type],
                "reason": "LLM 调用失败，沿用规则策略作为兜底",
                "confidence": 0.0,
                "confidence_basis": "LLM 调用失败，未生成置信度",
                "triggered_conditions": [],
                "not_triggered_conditions": [],
                "evidence_refs": [],
            },
            "evidence_gaps": [message],
        }

    def _write_trace(self, scenario, round_no, evidence, result):
        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(
            self.trace_dir,
            f"m3_llm_trace_{scenario}_r{round_no or 0}.json",
        )
        write_json(path, {
            "trace": result["trace"],
            "status": result["status"],
            "evidence": evidence,
            "structured_output": result,
        })
        return path

    def analyze(self, scenario, round_no, monitor, layer1, layer2, verdict):
        evidence = self.build_evidence(
            scenario, round_no, monitor, layer1, layer2, verdict
        )
        allowed = _evidence_paths(evidence)
        trace = self._trace_meta(scenario, round_no)
        api_key = None
        try:
            from dotenv import find_dotenv, load_dotenv
            from openai import OpenAI

            env_path = find_dotenv(filename=".env", usecwd=True)
            if env_path:
                load_dotenv(env_path)
            api_key = os.getenv("DEEPSEEK_APIKEY")
            if not api_key:
                raise RuntimeError("未找到 DEEPSEEK_APIKEY")

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=120.0,
                max_retries=0,
            )
            # Thinking is forced at the API layer; the prompt does not control it.
            response = client.chat.completions.create(
                model=MODEL,
                messages=self._messages(evidence),
                response_format={"type": "json_object"},
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                stream=False,
            )
            # Deliberately read only final content; reasoning_content is discarded.
            content = response.choices[0].message.content
            payload = json.loads(content)
            result = self._normalize(
                payload, allowed, trace, verdict.get("strategy", "none")
            )
        except Exception as exc:
            message = str(exc)
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
            result = self._failed(
                trace, message[:500], verdict.get("strategy", "none")
            )

        self._write_trace(scenario, round_no, evidence, result)
        return result

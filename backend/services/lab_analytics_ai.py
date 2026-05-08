"""
Lab Analytics AI Service
------------------------
Processes structured lab trends and uses Gemini to generate clinical interpretations
and actionable recommendations. Uses dynamic model routing based on severity to
stay within free-tier limits.
"""

import json
from typing import Dict, Any, List
from utils.gemini_client import call_gemini

def analyze_lab_trends(trends_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of lab trend summaries and enriches them with AI analysis.
    """
    if not trends_data:
        return []

    results = []
    
    for metric_data in trends_data:
        metric_name = metric_data.get("metric_name", "Unknown")
        trend_direction = metric_data.get("trend_direction")
        latest_value = metric_data.get("latest_value")
        data_points = metric_data.get("data_points", [])
        
        # If no significant trend or data points, return a simple rule-based response
        if len(data_points) < 2 or not trend_direction or trend_direction == "stable":
            # Skip AI call for stable/sparse metrics to save quota
            continue

        # Check if the trend is abnormal ("increasing_bad", "decreasing_bad")
        is_bad_trend = trend_direction in ["increasing_bad", "decreasing_bad"]
        
        # Check if latest value is abnormal
        is_latest_abnormal = False
        if data_points:
            is_latest_abnormal = data_points[-1].get("is_abnormal", False)

        # Skip AI if everything is normal to save quota
        if not is_bad_trend and not is_latest_abnormal:
            continue
            
        # Determine model to use
        # For a highly abnormal state or bad trend, use the deep reasoning model, else flash-lite
        model_to_use = "gemini-3.1-flash-lite"
        if is_bad_trend and is_latest_abnormal:
            model_to_use = "gemini-2.5-flash" # Use deep reasoning for critical cases

        history_list = [
            {"date": p.get("date"), "value": p.get("value"),
             "unit": p.get("unit"), "is_abnormal": p.get("is_abnormal")}
            for p in data_points[-5:]
        ]
        prompt = f"""You are the Caretaker.ai Clinical Lab Analyst.
Analyze this lab trend and provide a short, actionable insight for the caregiver.

METRIC: {metric_name}
TREND DIRECTION: {trend_direction}
LATEST VALUE: {latest_value}
HISTORICAL DATA: {json.dumps(history_list)}

OUTPUT FORMAT (Valid JSON only):
{{
  "summary": "One clear sentence explaining what this trend means in plain English.",
  "risk_level": "low | moderate | high | critical",
  "pattern": "A short phrase describing the pattern (e.g., 'Progressive upward trend')",
  "caregiver_action": "One specific action the caregiver should take (e.g., 'Schedule a doctor appointment', 'Monitor next week')",
  "lifestyle_tips": ["Tip 1", "Tip 2"]
}}

Do not include any markdown fences or extra text, just the raw JSON.
"""
        try:
            response = call_gemini({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "temperature": 0.2
                }
            }, timeout=30, caller="[lab_analytics_ai]", model_override=model_to_use)

            if response and "candidates" in response and response["candidates"]:
                parts = response["candidates"][0]["content"]["parts"]
                raw_text = ""
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        raw_text = part["text"].strip()
                if not raw_text:
                    raw_text = parts[-1].get("text", "").strip()
                
                # Clean JSON markdown fences
                import re
                cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.DOTALL)
                cleaned = re.sub(r"\s*```", "", cleaned, flags=re.DOTALL).strip()
                
                parsed_insight = json.loads(cleaned)
                parsed_insight["model_used"] = model_to_use
                parsed_insight["metric_name"] = metric_name
                
                results.append(parsed_insight)
                
        except Exception as e:
            print(f"[lab_analytics_ai] Failed to analyze {metric_name}: {e}")
            continue

    return results

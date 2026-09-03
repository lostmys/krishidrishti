from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_KNOWLEDGE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "knowledge" / "disease_knowledge.yaml"
)

SUPPORTED_LANGUAGES = {"en", "hi", "mr"}
DEFAULT_LANGUAGE = "en"

CROP_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "en": {"tomato": "Tomato", "soyabean": "Soybean", "cotton": "Cotton"},
    "hi": {"tomato": "टमाटर", "soyabean": "सोयाबीन", "cotton": "कपास"},
    "mr": {"tomato": "टोमॅटो", "soyabean": "सोयाबीन", "cotton": "कापूस"},
}


class DiseaseKnowledgeService:
    """Provides curated, multilingual disease knowledge, honest visual uncertainty,

    and agricultural safety advice.
    """

    def __init__(self, knowledge_path: str | Path | None = None) -> None:
        path = Path(knowledge_path or _DEFAULT_KNOWLEDGE_PATH).resolve()
        if not path.is_file():
            project_root = next(
                (c for c in (Path.cwd(), *Path.cwd().parents) if (c / "pyproject.toml").is_file()),
                Path.cwd(),
            )
            path = project_root / "data" / "knowledge" / "disease_knowledge.yaml"

        self.knowledge_data: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                content = yaml.safe_load(stream)
                if isinstance(content, dict) and "diseases" in content:
                    self.knowledge_data = content["diseases"]

        # Fast lookup: (crop.lower(), diagnosis.lower()) -> entry
        self._lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in self.knowledge_data:
            crop_key = str(entry.get("crop", "")).strip().lower()
            diag_key = str(entry.get("diagnosis", "")).strip().lower()
            self._lookup[(crop_key, diag_key)] = entry

    def get_crop_name(self, crop: str | None, language: str = "en") -> str:
        """Return localized crop name."""
        if not crop:
            return ""
        lang = self._norm_lang(language)
        return CROP_DISPLAY_NAMES.get(lang, {}).get(crop.strip().lower(), crop)

    def get_display_name(
        self,
        crop: str | None,
        diagnosis: str | None,
        language: str = "en",
    ) -> str:
        """Return localized display name for a given crop and diagnosis."""
        if not diagnosis:
            return ""
        lang = self._norm_lang(language)
        crop_norm = str(crop or "").strip().lower()
        diag_norm = str(diagnosis or "").strip().lower()
        entry = self._lookup.get((crop_norm, diag_norm))
        if entry:
            translations = entry.get("translations", {})
            t_data = translations.get(lang) or translations.get(DEFAULT_LANGUAGE, {})
            return t_data.get("display_name", diagnosis)
        return self._format_unknown_display_name(diagnosis, lang)

    def generate_analysis(
        self,
        crop: str | None,
        diagnosis: str | None,
        confidence: float,
        status: str,
        language: str = "en",
        is_ambiguous: bool = False,
        top_predictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Generate honest, non-hallucinatory analysis distinguishing classifier predictions

        from general disease information.
        """
        lang = self._norm_lang(language)
        crop_name = self.get_crop_name(crop, lang)
        disp_name = self.get_display_name(crop, diagnosis, lang)
        conf_pct = round(confidence * 100, 1)

        # Build differential list
        diff_list: list[dict[str, Any]] = []
        if top_predictions and len(top_predictions) > 1:
            for pred in top_predictions[1:]:
                alt_name = self.get_display_name(pred.get("crop"), pred.get("diagnosis"), language=lang)
                diff_list.append({
                    "crop": pred.get("crop"),
                    "diagnosis": pred.get("diagnosis"),
                    "display_name": alt_name,
                    "confidence": pred.get("confidence", 0.0),
                })

        # Summary
        if status == "IMAGE_QUALITY_REJECTED":
            if lang == "hi":
                summary = "छवि गुणवत्ता परीक्षण विफल। अपलोड की गई फोटो विश्वसनीय एआई निदान के लिए उपयुक्त नहीं है।"
            elif lang == "mr":
                summary = "फोटो गुणवत्ता निकष अपूर्ण. अपलोड केलेला फोटो अचूक एआय निदानासाठी योग्य दर्जाचा नाही."
            else:
                summary = "Image quality check failed. The uploaded image is not suitable for reliable AI diagnosis."
        elif status == "LOW_CONFIDENCE":
            if lang == "hi":
                summary = f"मॉडल इस {crop_name} छवि के लिए उच्च विश्वास के साथ रोग की पहचान नहीं कर सका (प्राथमिक अनुमान: {disp_name}, {conf_pct}% विश्वास)।"
            elif lang == "mr":
                summary = f"मॉडेल या {crop_name}च्या फोटोसाठी निश्चित रोग ओळखू शकले नाही (प्राथमिक अंदाज: {disp_name}, {conf_pct}% खात्री)."
            else:
                summary = f"The model could not identify a clear diagnosis for this {crop_name} image with high confidence (top prediction: {disp_name} at {conf_pct}%)."
        elif status == "REVIEW_RECOMMENDED":
            if lang == "hi":
                summary = f"मॉडल का इस {crop_name} छवि के लिए मुख्य पूर्वानुमान {disp_name} ({conf_pct}% विश्वास) है, लेकिन मध्यम अनिश्चितता के साथ।"
            elif lang == "mr":
                summary = f"मॉडेलचा या {crop_name}च्या फोटोसाठी मुख्य अंदाज {disp_name} ({conf_pct}% खात्री) आहे, परंतु मध्यम अनिश्चिततेसह."
            else:
                summary = f"The model's primary prediction for this {crop_name} image is {disp_name} ({conf_pct}% confidence), but with moderate certainty."
        else:  # AI_CONFIDENT
            if lang == "hi":
                summary = f"मॉडल ने {conf_pct}% विश्वास के साथ इस {crop_name} की छवि को {disp_name} के रूप में वर्गीकृत किया है।"
            elif lang == "mr":
                summary = f"मॉडेलने {conf_pct}% खात्रीसह या {crop_name}च्या फोटोचे वर्गीकरण {disp_name} असे केले आहे."
            else:
                summary = f"The model classified this {crop_name} image as {disp_name} with {conf_pct}% confidence."

        # Confidence explanation
        if status == "IMAGE_QUALITY_REJECTED":
            if lang == "hi":
                conf_exp = "अपलोड की गई फोटो न्यूनतम गुणवत्ता मानकों (अत्यधिक धुंधलापन, कम रेजोल्यूशन या खराब रोशनी) को पूरा नहीं करती है।"
            elif lang == "mr":
                conf_exp = "अपलोड केलेला फोटो किमान गुणवत्ता निकष (जास्त ब्लर, अपुरा रेझोल्यूशन किंवा अपुरा प्रकाश) पूर्ण करत नाही."
            else:
                conf_exp = "The uploaded photo does not meet minimum quality criteria (excessive blur, inadequate resolution, or poor lighting)."
        elif status == "LOW_CONFIDENCE":
            if lang == "hi":
                conf_exp = "कम मॉडल विश्वास (<45%)। छवि के लक्षण किसी एक प्रशिक्षित वर्ग से निर्णायक रूप से मेल नहीं खाते हैं।"
            elif lang == "mr":
                conf_exp = "कमी मॉडेल खात्री (<४५%). फोटोतील लक्षणे कोणत्याही एका रोगाशी स्पष्टपणे जुळत नाहीत."
            else:
                conf_exp = "Low model confidence (<45%). The visual features do not match any single trained class decisively."
        elif status == "REVIEW_RECOMMENDED":
            if lang == "hi":
                conf_exp = "मध्यम मॉडल विश्वास। खेत की परिस्थितियों में समान रोगों के पत्तियों के लक्षण आपस में मिल सकते हैं।"
            elif lang == "mr":
                conf_exp = "मध्यम मॉडेल खात्री. शेतातील नैसर्गिक वातावरणात वेगवेगळ्या रोगांची पानांवरील लक्षणे दिसायला सारखी असू शकतात."
            else:
                conf_exp = "Moderate model confidence. Visual foliar symptoms can overlap between similar conditions in field environments."
        else:
            if lang == "hi":
                conf_exp = "प्रशिक्षित रोग लक्षणों के आधार पर उच्च मॉडल विश्वास। नीचे दी गई सामान्य रोग संदर्भ जानकारी देखें।"
            elif lang == "mr":
                conf_exp = "प्रशिक्षित रोग लक्षणांवर आधारित उच्च मॉडेल खात्री. अधिक माहितीसाठी खाली दिलेला सर्वसाधारण रोग संदर्भ पहा."
            else:
                conf_exp = "High model confidence based on learned visual disease features. Review standard disease reference below."

        # Uncertainty note
        uncertainty_note = ""
        if status == "LOW_CONFIDENCE":
            if lang == "hi":
                uncertainty_note = "निदान अनिश्चित है। इस फोटो से लक्षणों की पुष्टि नहीं की जा सकती। व्यक्तिगत निदान के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।"
            elif lang == "mr":
                uncertainty_note = "निदान अनिश्चित आहे. या फोटोवरून निश्चित लक्षणे स्पष्ट होत नाहीत. प्रत्यक्ष पाहणीसाठी स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या."
            else:
                uncertainty_note = "Prediction is uncertain. Visual symptoms cannot be confirmed from this photo. Consult a local agricultural expert for an in-person assessment."
        elif status == "REVIEW_RECOMMENDED" or is_ambiguous:
            if diff_list:
                top_alt = diff_list[0]
                alt_name = top_alt.get("display_name", "")
                alt_conf = top_alt.get("confidence", 0.0) * 100
                if lang == "hi":
                    uncertainty_note = f"वैकल्पिक संभावना: {alt_name} ({alt_conf:.1f}% विश्वास)। खेत में पत्तियों के धब्बों, झुलसा या तनाव के लक्षण दिखने में समान हो सकते हैं; कोई भी उपचार करने से पहले स्थानीय कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से पुष्टि करने की सलाह दी जाती है।"
                elif lang == "mr":
                    uncertainty_note = f"पर्यायी शक्यता: {alt_name} ({alt_conf:.1f}% खात्री). शेतातील नैसर्गिक वातावरणात पानांवरील विविध ठिपके, करपा किंवा रोगांची लक्षणे एकमेकांसारखी दिसू शकतात; कोणताही उपाय करण्यापूर्वी स्थानिक कृषी विज्ञान केंद्र (KVK) किंवा कृषी अधिकाऱ्याकडून प्रत्यक्ष खात्री करून घेण्याचा सल्ला दिला जातो."
                else:
                    uncertainty_note = f"Alternative possibility: {alt_name} ({alt_conf:.1f}% confidence). Visual symptoms of leaf spots, blights, or foliar stress can overlap under field conditions; visual verification by a local agricultural extension officer (KVK) is recommended before applying any treatment."
            else:
                if lang == "hi":
                    uncertainty_note = "खेत में पत्तियों के लक्षणों में समानता हो सकती है; कोई भी उपचार करने से पहले स्थानीय कृषि अधिकारी या केवीके (KVK) से पुष्टि करें।"
                elif lang == "mr":
                    uncertainty_note = "शेतातील परिस्थितीत पानांवरील लक्षणांमध्ये साधर्म्य असू शकते; कोणताही उपाय करण्यापूर्वी स्थानिक कृषी अधिकारी किंवा केव्हीके (KVK) कडून खात्री करा."
                else:
                    uncertainty_note = "Symptoms can overlap under field conditions; please verify with a local agricultural extension officer or KVK before applying treatment."

        # Visual limitations note
        if lang == "hi":
            vis_note = "सूचना: यह एआई प्रणाली प्रशिक्षित पत्ती छवि पैटर्न के आधार पर स्वचालित प्रारंभिक जांच प्रदान करती है। यह सूक्ष्मदर्शीय जांच, बीजाणु पहचान या रोगजनक संवर्धन नहीं करती है। उपचार संबंधी कोई भी निर्णय लेने से पहले हमेशा स्थानीय कृषि विशेषज्ञों से पुष्टि करें।"
        elif lang == "mr":
            vis_note = "सूचना: ही एआय प्रणाली केवळ प्रशिक्षित पानांच्या प्रतिमांवर आधारित स्वयंचलित प्राथमिक तपासणी करते. यात सूक्ष्मदर्शक तपासणी किंवा बुरशीच्या बीजाणूंची प्रयोगशाळा तपासणी केली जात नाही. कोणताही उपाय करण्यापूर्वी नेहमी स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या."
        else:
            vis_note = "Notice: The AI system provides automated visual screening based on trained leaf image patterns. It does not perform microscopic examination, spore detection, or pathogen culture verification. Always verify symptoms with local agricultural authorities before making treatment decisions."

        return {
            "summary": summary,
            "confidence_explanation": conf_exp,
            "uncertainty_note": uncertainty_note,
            "visual_limitations_note": vis_note,
        }

    def get_advice(
        self,
        crop: str | None,
        diagnosis: str | None,
        status: str,
        language: str = "en",
        is_ambiguous: bool = False,
        top_predictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return structured, safety-checked advice payload for the given status & diagnosis."""
        lang = self._norm_lang(language)

        # 1. IMAGE QUALITY REJECTED
        if status == "IMAGE_QUALITY_REJECTED":
            return self._image_quality_rejected_advice(lang)

        # 2. LOW CONFIDENCE
        if status == "LOW_CONFIDENCE":
            return self._low_confidence_advice(lang, top_predictions)

        # 3. Normal / AI_CONFIDENT / REVIEW_RECOMMENDED
        if not crop or not diagnosis:
            return self._low_confidence_advice(lang, top_predictions)

        crop_norm = str(crop).strip().lower()
        diag_norm = str(diagnosis).strip().lower()
        entry = self._lookup.get((crop_norm, diag_norm))

        if not entry:
            advice = self._fallback_advice(diagnosis, lang)
        else:
            translations = entry.get("translations", {})
            t_data = translations.get(lang) or translations.get(DEFAULT_LANGUAGE, {})
            disp = t_data.get("display_name", diagnosis)
            symptoms = list(t_data.get("symptoms", []))
            advice = {
                "display_name": disp,
                "description": t_data.get("description", ""),
                "typical_symptoms": symptoms,
                "symptoms": symptoms,  # Alias for backward compatibility
                "immediate_actions": list(t_data.get("immediate_actions", [])),
                "prevention": list(t_data.get("prevention", [])),
                "severity_guidance": t_data.get("severity_guidance", ""),
                "source": entry.get("source", "KrishiDrishti Knowledge Base"),
            }

        # Build differential diagnosis list
        diff_list: list[dict[str, Any]] = []
        if top_predictions and len(top_predictions) > 1:
            for pred in top_predictions[1:]:
                alt_name = self.get_display_name(pred.get("crop"), pred.get("diagnosis"), language=lang)
                diff_list.append({
                    "crop": pred.get("crop"),
                    "diagnosis": pred.get("diagnosis"),
                    "display_name": alt_name,
                    "confidence": pred.get("confidence", 0.0),
                })
        advice["differential_diagnosis"] = diff_list

        # Status & Ambiguity guidance
        analysis_info = self.generate_analysis(
            crop, diagnosis, 0.0, status, language=lang, is_ambiguous=is_ambiguous, top_predictions=top_predictions
        )
        advice["status_guidance"] = analysis_info["confidence_explanation"]
        advice["ambiguity_note"] = analysis_info["uncertainty_note"]

        if status == "REVIEW_RECOMMENDED" and analysis_info["uncertainty_note"]:
            if not any(analysis_info["uncertainty_note"] in a for a in advice["immediate_actions"]):
                advice["immediate_actions"].insert(0, analysis_info["uncertainty_note"])

        return advice

    @staticmethod
    def _norm_lang(language: str | None) -> str:
        lang = str(language or "").lower().strip()
        return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    @staticmethod
    def _format_unknown_display_name(diagnosis: str, lang: str) -> str:
        clean = diagnosis.replace("___", " - ").replace("_", " ")
        if lang == "hi":
            return f"{clean} (अज्ञात रोग / Unknown)"
        if lang == "mr":
            return f"{clean} (अज्ञात रोग / Unknown)"
        return clean

    @staticmethod
    def _low_confidence_advice(lang: str, top_predictions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if lang == "hi":
            return {
                "display_name": "कम विश्वास परिणाम (Low Confidence)",
                "description": "एआई मॉडल इस छवि से बीमारी की निश्चित पहचान नहीं कर सका।",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "एक पत्ती की स्पष्ट और अच्छी रोशनी वाली फोटो फिर से लें।",
                    "सटीक पहचान के लिए स्थानीय कृषि विस्तार अधिकारी या केवीके (KVK) से परामर्श करें।",
                ],
                "prevention": ["फसल की फोटो लेते समय अच्छी रोशनी और स्पष्ट फोकस सुनिश्चित करें।"],
                "severity_guidance": "अनिश्चित",
                "source": "कृषिदृष्टि सुरक्षा नीति",
                "status_guidance": "कम विश्वास परिणाम। कृपया स्पष्ट फोटो के साथ पुनः प्रयास करें या कृषि विशेषज्ञ से संपर्क करें।",
                "ambiguity_note": "निदान अनिश्चित है। प्रत्यक्ष मूल्यांकन के लिए कृषि विशेषज्ञ से संपर्क करें।",
                "differential_diagnosis": [],
            }
        if lang == "mr":
            return {
                "display_name": "कमी खात्रीचा निकाल (Low Confidence)",
                "description": "एआय मॉडेल या फोटोवरून रोगाची निश्चित ओळख करू शकले नाही.",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "एका पानाचा स्पष्ट आणि चांगल्या प्रकाशातील फोटो पुन्हा घ्या.",
                    "तज्ज्ञ सल्ल्यासाठी स्थानिक कृषी विस्तार अधिकारी किंवा केव्हीके (KVK) शी संपर्क साधा.",
                ],
                "prevention": ["पिकाचा फोटो काढताना योग्य प्रकाश आणि स्पष्ट फोकस ठेवा."],
                "severity_guidance": "अनिश्चित",
                "source": "कृषिदृष्टी सुरक्षा धोरण",
                "status_guidance": "कमी खात्रीचा निकाल. कृपया स्पष्ट फोटोसह पुन्हा प्रयत्न करा किंवा कृषी तज्ज्ञांशी संपर्क साधा.",
                "ambiguity_note": "निदान अनिश्चित आहे. प्रत्यक्ष पाहणीसाठी कृषी तज्ज्ञांचा सल्ला घ्या.",
                "differential_diagnosis": [],
            }
        return {
            "display_name": "Low Confidence Result",
            "description": "The AI model could not confidently identify the crop disease from this image.",
            "typical_symptoms": [],
            "symptoms": [],
            "immediate_actions": [
                "Retake a clear, well-lit photo focusing on a single leaf.",
                "Consult a local agricultural extension officer (KVK) for expert in-person diagnosis.",
            ],
            "prevention": ["Ensure good lighting and sharp focus when photographing crops."],
            "severity_guidance": "Uncertain",
            "source": "KrishiDrishti Safety Policy",
            "status_guidance": "Low confidence result. Retake photo under good lighting or consult an agricultural expert.",
            "ambiguity_note": "Prediction is uncertain. Consult an agricultural expert for in-person assessment.",
            "differential_diagnosis": [],
        }

    @staticmethod
    def _image_quality_rejected_advice(lang: str) -> dict[str, Any]:
        if lang == "hi":
            return {
                "display_name": "छवि अस्वीकृत - खराब गुणवत्ता",
                "description": "अपलोड की गई फोटो की गुणवत्ता सटीक एआई निदान के लिए अपर्याप्त है।",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "धुंधलेपन से बचने के लिए कैमरे को स्थिर रखें।",
                    "पर्याप्त प्राकृतिक रोशनी सुनिश्चित करें।",
                    "लक्षणों वाली एक ही पत्ती की पास से (15-30 सेमी) फोटो लें।",
                    "अत्यधिक छाया या बहुत दूरी से बचें।",
                ],
                "prevention": ["दिन के उजाले में 15-30 सेमी की दूरी से फोटो लें।"],
                "severity_guidance": "गुणवत्ता अस्वीकृत",
                "source": "कृषिदृष्टि गुणवत्ता नियंत्रण",
                "status_guidance": "अपलोड की गई फोटो की गुणवत्ता परीक्षण पास नहीं कर सकी। कृपया नई फोटो लें।",
                "ambiguity_note": "",
                "differential_diagnosis": [],
            }
        if lang == "mr":
            return {
                "display_name": "फोटो नाकारला - निकृष्ट दर्जा",
                "description": "अपलोड केलेला फोटो अचूक निदानासाठी योग्य दर्जाचा नाही.",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "फोटो अस्पष्ट येऊ नये म्हणून कॅमेरा स्थिर धरा.",
                    "पुरेसा नैसर्गिक प्रकाश असल्याची खात्री करा.",
                    "लक्षणे असलेल्या एका पानाचा जवळून (१५-३० सेमी) फोटो घ्या.",
                    "अति सावली किंवा खूप लांबून फोटो घेणे टाळा.",
                ],
                "prevention": ["दिवसाच्या प्रकाशात १५-३० सेमी अंतरावरून फोटो घ्या."],
                "severity_guidance": "दर्जा नाकारला",
                "source": "कृषिदृष्टी गुणवत्ता नियंत्रण",
                "status_guidance": "अपलोड केलेला फोटो गुणवत्ता निकष पूर्ण करत नाही. कृपया नवीन फोटो घ्या.",
                "ambiguity_note": "",
                "differential_diagnosis": [],
            }
        return {
            "display_name": "Image Rejected - Poor Quality",
            "description": "The uploaded photo does not meet quality requirements for accurate AI diagnosis.",
            "typical_symptoms": [],
            "symptoms": [],
            "immediate_actions": [
                "Hold the camera steady to avoid motion blur.",
                "Ensure adequate natural daylight on the leaf.",
                "Capture a close-up (15-30 cm) of a single leaf showing disease symptoms.",
                "Avoid extreme shadows, reflections, or photographing from a distance.",
            ],
            "prevention": ["Take photos in clear daylight from 15-30 cm distance."],
            "severity_guidance": "Quality Rejected",
            "source": "KrishiDrishti Quality Gate",
            "status_guidance": "Image quality check failed. Please retake photo following guidance below.",
            "ambiguity_note": "",
            "differential_diagnosis": [],
        }

    @staticmethod
    def _fallback_advice(diagnosis: str, lang: str) -> dict[str, Any]:
        clean = diagnosis.replace("___", " - ").replace("_", " ")
        if lang == "hi":
            return {
                "display_name": f"{clean} (अज्ञात रोग)",
                "description": f"पूर्वानुमानित स्थिति: {clean}। इस स्थिति के लिए विशिष्ट संदर्भ जानकारी उपलब्ध नहीं है।",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "अपनी फसल और क्षेत्र के लिए उपयुक्त स्वीकृत उपचार हेतु स्थानीय कृषि विस्तार अधिकारी या केवीके से परामर्श करें।"
                ],
                "prevention": ["खेत की स्वच्छता बनाए रखें और नियमित फसल निरीक्षण करें।"],
                "severity_guidance": "अज्ञात",
                "source": "कृषिदृष्टि सामान्य मार्गदर्शन",
                "status_guidance": "अज्ञात रोग स्थिति।",
                "ambiguity_note": "",
                "differential_diagnosis": [],
            }
        if lang == "mr":
            return {
                "display_name": f"{clean} (अज्ञात रोग)",
                "description": f"अंदाजित स्थिती: {clean}. या स्थितीसाठी विशिष्ट संदर्भ माहिती उपलब्ध नाही.",
                "typical_symptoms": [],
                "symptoms": [],
                "immediate_actions": [
                    "तुमच्या पिकासाठी आणि क्षेत्रासाठी योग्य उपायांसाठी स्थानिक कृषी विस्तार अधिकारी किंवा केव्हीकेचा सल्ला घ्या."
                ],
                "prevention": ["शेताची स्वच्छता राखा आणि नियमित पिकाची पाहणी करा."],
                "severity_guidance": "अज्ञात",
                "source": "कृषिदृष्टी सर्वसाधारण मार्गदर्शन",
                "status_guidance": "अज्ञात रोग स्थिती.",
                "ambiguity_note": "",
                "differential_diagnosis": [],
            }
        return {
            "display_name": f"{clean} (Unknown Disease)",
            "description": f"Predicted condition: {clean}. Specific reference data is not available in knowledge base.",
            "typical_symptoms": [],
            "symptoms": [],
            "immediate_actions": [
                "Consult the local agricultural extension officer (KVK) for an approved treatment suitable for your crop and region."
            ],
            "prevention": ["Maintain proper field sanitation and monitor crops regularly."],
            "severity_guidance": "Unknown",
            "status_guidance": "Unknown disease condition.",
            "ambiguity_note": "",
            "differential_diagnosis": [],
        }

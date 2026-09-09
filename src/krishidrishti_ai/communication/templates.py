"""Multilingual WhatsApp message templates for KrishiDrishti Farmer Flow.

Languages supported: Marathi (mr), Hindi (hi), English (en).
Default primary language for Maharashtra SIH 26131 scope is Marathi (mr).
"""
from __future__ import annotations

from typing import Any

# ============================================================================
# GREETING & ONBOARDING TEMPLATES
# ============================================================================

WELCOME_TEMPLATES = {
    "mr": (
        "🌾 *नमस्कार! कृषीदृष्टी (KrishiDrishti) शेती सहाय्यकामध्ये आपले स्वागत आहे.*\n\n"
        "आम्ही पीक रोग निदान आणि उपग्रह देखरेखीद्वारे आपल्या शेताचे रक्षण करतो.\n\n"
        "कृपया आपल्या पिकाची निवड करा:\n"
        "1️⃣ *टोमॅटो (Tomato)*\n"
        "2️⃣ *सोयाबीन (Soybean)*\n"
        "3️⃣ *कापूस (Cotton)*\n\n"
        "👉 _पर्याय क्रमांक (1, 2, किंवा 3) किंवा पिकाचे नाव पाठवा._"
    ),
    "hi": (
        "🌾 *नमस्ते! कृषिदृष्टि (KrishiDrishti) में आपका स्वागत है।*\n\n"
        "हम फसल रोग निदान और उपग्रह निगरानी द्वारा आपके खेत की सुरक्षा करते हैं।\n\n"
        "कृपया अपनी फसल का चयन करें:\n"
        "1️⃣ *टमाटर (Tomato)*\n"
        "2️⃣ *सोयाबीन (Soybean)*\n"
        "3️⃣ *कपास (Cotton)*\n\n"
        "👉 _विकल्प संख्या (1, 2, या 3) या फसल का नाम भेजें।_"
    ),
    "en": (
        "🌾 *Welcome to KrishiDrishti Agricultural Surveillance Assistant!*\n\n"
        "We protect your crops using real-time Image AI diagnostics and satellite screening.\n\n"
        "Please select your crop:\n"
        "1️⃣ *Tomato*\n"
        "2️⃣ *Soybean*\n"
        "3️⃣ *Cotton*\n\n"
        "👉 _Reply with the number (1, 2, or 3) or crop name._"
    ),
}

CROP_CONFIRMED_PHOTO_PROMPT = {
    "mr": (
        "✅ *पीक निवडले: {crop_display}*\n\n"
        "📸 कृपया बाधित पानाचा किंवा झाडाचा *स्पष्ट फोटो* पाठवा.\n\n"
        "💡 *चांगल्या निदानासाठी सूचना:*\n"
        "• रोगाचे डाग/लक्षण स्पष्ट दिसू द्या.\n"
        "• कॅमेरा स्थिर ठेवा व सूर्यप्रकाश पुरेसा असावा.\n"
        "• आपण व्हॉइस मेसेज (ऑडिओ) द्वारेही समस्या सांगू शकता."
    ),
    "hi": (
        "✅ *चयनित फसल: {crop_display}*\n\n"
        "📸 कृपया प्रभावित पत्ती या पौधे का *साफ फोटो* भेजें।\n\n"
        "💡 *सटीक निदान के लिए सुझाव:*\n"
        "• बीमारी के लक्षण या धब्बे स्पष्ट दिखने चाहिए।\n"
        "• कैमरा स्थिर रखें और पर्याप्त प्रकाश में फोटो लें।"
    ),
    "en": (
        "✅ *Selected Crop: {crop_display}*\n\n"
        "📸 Please send a *clear photo* of the affected leaf or crop lesion.\n\n"
        "💡 *Tips for best AI diagnosis:*\n"
        "• Ensure disease symptoms/spots are clearly focused.\n"
        "• Take photo in good natural lighting without motion blur.\n"
        "• You can also send a voice message describing your problem."
    ),
}

# ============================================================================
# DIAGNOSIS RESPONSE CARD
# ============================================================================

def format_diagnosis_response(
    case_id: str,
    crop: str,
    disease: str,
    confidence: float,
    status: str,
    advice: dict[str, Any] | None = None,
    satellite_info: dict[str, Any] | None = None,
    fused_risk: str = "HIGH",
    risk_score: int = 81,
    language: str = "mr",
) -> str:
    """Format the comprehensive localized WhatsApp diagnosis card sent to farmer."""
    advice = advice or {}
    clean_disease = disease.replace("___", " — ").replace("_", " ")
    conf_pct = f"{confidence * 100:.1f}%"
    immediate = advice.get("immediate_action", "स्थानिक कृषी सल्लागाराशी संपर्क साधा.")

    if language == "mr":
        risk_label = "उच्च जोखीम (HIGH)" if fused_risk == "HIGH" else "मध्यम जोखीम (MEDIUM)" if fused_risk == "MEDIUM" else "कमी जोखीम (LOW)"
        msg = (
            f"🔬 *कृषीदृष्टी AI रोग निदान अहवाल*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *तक्रार क्र (Case ID):* `{case_id}`\n"
            f"🌱 *पीक:* {crop.title()}\n"
            f"🦠 *संभाव्य रोग:* *{clean_disease}*\n"
            f"📊 *AI अचूकता (Confidence):* {conf_pct}\n"
            f"⚠️ *एकत्रित जोखीम स्तर:* {risk_label} ({risk_score}/100)\n\n"
            f"🛰️ *उपग्रह तपासणी (Sentinel Satellite):*\n"
            f"कॅनॉपी वनस्पती ताण व आर्द्रता तूट नोंदवली गेली आहे.\n\n"
            f"💡 *तातडीचा कृषी सल्ला:*\n"
            f"{immediate}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👨‍🌾 *पुढील पायरी:* ही केस तालुका कृषी अधिकारी (DAO/KVK) यांच्या डॅशबोर्डवर पडताळणीसाठी पाठवली आहे. "
            f"तज्ज्ञांच्या मान्यतेनंतर अंतिम सल्ला पाठवला जाईल.\n\n"
            f"📞 मदतीसाठी: *किसान कॉल सेंटर १८००-१८०-१५५१*"
        )
        return msg

    elif language == "hi":
        risk_label = "उच्च जोखिम (HIGH)" if fused_risk == "HIGH" else "मध्यम जोखिम (MEDIUM)"
        msg = (
            f"🔬 *कृषिदृष्टि AI रोग निदान रिपोर्ट*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *केस आईडी:* `{case_id}`\n"
            f"🌱 *फसल:* {crop.title()}\n"
            f"🦠 *संभावित रोग:* *{clean_disease}*\n"
            f"📊 *AI सटीकता:* {conf_pct}\n"
            f"⚠️ *जोखिम स्तर:* {risk_label} ({risk_score}/100)\n\n"
            f"💡 *त्वरित सलाह:*\n"
            f"{immediate}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👨‍🌾 यह रिपोर्ट कृषि अधिकारी के पास सत्यापन हेतु भेज दी गई है।"
        )
        return msg

    else:
        msg = (
            f"🔬 *KrishiDrishti AI Diagnosis Report*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *Case ID:* `{case_id}`\n"
            f"🌱 *Crop:* {crop.title()}\n"
            f"🦠 *Detected Condition:* *{clean_disease}*\n"
            f"📊 *Confidence:* {conf_pct}\n"
            f"⚠️ *Multi-Source Risk:* {fused_risk} ({risk_score}/100)\n\n"
            f"🛰️ *Satellite Screening (Sentinel-2/1):*\n"
            f"Vegetation deficit and moisture anomaly detected in parcel canopy.\n\n"
            f"💡 *Immediate Advisory Action:*\n"
            f"{immediate}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👨‍🌾 *Next Step:* Case forwarded to District Agricultural Officer for expert review. "
            f"Verified treatment plan will be delivered upon confirmation."
        )
        return msg


# ============================================================================
# EXPERT VERIFICATION CONFIRMATION (Sent after officer clicks CONFIRM)
# ============================================================================

def format_officer_verified_alert(
    case_id: str,
    crop: str,
    diagnosis: str,
    officer_name: str = "Dr. S. Kulkarni (District Agronomist)",
    notes: str = "Diagnosis verified. Standard triazole fungicide spray recommended within 48h.",
    language: str = "mr",
) -> str:
    """Alert dispatched to farmer when Officer confirms case on Dashboard."""
    clean_diagnosis = diagnosis.replace("___", " — ").replace("_", " ")

    if language == "mr":
        return (
            f"✅ *कृषी अधिकारी पडताळणी पूर्ण (Case #{case_id})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👨‍🌾 प्रिय शेतकरी मित्र, आपल्या {crop.title()} पिकावरील समस्येची कृषी अधिकाऱ्यांनी प्रत्यक्ष तपासणी केली आहे.\n\n"
            f"📋 *अंतिम निदान:* *{clean_diagnosis}*\n"
            f"👨‍💼 *पडताळणी अधिकारी:* {officer_name}\n"
            f"📝 *शिफारस / उपाययोजना:*\n"
            f"{notes}\n\n"
            f"⏰ पुढील ४८ तासांत औषध फवारणी करा व पिकाच्या स्थितीबाबत पुन्हा अपडेट द्या.\n"
            f"— *कृषी विभाग & कृषीदृष्टी पथक*"
        )
    elif language == "hi":
        return (
            f"✅ *कृषि अधिकारी सत्यापन पूर्ण (Case #{case_id})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👨‍🌾 प्रिय किसान मित्र, आपकी {crop.title()} फसल की समस्या की कृषि अधिकारी द्वारा पुष्टि कर दी गई है।\n\n"
            f"📋 *अंतिम निदान:* *{clean_diagnosis}*\n"
            f"👨‍💼 *सत्यापन अधिकारी:* {officer_name}\n"
            f"📝 *सिफारिश / उपाय:*\n"
            f"{notes}\n\n"
            f"⏰ अगले 48 घंटों में उपचार शुरू करें और फसल की स्थिति साझा करें।\n"
            f"— *कृषि विभाग & कृषिदृष्टि टीम*"
        )
    else:
        return (
            f"✅ *Agricultural Officer Verification Completed (Case #{case_id})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Dear Farmer, your {crop.title()} case has been reviewed and confirmed by agricultural experts.\n\n"
            f"📋 *Confirmed Diagnosis:* *{clean_diagnosis}*\n"
            f"👨‍💼 *Reviewer:* {officer_name}\n"
            f"📝 *Prescribed Advisory / Treatment:*\n"
            f"{notes}\n\n"
            f"⏰ Apply recommended treatment within 48 hours and send a follow-up status photo.\n"
            f"— *KrishiDrishti Officer Team*"
        )


# ============================================================================
# IMAGE QUALITY REJECTION TEMPLATE
# ============================================================================

def format_image_quality_failed_message(
    reason: str | None = None,
    blur_score: float | None = None,
    language: str = "mr",
) -> str:
    """Formatted message dispatched to farmer when uploaded photo fails quality check."""
    reason_map = {
        "Image is too blurry": {
            "mr": "फोटो खूप अंधुक / अस्पष्ट (blurry) आहे",
            "hi": "फोटो बहुत धुंधली (blurry) है",
            "en": "image is too blurry",
        },
        "Image resolution is too low": {
            "mr": "फोटोचा रेझोल्यूशन खूप कमी आहे",
            "hi": "फोटो का रिज़ॉल्यूशन बहुत कम है",
            "en": "image resolution is too low",
        },
        "Image brightness is unsuitable": {
            "mr": "फोटोमध्ये प्रकाश अपुरा किंवा खूप जास्त आहे",
            "hi": "फोटो में रोशनी अपर्याप्त या अत्यधिक है",
            "en": "image lighting is unsuitable",
        },
        "Unable to decode image": {
            "mr": "फोटो फॉरमॅट वाचता आला नाही",
            "hi": "फोटो प्रारूप पढ़ा नहीं जा सका",
            "en": "unable to decode image file",
        },
    }
    r_key = reason or "Image is too blurry"
    trans = reason_map.get(r_key, {"mr": r_key, "hi": r_key, "en": r_key})

    if language == "mr":
        return (
            "⚠️ *फोटोची गुणवत्ता तपासणी अयशस्वी (Image Quality Check Failed)*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"आपण पाठवलेला फोटो अचूक निदानासाठी योग्य दर्जाचा नाही ({trans['mr']}).\n\n"
            "💡 *कृपया पुढील दक्षता घेऊन पुन्हा स्पष्ट फोटो पाठवा:*\n"
            "• कॅमेरा स्थिर धरा, जेणेकरून फोटो अस्पष्ट येणार नाही.\n"
            "• पानावर पुरेसा नैसर्गिक सूर्यप्रकाश असल्याची खात्री करा.\n"
            "• लक्षणे असलेल्या एकाच पानाचा १५-३० सेमी अंतरावरून जवळून (Close-up) फोटो काढा.\n"
            "• खूप लांबून किंवा सावलीत फोटो काढणे टाळा.\n\n"
            "📷 *कृपया नवीन स्पष्ट फोटो पाठवा.*"
        )
    elif language == "hi":
        return (
            "⚠️ *फोटो गुणवत्ता परीक्षण विफल (Image Quality Check Failed)*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"भेजी गई फोटो सटीक निदान के लिए उपयुक्त नहीं है ({trans['hi']})।\n\n"
            "💡 *कृपया निम्नलिखित सुझावों के साथ पुनः फोटो भेजें:*\n"
            "• कैमरा स्थिर रखें और धुंधलेपन से बचें।\n"
            "• पर्याप्त प्राकृतिक रोशनी में फोटो लें।\n"
            "• लक्षण वाली पत्ती का 15-30 सेमी दूरी से फोटो लें।\n\n"
            "📷 *कृपया नई और स्पष्ट फोटो भेजें।*"
        )
    else:
        return (
            "⚠️ *Image Quality Check Failed*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"The uploaded photo does not meet minimum quality criteria ({trans['en']}).\n\n"
            "💡 *Tips for an accurate diagnosis:*\n"
            "• Hold your camera steady to prevent motion blur.\n"
            "• Ensure adequate natural daylight on the leaf.\n"
            "• Capture a close-up (15-30 cm) of a single leaf showing disease symptoms.\n"
            "• Avoid extreme shadows, glare, or shooting from too far away.\n\n"
            "📷 *Please retake and resend a clear photo.*"
        )


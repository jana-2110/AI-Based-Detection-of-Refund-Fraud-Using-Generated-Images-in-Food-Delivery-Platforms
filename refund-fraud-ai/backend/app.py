from image_authenticity import is_ai_generated
from image_reuse import is_reused
from fraud_score import calculate_score

def refund_decision(image_path, user):
    ai_prob, ai_flag = is_ai_generated(image_path)
    reused = is_reused(image_path)
    score = calculate_score(
        user["refunds"], ai_flag, reused, user["account_age"]
    )

    if ai_flag or reused or score > 70:
        return "Rejected ❌"
    elif score > 30:
        return "Manual Review ⚠️"
    else:
        return "Approved ✅"

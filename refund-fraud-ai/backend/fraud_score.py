def calculate_fraud_score(
    ai_prob,
    reused,
    exif_mismatch=False,
    user_history_score=0.0
):
    score = 0

    score += ai_prob * 40

    if reused:
        score += 30

    if exif_mismatch:
        score += 20

    score += user_history_score * 10

    return min(int(score), 100)

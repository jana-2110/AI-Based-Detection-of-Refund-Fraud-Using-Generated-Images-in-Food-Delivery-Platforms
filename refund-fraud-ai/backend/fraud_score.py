def calculate_score(refunds, ai_flag, reused, account_age):
    score = (refunds*0.4) + (ai_flag*30) + (reused*20) - (account_age*0.1)
    return min(max(score, 0), 100)

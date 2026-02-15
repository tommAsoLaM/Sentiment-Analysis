def accuracy_score(groundTruth, prediction):
    total = 0
    right = 0
    
    for gScore, pScore_raw in zip(groundTruth, prediction):
        total += 1
        
        # SAFEGUARD: If your prediction is a string like '5 stars' from the Hugging Face pipeline,
        # this extracts just the integer 5. If it's already an integer, it just leaves it alone.
        pScore = int(str(pScore_raw)[0])
        
        # -- Exact Matches --
        if gScore == 5 and pScore == 5:
            right += 1
        elif gScore == 4 and pScore == 4:
            right += 1
        elif gScore == 3 and pScore == 3:
            right += 1
        elif gScore == 2 and pScore == 2:
            right += 1
        elif gScore == 1 and pScore == 1:
            right += 1
            
        # -- Relaxed "Positive" Matches --
        # Ground truth is 4, but predicted 5
        elif gScore == 4 and pScore == 5:
            right += 1
        # Ground truth is 5, but predicted 4
        elif gScore == 5 and pScore == 4:
            right += 1
            
        # -- Relaxed "Negative" Matches --
        # Ground truth is 1, but predicted 2
        elif gScore == 1 and pScore == 2:
            right += 1
        # Ground truth is 2, but predicted 1
        elif gScore == 2 and pScore == 1:
            right += 1

    # Return the final accuracy as a decimal percentage
    return right / total if total > 0 else 0.0
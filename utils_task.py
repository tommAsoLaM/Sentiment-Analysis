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

def precision_score(groundTruth, prediction):
    # Helper function to check if prediction is correct (using relaxed matching)
    def is_correct(gScore, pScore):
        # Exact matches
        if gScore == pScore:
            return True
        # Relaxed "Positive" Matches
        if (gScore == 4 and pScore == 5) or (gScore == 5 and pScore == 4):
            return True
        # Relaxed "Negative" Matches
        if (gScore == 1 and pScore == 2) or (gScore == 2 and pScore == 1):
            return True
        return False
    
    # Calculate per-class precision
    precision_per_class = {}
    
    for label in [1, 2, 3, 4, 5]:
        tp = 0  # True Positives: correct predictions for this label
        fp = 0  # False Positives: incorrect predictions for this label
        
        for gScore, pScore_raw in zip(groundTruth, prediction):
            pScore = int(str(pScore_raw)[0])
            
            if pScore == label:
                if is_correct(gScore, pScore):
                    tp += 1
                else:
                    fp += 1
        
        # Precision for this class
        if tp + fp > 0:
            precision_per_class[label] = tp / (tp + fp)
        else:
            precision_per_class[label] = 0.0
    
    # Return average precision across all classes
    return sum(precision_per_class.values()) / len(precision_per_class)

def recall_score(groundTruth, prediction):
    # Helper function to check if prediction is correct (using relaxed matching)
    def is_correct(gScore, pScore):
        # Exact matches
        if gScore == pScore:
            return True
        # Relaxed "Positive" Matches
        if (gScore == 4 and pScore == 5) or (gScore == 5 and pScore == 4):
            return True
        # Relaxed "Negative" Matches
        if (gScore == 1 and pScore == 2) or (gScore == 2 and pScore == 1):
            return True
        return False
    
    # Calculate per-class recall
    recall_per_class = {}
    
    for label in [1, 2, 3, 4, 5]:
        tp = 0  # True Positives: correct predictions for this label
        fn = 0  # False Negatives: missed predictions for this label
        
        for gScore, pScore_raw in zip(groundTruth, prediction):
            pScore = int(str(pScore_raw)[0])
            
            if gScore == label:
                if is_correct(gScore, pScore):
                    tp += 1
                else:
                    fn += 1
        
        # Recall for this class
        if tp + fn > 0:
            recall_per_class[label] = tp / (tp + fn)
        else:
            recall_per_class[label] = 0.0
    
    # Return average recall across all classes
    return sum(recall_per_class.values()) / len(recall_per_class)
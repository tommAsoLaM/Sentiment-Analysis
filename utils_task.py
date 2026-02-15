def accuracy_score(groundTruth, prediction):
    total = 0
    right_strict = 0
    right_relaxed = 0
    
    for gScore, pScore_raw in zip(groundTruth, prediction):
        total += 1
        
        # SAFEGUARD: If your prediction is a string like '5 stars' from the Hugging Face pipeline,
        # this extracts just the integer 5. If it's already an integer, it just leaves it alone.
        pScore = int(str(pScore_raw)[0])
        
        # -- Strict: Exact Matches Only --
        if gScore == pScore:
            right_strict += 1
            right_relaxed += 1  # Exact match also counts as relaxed
        # -- Relaxed Matches (only) --
        elif (gScore == 4 and pScore == 5) or (gScore == 5 and pScore == 4):
            right_relaxed += 1
        elif (gScore == 1 and pScore == 2) or (gScore == 2 and pScore == 1):
            right_relaxed += 1

    # Return tuple: (strict_accuracy, relaxed_accuracy)
    strict = right_strict / total if total > 0 else 0.0
    relaxed = right_relaxed / total if total > 0 else 0.0
    return (strict, relaxed)

def precision_score(groundTruth, prediction):
    # Helper to check relaxed correctness
    def is_correct_relaxed(gScore, pScore):
        if gScore == pScore:
            return True
        if (gScore == 4 and pScore == 5) or (gScore == 5 and pScore == 4):
            return True
        if (gScore == 1 and pScore == 2) or (gScore == 2 and pScore == 1):
            return True
        return False
    
    # Calculate per-class precision
    precision_strict = {}
    precision_relaxed = {}
    
    for label in [1, 2, 3, 4, 5]:
        tp_strict = 0
        tp_relaxed = 0
        fp = 0
        
        for gScore, pScore_raw in zip(groundTruth, prediction):
            pScore = int(str(pScore_raw)[0])
            
            if pScore == label:
                if gScore == label:
                    tp_strict += 1
                    tp_relaxed += 1
                elif is_correct_relaxed(gScore, pScore):
                    tp_relaxed += 1
                else:
                    fp += 1
        
        # Precision for this class
        if tp_strict > 0 or fp > 0:
            precision_strict[label] = tp_strict / (tp_strict + fp)
        else:
            precision_strict[label] = 0.0
            
        if tp_relaxed > 0 or fp > 0:
            precision_relaxed[label] = tp_relaxed / (tp_relaxed + fp)
        else:
            precision_relaxed[label] = 0.0
    
    # Return tuple: (strict_precision, relaxed_precision)
    strict = sum(precision_strict.values()) / len(precision_strict)
    relaxed = sum(precision_relaxed.values()) / len(precision_relaxed)
    return (strict, relaxed)

def recall_score(groundTruth, prediction):
    # Helper to check relaxed correctness
    def is_correct_relaxed(gScore, pScore):
        if gScore == pScore:
            return True
        if (gScore == 4 and pScore == 5) or (gScore == 5 and pScore == 4):
            return True
        if (gScore == 1 and pScore == 2) or (gScore == 2 and pScore == 1):
            return True
        return False
    
    # Calculate per-class recall
    recall_strict = {}
    recall_relaxed = {}
    
    for label in [1, 2, 3, 4, 5]:
        tp_strict = 0
        tp_relaxed = 0
        fn = 0
        
        for gScore, pScore_raw in zip(groundTruth, prediction):
            pScore = int(str(pScore_raw)[0])
            
            if gScore == label:
                if pScore == label:
                    tp_strict += 1
                    tp_relaxed += 1
                elif is_correct_relaxed(gScore, pScore):
                    tp_relaxed += 1
                else:
                    fn += 1
        
        # Recall for this class
        if tp_strict + fn > 0:
            recall_strict[label] = tp_strict / (tp_strict + fn)
        else:
            recall_strict[label] = 0.0
            
        if tp_relaxed + fn > 0:
            recall_relaxed[label] = tp_relaxed / (tp_relaxed + fn)
        else:
            recall_relaxed[label] = 0.0
    
    # Return tuple: (strict_recall, relaxed_recall)
    strict = sum(recall_strict.values()) / len(recall_strict)
    relaxed = sum(recall_relaxed.values()) / len(recall_relaxed)
    return (strict, relaxed)
# -*- coding: utf-8 -*-
import numpy as np
import bisect

# Define cosine similarity values from -1 to 1 with step 0.01, rounded to 2 decimals
C_VALUES = np.round(np.arange(-1, 1.0001, 0.01), decimals=2)

def calculate_pairwise_loss(weight_c, weight_a):
    # Calculate the minimum possible scores under current weights (based on cosine similarity)
    start_scores = [c * weight_c for c in C_VALUES]

    # Sort start_scores to facilitate counting loss later
    sorted_start_scores = sorted(start_scores)
    loss = 0

    for i in range(len(sorted_start_scores)):
        # Find the final score interval for the current cosine similarity score (sorted_start_scores)
        start_score_i = sorted_start_scores[i]
        end_score_i = start_score_i + weight_a

        # Find the insertion position of end_score_i in sorted_start_scores
        insertion_point = bisect.bisect_left(sorted_start_scores, end_score_i)

        # Count how many scores lie between insertion_point and start_score_i, contributing to loss
        count = max(0, insertion_point - i - 1)
        loss += count

    return loss


def find_best_weights(max_weight_c, max_weight_a):
    min_loss = float('inf')  # Initialize min loss to a very large number for comparison
    best_weights = []        # Store best weight combinations

    # Calculate loss for all possible weight combinations
    for weight_c in range(1, max_weight_c + 1):
        for weight_a in range(1, max_weight_a + 1):
            current_loss = calculate_pairwise_loss(weight_c, weight_a)

            if current_loss < min_loss:
                # Update min loss if a smaller loss is found
                min_loss = current_loss
                best_weights = [(weight_c, weight_a)]
            elif current_loss == min_loss:
                # Also save weight combinations with the same minimum loss
                best_weights.append((weight_c, weight_a))

    return min_loss, best_weights

if __name__ == "__main__":
    MAX_WEIGHT_C = 200
    MAX_WEIGHT_A = 200
    
    min_loss, best_weights = find_best_weights(MAX_WEIGHT_C, MAX_WEIGHT_A)
    
    print(f"Minimum Pairwise Ranking Loss: {min_loss}")
    print("Best Weights (Cosine Score Weight, Association Score Weight):")
    for wc, wa in best_weights:
        print(f"Cosine weight = {wc}, Association weight = {wa}")

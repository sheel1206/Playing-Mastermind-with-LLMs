import itertools
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import random


# Define the parameters
n = 4  # Code length
digits = [1, 2, 3, 4]  # Allowed digits
num_simulations = 2000  # Number of games to simulate

# Generate all possible codes
all_codes = list(itertools.product(digits, repeat=n))


def tuple_to_string(code):
    """Convert a tuple to a string by concatenating its digits."""
    return ''.join(map(str, code))

def feedback(hidden_code, guess):
    """
    Calculate feedback (b, w) for a guess against the hidden code.
    b: Correct color and position
    w: Correct color but wrong position
    """
    b = sum(h == g for h, g in zip(hidden_code, guess))
    w = sum(min(hidden_code.count(c), guess.count(c)) for c in set(hidden_code)) - b
    return b, w

def likelihood(hidden_code, guess, feedback_observed):
    """
    Compute the likelihood of feedback_observed given hidden_code.
    Returns 1 if feedback matches; otherwise 0.
    """
    return 1.0 if feedback(hidden_code, guess) == feedback_observed else 0.0

def bayesian_update(priors, guess, feedback_observed):
    """
    Update probabilities using Bayes' rule.
    """
    posteriors = []
    for i, code in enumerate(all_codes):
        P_E_given_C = likelihood(code, guess, feedback_observed)
        posteriors.append(P_E_given_C * priors[i])
    posteriors = np.array(posteriors)
    return posteriors / posteriors.sum()
results = []

codes_to_guess =[(3, 1, 3, 4), (1, 2, 4, 3), (3, 3, 4, 1), (1, 1, 2, 4), (1, 2, 4, 2), (4, 4, 4, 1), (2, 4, 4, 2),
                  (2, 1, 2, 1), (1, 2, 3, 3), (2, 3, 4, 2), (4, 1, 4, 3), (1, 3, 2, 4), (4, 4, 1, 2), (4, 4, 3, 2), 
                  (2, 2, 3, 1), (3, 1, 4, 3), (3, 4, 2, 2), (2, 3, 2, 2), 
                 (3, 3, 3, 2), (1, 4, 4, 4), (3, 1, 2, 4), (3, 2, 4, 1), (2, 4, 2, 2), 
                 (4, 3, 3, 3), (1, 4, 1, 4), (3, 4, 1, 1), (4, 1, 2, 3), (1, 3, 1, 3), (2, 2, 3, 3), (2, 4, 2, 4), (1, 2, 2, 4), 
                 (4, 4, 1, 3), (3, 4, 2, 2), (4, 1, 2, 2), (3, 4, 3, 4),
                   (1, 3, 2, 3), (4, 3, 4, 2), (3, 3, 4, 2), (1, 1, 2, 1), (3, 2, 1, 3), (2, 1, 3, 4), 
                   (1, 3, 3, 3), (3, 2, 2, 1), (2, 3, 1, 2), (4, 2, 4, 1),
                   (4, 1, 3, 1), (4, 4, 2, 4), (2, 1, 4, 2), (3, 1, 4, 1), (4, 3, 2, 3), (3, 1, 3, 1), (4, 2, 3, 4), 
                   (3, 4, 2, 4), (4, 4, 3, 4), (1, 1, 1, 3), (3, 4, 1, 1), (4, 2, 4, 2), (4, 4, 2, 3), 
                   (4, 1, 3, 3), (2, 2, 4, 3), (1, 1, 1, 1), (2, 3, 1, 4), (3, 4, 4, 1), (1, 3, 2, 3), (1, 3, 1, 2),
                     (4, 1, 3, 2), (3, 1, 4, 3), (2, 2, 1, 3), (4, 3, 1, 2), (1, 1, 2, 2), (4, 4, 3, 2), (2, 3, 4, 3),
                     (4, 2, 3, 4), (2, 4, 3, 3), (1, 3, 2, 1), (2, 1, 4, 4), (4, 3, 4, 2), (3, 1, 4, 4), (4, 1, 4, 4), 
                     (4, 1, 2, 4), (3, 2, 4, 3), (4, 2, 4, 3), (3, 1, 4, 1), (4, 1, 4, 3), (3, 4, 4, 4), (2, 2, 4, 1), 
                     (1, 2, 4, 2), (3, 2, 4, 4), (4, 1, 4, 2), (3, 4, 4, 2), (3, 1, 2, 1), (2, 4, 1, 1), (4, 3, 2, 1), (4, 2, 4, 4), (2, 3, 2, 2), (4, 1, 3, 2)]


# Run simulations
for simulation in range(num_simulations):
    # Randomly select a hidden code
    hidden_code = random.choice(all_codes)
    
    # Initialize uniform priors
    priors = np.ones(len(all_codes)) / len(all_codes)
    current_priors = priors.copy()
    guess_number = 1
    print('True code:', hidden_code)
    while True:
        # Number of codes remaining prior to the guess
        codes_remaining_prior = (current_priors > 0).sum()
        # Choose the next guess: Select the code with the highest posterior probability
        max_prob_indices = np.flatnonzero(current_priors == current_priors.max())
        guess_idx = np.random.choice(max_prob_indices)
        guess = all_codes[guess_idx]
        
        fb = feedback(hidden_code, guess)
        
        # Update priors
        current_priors = bayesian_update(current_priors, guess, fb)
        
        # Number of codes remaining after the guess
        codes_remaining_after = (current_priors > 0).sum()
        
        # Save the result of the guess
        results.append({
        "trial_id": simulation,
        "guess_number": guess_number,
        "true_code": tuple_to_string(hidden_code),
        "guess": tuple_to_string(guess),
        "n_green": fb[0],
        "n_yellow": fb[1],
        "codes_remaining_prev": codes_remaining_prior,
        "codes_remaining_post": codes_remaining_after
    })
        guess_number+=1
        # Increment guess number
        if guess == hidden_code or guess_number > 15:
            break

# Save results to a CSV file
df_results = pd.DataFrame(results)
df_results.to_csv("random_baseline.csv", index=False)
print(df_results)

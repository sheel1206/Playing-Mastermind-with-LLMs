import itertools
import numpy as np
import pandas as pd
import time
import together
import re
from sklearn.metrics import mean_absolute_error

# API Key for Together AI
YOUR_API_KEY = ''
together.api_key = YOUR_API_KEY

# Define the parameters
n = 4  # Code length
digits = [1, 2, 3, 4]  # Allowed digits
all_codes = list(itertools.product(digits, repeat=n))

# Model configurations
model_names = [
    "meta-llama/Llama-2-7b-chat-hf",  # LLaMa-2-7B
    "meta-llama/Llama-2-13b-chat-hf", # LLaMa-2-13B
    "meta-llama/Llama-2-70b-hf"       # LLaMa-2-70B
]

prompt_configs = {
    'max_tokens': 4,
    'temperature': 0.9,
    'top_k': 50,
    'top_p': 0.7,
    'repetition_penalty': 1.1,
    'stop': []
}

# Utility functions
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

def call_together_api(prompt, config, post_processing, model='meta-llama/Llama-2-7b-chat-hf'):
    output = together.Complete.create(prompt=prompt, model=model, **config)
    try:
        res = output['output']['choices'][0]['text'].strip()
    except KeyError:
        raise ValueError("Unexpected API response structure.")
    return post_processing(res)

def post_process_output(output_string):
    """Extract digits from the model's response."""
    only_digits = re.sub(r"\D", "", output_string)
    return only_digits[:4]  # Ensure only the first 4 digits are returned

# Initial prompt for the first guess
initial_prompt_template = (
    "Question: Let's play Mastermind. Try to guess the 4-digit code using only the numbers 1, 2, 3, or 4. "
    "You have 15 guesses remaining. Your first guess is?\nAnswer: "
)

# Subsequent prompt template
prompt_template = (
    "Question: We are playing Mastermind. Try to guess the 4-digit code using only the numbers 1, 2, 3, or 4. Use the feedback from your previous guesses to make your next guess. "
    "Green means correct digit and position, yellow means correct digit but wrong position. Your goal is to get (green = 4, yellow = 0). Digits can be repeated in the code."
    "Your previous guesses are: {g_history}. Your previous feedback is:{f_history}. You have {remaining} guesses remaining. "
    "You previous guesses are incorrect. What is your next guess? Output exactly 4 digits and do not repeat a guess. You can choose any combination of one, two, three and four.\nAnswer: "
)

# Simulation
results = []
num_simulations = 1
codes_to_guess =[(3, 1, 3, 4), (1, 2, 4, 3), (3, 3, 4, 1), (1, 1, 2, 4), (1, 2, 4, 2), (4, 4, 4, 1), (2, 4, 4, 2),
                  (2, 1, 2, 1), (1, 2, 3, 3), (2, 3, 4, 2), (4, 1, 4, 3), (1, 3, 2, 4), (4, 4, 1, 2), (4, 4, 3, 2), 
                  (2, 2, 3, 1), (3, 1, 4, 3), (3, 4, 2, 2), (2, 3, 2, 2), 
                 (3, 3, 3, 2), (1, 4, 4, 4), (3, 1, 2, 4), (3, 2, 4, 1), (2, 4, 2, 2), 
                 (4, 3, 3, 3), (1, 4, 1, 4), (3, 4, 1, 1), (4, 1, 2, 3), (1, 3, 1, 3), (2, 2, 3, 3), (2, 4, 2, 4), (1, 2, 2, 4), 
                 (4, 4, 1, 3), (3, 4, 2, 2), (4, 1, 2, 2), (3, 4, 3, 4),
                   (1, 3, 2, 3), (4, 3, 4, 2), (3, 3, 4, 2), (1, 1, 2, 1), (3, 2, 1, 3), (2, 1, 3, 4), 
                   (1, 3, 3, 3), (3, 2, 2, 1), (2, 3, 1, 2), (4, 2, 4, 1)]
for simulation in range(len(codes_to_guess)):
    hidden_code = codes_to_guess[simulation]  # Example hidden code for testing
    priors = np.ones(len(all_codes)) / len(all_codes)
    current_priors = priors.copy()
    guess_number = 1  # Reset numbering to start at 1
    guess_history = ""
    feedback_history = ""
    print("True code:", hidden_code)

    # Maintain a set for tracking previous guesses
    previous_guesses = set()

    while True:
        codes_remaining_prior = (current_priors > 0).sum()
        # Determine the prompt based on guess number
        if guess_number == 1:
            prompt = initial_prompt_template
        else:
            prompt = prompt_template.format(
                g_history=guess_history, 
                f_history=feedback_history, 
                remaining=15 - guess_number + 1
            )
        tries = 0
        # Call the model and validate uniqueness
        while True:
            time.sleep(1)
            str_guess = call_together_api(prompt, prompt_configs, post_process_output, model=model_names[2])
            guess = tuple(int(digit) for digit in str_guess)
            # Check if the guess is a repeat
            
            if guess not in previous_guesses:
                previous_guesses.add(guess)  # Add to the set of previous guesses
                break
            elif tries >6:
                print("Stuck in Loop")
                break
            else:
                print(f"Repeated guess detected: {guess}. Regenerating...")
                tries +=1

        print(f"Guess #{guess_number}: {guess}")

        # Get feedback
        fb = feedback(hidden_code, guess)
        guess_history += f"({str_guess}),"
        feedback_history += f"(green = {fb[0]} yellow = {fb[1]}),"
        # Update priors
        current_priors = bayesian_update(current_priors, guess, fb)
        codes_remaining_after = (current_priors > 0).sum()
        # Save results
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
        guess_number += 1
        # Check stopping conditions
        if guess == hidden_code or guess_number > 15:
            # checkpoint saving in case Together AI credit runs out
            df_results = pd.DataFrame(results)
            df_results.to_csv("LLM_results_initial.csv", index=False)
            break

        

df_results = pd.DataFrame(results)

df_results.to_csv("LLM_results.csv", index=False)
print(df_results)
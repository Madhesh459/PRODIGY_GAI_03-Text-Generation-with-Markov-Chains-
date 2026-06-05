"""
Generative AI Task 3: Text Generation using Markov Chains
---------------------------------------------------------
This program implements an N-gram Markov Chain text generator from scratch
using only Python's standard library. It processes a text corpus, builds a
state-transition probability model, and generates coherent, randomized text
based on user-specified parameters.

Author: Internship Submission
Date: June 2026
"""

import collections
import os
import random
import re


class MarkovChainGenerator:
    """
    A class that represents a Markov Chain text generator.
    Supports customizable order (N-grams) and robust token-based generation.
    """

    def __init__(self, order: int = 2):
        """
        Initialize the generator.

        :param order: The lookback depth (context size) of the Markov Chain.
                      E.g., order=1 uses the current word to predict the next.
                            order=2 uses the last 2 words to predict the next.
        """
        if order < 1:
            raise ValueError("Markov Chain order must be at least 1.")

        self.order = order
        # Maps a state (tuple of length self.order) to a list of possible next tokens
        self.transitions = collections.defaultdict(list)
        # Tracks starting states that begin with a capitalized word (for natural capitalization)
        self.start_states = []

    def _tokenize(self, text: str) -> list:
        """
        Helper method to split text into words and punctuation marks as separate tokens.
        Preserves contractions (e.g., Alistair's) and hyphenated words (e.g., gold-plated).

        :param text: Raw string content.
        :return: List of string tokens.
        """
        # Matches alphanumeric words with internal hyphens/apostrophes, or punctuation marks
        return re.findall(r"[\w'-]+|[.,!?;]", text)

    def fit(self, file_path: str):
        """
        Read the file, preprocess the text, and build the Markov Chain transitions.

        :param file_path: Path to the .txt source dataset.
        """
        # 1. Error Handling: Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: '{file_path}'")

        # 2. Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # 3. Error Handling: Check if file is empty
        if not content:
            raise ValueError(f"The dataset file '{file_path}' is empty.")

        # 4. Tokenization
        tokens = self._tokenize(content)

        # Ensure we have enough tokens to build a chain of the specified order
        if len(tokens) <= self.order:
            raise ValueError(
                f"Dataset is too short ({len(tokens)} tokens) to build "
                f"a Markov Chain of order {self.order}. Please add more text."
            )

        # Clear any existing model data
        self.transitions.clear()
        self.start_states.clear()

        # 5. Populate transition table using a sliding window
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            self.transitions[state].append(next_token)

            # Define suitable starting states: the first token of the state
            # should start with a capital letter and be alphabetical
            first_token = state[0]
            if first_token[0].isupper() and first_token.isalpha():
                self.start_states.append(state)

        # Fallback if no capitalized starting states were found
        if not self.start_states:
            self.start_states = list(self.transitions.keys())

    def generate(self, num_words: int) -> str:
        """
        Generate natural-looking text based on the learned transitions.

        :param num_words: Approximate number of words to generate.
        :return: Formatted output string.
        """
        if not self.transitions:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Start with a random capitalized state
        current_state = random.choice(self.start_states)
        generated_tokens = list(current_state)

        # We generate until we reach approximately the requested count
        # (excluding punctuation from the target word count if possible, or total token count)
        word_count = sum(1 for t in generated_tokens if t.isalnum() or '-' in t)

        while word_count < num_words:
            # If we reach a dead end (a state with no outgoing transitions),
            # pick a new random starting state to keep generating
            if current_state not in self.transitions:
                current_state = random.choice(self.start_states)
                # Add a sentence-ending punctuation if the last token wasn't one
                if generated_tokens and generated_tokens[-1] not in {".", "!", "?"}:
                    generated_tokens.append(".")
                generated_tokens.extend(current_state)
                word_count += sum(1 for t in current_state if t.isalnum() or '-' in t)
                continue

            # Probabilistic selection: choose from the list of observed next tokens
            next_token = random.choice(self.transitions[current_state])
            generated_tokens.append(next_token)

            if next_token.isalnum() or '-' in next_token:
                word_count += 1

            # Shift the state window forward
            current_state = tuple(generated_tokens[-self.order :])

        return self._format_tokens(generated_tokens)

    def _format_tokens(self, tokens: list) -> str:
        """
        Helper method to format the generated tokens into readable paragraph text.
        Avoids putting spaces before punctuation marks and ensures proper capitalization
        at sentence starts.

        :param tokens: List of generated tokens.
        :return: Formatted text paragraph.
        """
        formatted = []
        capitalize_next = True

        for token in tokens:
            if token in {".", ",", "!", "?", ";", ":"}:
                # Append punctuation immediately without prepended space
                formatted.append(token)
                # Next word after a sentence-ender must be capitalized
                if token in {".", "!", "?"}:
                    capitalize_next = True
            else:
                # Capitalize words at sentence starts
                if capitalize_next:
                    token = token.capitalize()
                    capitalize_next = False

                # Add a space between words
                if formatted:
                    formatted.append(" " + token)
                else:
                    formatted.append(token)

        # Ensure the final generated text ends with a period if it doesn't already
        result = "".join(formatted).strip()
        if result and result[-1] not in {".", "!", "?"}:
            result += "."

        return result


def display_banner():
    """Print a professional CLI banner for the application."""
    print("=" * 70)
    print("        * TEXT GENERATION SYSTEM USING MARKOV CHAINS *")
    print("                (Internship Submission - Task 3)       ")
    print("=" * 70)
    print(" This tool learns language patterns from a text file and generates")
    print(" original, contextually plausible sentences using probabilistic state")
    print(" transitions (N-gram Markov Chain).")
    print("-" * 70)


def get_integer_input(prompt: str, min_value: int, default_value: int) -> int:
    """
    Safely get and validate an integer input from the user.

    :param prompt: Input prompt message.
    :param min_value: Minimum allowed value.
    :param default_value: Default value if user presses Enter.
    :return: Validated integer.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default_value
        try:
            val = int(user_input)
            if val >= min_value:
                return val
            print(f"[WARNING] Input must be at least {min_value}.")
        except ValueError:
            print("[WARNING] Invalid input. Please enter a valid integer.")


def main():
    """Main execution function handling interactive CLI and error states."""
    display_banner()

    # Default file path
    default_dataset = os.path.join(os.path.dirname(__file__), "sample.txt")

    # Ask user for dataset path
    file_path = input(f"[INPUT] Enter path to training dataset text file [Default: {default_dataset}]: ").strip()
    if not file_path:
        file_path = default_dataset

    # Ask user for Markov Chain order
    print("\n[INFO] Markov Chain Order refers to the lookback history size.")
    print("   - Order 1: Generates more random and abstract text.")
    print("   - Order 2: Generates moderately coherent, natural text (Recommended).")
    print("   - Order 3: Generates text highly similar or identical to source phrasing.")
    order = get_integer_input("[INPUT] Enter Markov Chain Order (1-5) [Default: 2]: ", min_value=1, default_value=2)

    # Initialize and train the model
    generator = MarkovChainGenerator(order=order)
    print(f"\n[LOAD] Reading and training on '{os.path.basename(file_path)}'...")

    try:
        generator.fit(file_path)
        print("[SUCCESS] Model training completed successfully!")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please verify the file path and ensure the dataset file exists.")
        return
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please check that the file is not empty and contains enough text content.")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error occurred during training: {e}")
        return

    # Enter generation loop
    while True:
        print("\n" + "-" * 50)
        num_words = get_integer_input("[INPUT] Enter number of words to generate [Default: 50]: ", min_value=5, default_value=50)

        print(f"\n[GEN] Generating {num_words} words...")
        try:
            generated_text = generator.generate(num_words)
            
            # Print the output in a styled box
            print("\n" + "+" + "-" * 68 + "+")
            print("| GENERATED TEXT:                                                    |")
            print("+" + "-" * 68 + "+")
            
            # Format lines to fit within 66 characters width
            words = generated_text.split()
            current_line = []
            current_length = 0
            for word in words:
                if current_length + len(word) + 1 > 66:
                    line_str = " ".join(current_line)
                    print(f"| {line_str:<66} |")
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            if current_line:
                line_str = " ".join(current_line)
                print(f"| {line_str:<66} |")
                
            print("+" + "-" * 68 + "+")

            # Save output to outputs/sample_output.txt
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "sample_output.txt")
            
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(generated_text)
            
            print(f"\n[SAVE] Generated text saved to: '{os.path.relpath(output_path)}'")

        except Exception as e:
            print(f"\n[ERROR] Error during generation: {e}")

        # Prompt for regeneration
        run_again = input("\n[INPUT] Generate another text with same model? (y/n) [Default: y]: ").strip().lower()
        if run_again == 'n':
            print("\n[BYE] Thank you for using the Markov Chain Text Generator!")
            break


if __name__ == "__main__":
    main()

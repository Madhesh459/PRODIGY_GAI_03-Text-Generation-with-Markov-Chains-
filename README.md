# Text Generation with Markov Chains 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Madhesh459/PRODIGY_GAI_03-Text-Generation-with-Markov-Chains-/blob/main/notebook.ipynb)

A complete, production-grade, object-oriented implementation of an N-gram Markov Chain Text Generation system in Python. This project is built entirely from scratch using only Python's standard library to demonstrate core concepts of probabilistic modeling, natural language tokenization, and state-machine transitions.

---

## 📌 Project Objective

The primary goal of this project is to build a generative text system based on **Markov Chains**. The application:
1. Reads and tokenizes an input text dataset (`sample.txt`).
2. Preprocesses raw text while retaining sentence-ending punctuation, contractions, and hyphenated structures.
3. Constructs an N-order transition probability matrix mapping lookback sequences to downstream words.
4. Simulates random walks on the transition matrix to generate novel, coherent paragraphs.
5. Implements robust standard library-only file handling, interactive CLI, and comprehensive exception management.

---

## ⚙️ Technologies Used

* **Language**: Python 3.7+
* **Libraries (Standard Library Only)**:
  * `random` - Used for selection of states based on transition weights.
  * `collections` - Implements `defaultdict` for constructing the transition table.
  * `re` - Regular expressions for advanced tokenization and text sanitization.
  * `os` - Used for safe, platform-independent file path resolution and directory construction.

---

## 🧠 Working Principle of Markov Chains

A **Markov Chain** is a stochastic model describing a sequence of events where the probability of transitioning to the next state depends *only* on the current state (called the **Markov Property**). 

### The N-gram Approach (Order of lookback)
In text generation, the "state" is determined by the last $N$ words (where $N$ is the **order** of the Markov Chain).
* **Order 1 (1-Gram lookback)**: Predicts the next word using only the single current word.
* **Order 2 (2-Gram lookback)**: Predicts the next word based on the sequence of the last two words. This provides higher contextual coherence.

#### Mathematical Model
Let $W = (w_1, w_2, \dots, w_k)$ be a sequence of words. In a first-order Markov Chain:
$$P(w_i \mid w_1, w_2, \dots, w_{i-1}) \approx P(w_i \mid w_{i-1})$$

In a second-order Markov Chain (Order 2):
$$P(w_i \mid w_1, w_2, \dots, w_{i-1}) \approx P(w_i \mid w_{i-2}, w_{i-1})$$

### Concrete Example (Order = 1)
Given the training text: `"I think, therefore I am. I think it is cold."`

**1. Tokenization Output:**
`['I', 'think', ',', 'therefore', 'I', 'am', '.', 'I', 'think', 'it', 'is', 'cold', '.']`

**2. Transition Table Construction:**
| Current State (Key) | Possible Next Tokens (Values) | Transition Probability |
| :--- | :--- | :--- |
| `("I",)` | `["think", "am", "think"]` | $P(\text{"think"}\mid\text{"I"}) = 2/3$, $P(\text{"am"}\mid\text{"I"}) = 1/3$ |
| `("think",)` | `[",", "it"]` | $P(\text{","}\mid\text{"think"}) = 1/2$, $P(\text{"it"}\mid\text{"think"}) = 1/2$ |
| `("therefore",)` | `["I"]` | $P(\text{"I"}\mid\text{"therefore"}) = 1.0$ |
| `("am",)` | `["."]` | $P(\text{"."}\mid\text{"am"}) = 1.0$ |

---

## 📂 Directory Structure

The project maintains a clean, modular layout to ensure scalability and professional presentation:

```
Text-Generation-Markov-Chain/
│
├── main.py                # Main script housing MarkovChainGenerator and interactive CLI
├── sample.txt             # Default rich corpus for model training
├── requirements.txt       # Environment and dependency documentation
├── README.md              # Comprehensive documentation and project guide
│
├── outputs/
│   └── sample_output.txt  # Automatically created; contains text from the latest run
│
└── screenshots/
    └── placeholder.txt    # Folder reserved for console and output screenshots
```

---

## 🔧 Installation & Setup

No third-party packages are required. To install and run:

1. **Clone the Repository / Extract files**:
   Ensure all files are placed in a folder named `Text-Generation-Markov-Chain/`.

2. **Check Python Version**:
   Verify you have Python 3.7+ installed:
   ```bash
   python --version
   ```

3. **Verify Dependencies**:
   As documented in `requirements.txt`, only standard modules are utilized:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This command will run successfully and take 0 seconds, as no external packages are listed).*

---

## 🚀 Usage Instructions

To launch the interactive generator:

```bash
python main.py
```

### Interactive CLI Walkthrough

1. **Training Dataset Path**: The program will prompt for a `.txt` file path. Press `Enter` to use the built-in `sample.txt`, or type an absolute/relative path to your own custom text.
2. **Markov Chain Order**: Select the chain depth (1 to 5). 
   * Use **`2`** (default) for natural-sounding sentences.
   * Use **`1`** for abstract, dream-like text.
3. **Word Count**: Choose how many words you want the system to generate.
4. **Text Display**: The output will print inside an ASCII-bordered layout in your console.
5. **Auto-Save**: The generated text is automatically saved to `outputs/sample_output.txt`.
6. **Repeat**: Prompt will ask if you'd like to run again. Type `n` to exit or `y` to regenerate.

---

## 📄 Input Dataset (sample.txt)

To understand the generated output, here is the source dataset (`sample.txt`) used for training:

```text
In the heart of the forgotten city, there stood a grand clockwork library. Its towering walls were lined with thousands of ancient brass shelves, each holding leather-bound journals, silver scrolls, and heavy stone tablets. The air smelled of aged parchment, sweet vanilla, and machine oil. Every hour, a massive gold-plated pendulum swung across the central atrium, ticking with a low, mechanical pulse that echoed through the quiet corridors.

Visitors to the clockwork library were rare, for the city was hidden deep within the whispering mountains. Only those who followed the star charts and decoded the ancient copper gears could find the entrance. Inside, copper automations, shaped like mechanical owls, glided silently between the shelves. They polished the metal gears and cataloged the incoming books, their crystal eyes glowing with a soft amber light.

At the center of the library lay the Great Astrolabe, a massive mechanism that mapped the movement of stars, moons, and forgotten dimensions. The astrolabe was connected to a series of pneumatic tubes, through which brass canisters were sent to deliver messages across the library. If a reader requested a scroll, the astrolabe would spin, the gears would click, and the pneumatic tubes would hiss as the correct cylinder arrived at the desk.

The head librarian, a wise man named Alistair, spent his days studying the celestial maps. He believed that the stars were not merely burning gas, but giant clockwork gears in a cosmic machine. "Every tick of the universe," Alistair would say, "writes a new page in our books." He watched the owls carry scrolls back and forth, wondering if the machine would ever run down, or if the pendulum would keep swinging forever.

As night fell over the whispering mountains, the library grew darker. The amber light of the mechanical owls cast long, dancing shadows across the brass floor. The hum of the Great Astrolabe quieted to a gentle purr, and the scent of sweet vanilla grew stronger in the cool evening air. In this quiet sanctuary, the history of the world continued to write itself, page by page, gear by gear, waiting for the next traveler to unlock its secrets.
```

---

## 📋 Sample Output

The following is an actual execution trace trained on our clockwork library dataset (`sample.txt`) at **Order 2** for **50 words**:

```text
======================================================================
        * TEXT GENERATION SYSTEM USING MARKOV CHAINS *       
======================================================================
 This tool learns language patterns from a text file and generates
 original, contextually plausible sentences using probabilistic state
 transitions (N-gram Markov Chain).
----------------------------------------------------------------------
[INPUT] Enter path to training dataset text file [Default: ...\sample.txt]: 

[INFO] Markov Chain Order refers to the lookback history size.
   - Order 1: Generates more random and abstract text.
   - Order 2: Generates moderately coherent, natural text (Recommended).
   - Order 3: Generates text highly similar or identical to source phrasing.
[INPUT] Enter Markov Chain Order (1-5) [Default: 2]: 2

[LOAD] Reading and training on 'sample.txt'...
[SUCCESS] Model training completed successfully!

--------------------------------------------------
[INPUT] Enter number of words to generate [Default: 50]: 50

[GEN] Generating 50 words...

+--------------------------------------------------------------------+
| GENERATED TEXT:                                                    |
+--------------------------------------------------------------------+
| As night fell over the whispering mountains. Only those who        |
| followed the star charts and decoded the ancient copper gears      |
| could find the entrance. Inside, copper automations, shaped like   |
| mechanical owls, glided silently between the shelves. They         |
| polished the metal gears and cataloged the incoming books, their   |
| crystal eyes glowing.                                              |
+--------------------------------------------------------------------+

[SAVE] Generated text saved to: 'outputs\sample_output.txt'

[INPUT] Generate another text with same model? (y/n) [Default: y]: n

[BYE] Thank you for using the Markov Chain Text Generator!
```

---

## 🛠️ Error Handling Configurations

To ensure the system never crashes during use, robust error boundary assertions are programmed into the entry points:

1. **Missing File Safeguard**: If the specified dataset file path does not exist, the program catches `FileNotFoundError` and logs a clean explanation instead of a stack trace.
2. **Empty Dataset Handling**: If the corpus contains only white spaces or is completely empty, a `ValueError` is raised and caught, warning the user.
3. **Insufficient Text Size**: If a user attempts to train an Order 3 model on a file containing only 2 words, the model alerts them that the corpus is too short.
4. **Invalid Interactive Inputs**: The inputs for *Order* and *Word Count* are wrapped inside validation loops. If a user types letters, negative values, or empty junk, the program prints a warning and prompts them again.
5. **Dead-End Resolution**: If the generator reaches a final phrase state that has no future transitions (e.g. the last words in the file), it dynamically switches back to a random start state, appends punctuation, and proceeds seamlessly.

---

## 🔮 Future Improvements

To expand the capabilities of this generator, the following iterations are planned:
* **Weighted Temperature Sampling**: Implement a temperature setting to scale probabilities. Higher temperatures add chaos/creativity, while lower temperatures follow the source text strictly.
* **Laplace / Kneser-Ney Smoothing**: Incorporate smoothing algorithms to allow small transition probabilities for word sequences not directly present in the source text.
* **Web UI GUI Port**: Transition this system to a local web page using Streamlit or a fast Flask backend to allow slider-based control and real-time visualization of the transition graphs.

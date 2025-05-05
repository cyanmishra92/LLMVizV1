# LLM Inference Simulator

This interactive web application, built with Streamlit, estimates the compute (FLOPs) and memory usage (Parameters, KV Cache, Activations) for the prefill and decode stages of Large Language Model (LLM) inference.

## Features

*   **Interactive Parameter Input:** Adjust model architecture (layers, hidden size, heads, MLP dim, vocab size) and runtime parameters (batch size, prompt length, tokens to generate) directly in the interface.
*   **Configuration File Upload:** Upload a JSON file specifying the model architecture (see `config_example.json` for format).
*   **Prefill & Decode Analysis:** View detailed FLOPs and memory estimates for both inference stages.
*   **Calculation Breakdowns:** Expand sections to see the formulas used for calculating parameters, memory, and FLOPs.
*   **Visualizations:** Compare prefill vs. decode FLOPs, see peak memory breakdown, and observe KV cache growth and cumulative decode FLOPs.
*   **Model Structure Overview:** A textual description of the assumed decoder-only transformer architecture.

## Setup and Running Locally

1.  **Prerequisites:**
    *   Python 3.8+ installed.
    *   `pip` (Python package installer).

2.  **Clone or Download:**
    *   Obtain the application files (`app.py`, `llm_calc.py`, `requirements.txt`, `config_example.json`).

3.  **Navigate to Directory:**
    *   Open your terminal or command prompt and change to the directory containing the downloaded files.
    ```bash
    cd path/to/llm_simulator
    ```

4.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

7.  **Access the App:**
    *   Streamlit will provide a local URL (usually `http://localhost:8501`) in your terminal. Open this URL in your web browser.

## Files

*   `app.py`: The main Streamlit application code.
*   `llm_calc.py`: Contains the `LLMCalculator` class with the core computation logic.
*   `requirements.txt`: Lists the required Python packages.
*   `config_example.json`: An example JSON file for the model configuration upload feature.
*   `README.md`: This file.


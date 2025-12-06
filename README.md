# STAT 663 - Final project
## Integrating Chicago Crime Data with Zillow Housing Data

This repository contains the code and documentation for the final project of STAT 663. The project focuses on integrating Chicago crime data with Zillow housing data to analyze the relationship between crime rates and housing prices in Chicago.

To smoothly run the code in this repository, please follow the steps below:

1. **Clone the Repository**: Start by cloning this repository to your local machine using the following command:
   ```
   git clone
    ```
2. **Install Streamlit nightly**: This project requires the nightly version of Streamlit. You can install it using pip:
   ```
   pip uninstall streamlit
   pip install streamlit-nightly --upgrade
   ```

3. **Set Up API Key**: The project uses groq api for the chatbot functionality. You need to obtain an API key from [groq](https://console.groq.com/keys) and store it in a file named `key.txt` inside the `dashboard_v2` directory. Make sure to keep this file secure and do not share it publicly.

4. **Run the Streamlit App**: Navigate to the `dashboard_v2` directory and run the Streamlit app using the following command:
   ```
   conda activate your_env_name_with_streamlit_nightly
   cd dashboard_v2
   streamlit run Home.py
   ```
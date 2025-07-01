import sys
import json
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine, text
from groq import Groq
import re
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Define output files
LOG_FILE = "static/output.txt"
QUERY_OUTPUT_FILE = "static/query_output.html"
PLOT_OUTPUT_FILE = "static/plot_output.html"
ML_SUGGESTION_FILE = "static/ml_suggestion.txt"

# Ensure static directory exists before creating files
os.makedirs("static", exist_ok=True)

# Initialize output files
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")  # Clear log file
with open(QUERY_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")  # Clear query output file
with open(PLOT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")  # Clear plot output file
with open(ML_SUGGESTION_FILE, "w", encoding="utf-8") as f:
    f.write("")  # Clear ML suggestion file

# Custom print function that writes to both console and log file
def log_print(*args, **kwargs):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)
    print(*args, **kwargs)  # Also print to console

# Redirect stdout and stderr to log file
sys.stdout = open(LOG_FILE, "a", encoding="utf-8")
sys.stderr = sys.stdout

# Groq API client
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env

# Get the API key securely from environment
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    log_print("❌ GROQ_API_KEY not found in environment variables.")
    sys.exit(1)

client = Groq(api_key=api_key)

# Ensure model save directory
os.makedirs("models", exist_ok=True)

# Load or create model registry
registry_file = "models/registry.json"
model_registry = {}
if os.path.exists(registry_file):
    try:
        with open(registry_file, "r") as f:
            model_registry = json.load(f)
    except json.JSONDecodeError:
        log_print("Warning: Model registry file is corrupted. Creating a new one.")
        model_registry = {}

# ML Workflow
def train_and_save_model(df, target, features, model_type=None):
    """
    Trains and saves a machine learning model based on the provided DataFrame,
    target, and features.
    """
    log_print(f"\n✅ Training ML model | Target: {target} | Features: {features}")

    # Validate target and features exist in DataFrame
    if target not in df.columns:
        log_print(f"❌ Invalid target column: '{target}'. Skipping model training.")
        return
    if not all(f in df.columns for f in features):
        missing_features = [f for f in features if f not in df.columns]
        log_print(f"❌ Missing feature columns: {missing_features}. Skipping model training.")
        return

    # Drop rows with NaN values in target or features
    df_cleaned = df.dropna(subset=[target] + features).copy()

    if df_cleaned.empty:
        log_print("❌ No valid data after dropping NaNs. Skipping model training.")
        return

    # Determine model type if not explicitly provided
    if not model_type:
        if pd.api.types.is_numeric_dtype(df_cleaned[target]) and df_cleaned[target].nunique() > 10:
            model_type = "regression"
        else:
            model_type = "classification"

    # Handle categorical features using one-hot encoding
    X = pd.get_dummies(df_cleaned[features])
    y = df_cleaned[target]

    # Check if there's enough data for splitting
    if len(X) < 2:
        log_print("❌ Not enough data points to split for training. Skipping model training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None
    if model_type == "regression":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "classification":
        # Ensure target is suitable for classification
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 50: # Arbitrary threshold for too many classes
             log_print(f"❌ Target column '{target}' has too many unique numeric values for classification. Skipping.")
             return
        model = RandomForestClassifier(random_state=42)
    else:
        log_print(f"❌ Unknown model type: {model_type}. Skipping model training.")
        return

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            log_print(f"\n✅ Model trained. Accuracy: {acc:.2%}")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            log_print(f"\n✅ Model trained. RMSE: {rmse:.2f}")

        model_name = f"{target}_model.pkl"
        model_path = os.path.join("models", model_name)
        joblib.dump(model, model_path)
        log_print(f"💾 Model saved to {model_path}")

        # Update model registry
        model_registry[target] = {
            "model_file": model_path,
            "features": features,
            "type": model_type,
            "trained_on": datetime.now().isoformat()
        }
        with open(registry_file, "w") as f:
            json.dump(model_registry, f, indent=2)
        log_print("📚 Model registry updated.")

    except Exception as e:
        log_print(f"❌ Error during model training or saving: {e}")

# Chart rendering function
# [Previous imports and setup remain the same until chart rendering function]

def render_chart(code, df, chart_idx):
    """
    Executes Python code to render a chart and saves it to a file.
    Supports Matplotlib/Seaborn (PNG) and Plotly (HTML).
    """
    local_env = {"df": df, "pd": pd, "sns": sns, "plt": plt, "px": px}
    try:
        start = time.time()
        
        # Remove fig.show() calls to prevent console output
        code = code.replace("fig.show()", "")
        
        exec(code, local_env)
        end = time.time()
        duration = round(end - start, 2)
        log_print(f"⏱️ Chart #{chart_idx} rendered in {duration} seconds")

        # Save Matplotlib/Seaborn charts
        if "plt.show()" in code:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"static/chart_{chart_idx}_{timestamp}.png"
            plt.savefig(filename)
            log_print(f"🖼️ Chart #{chart_idx} saved to {filename}")
            plt.clf()
            return filename

        # Save Plotly charts
        fig = local_env.get("fig")
        if fig is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"static/chart_{chart_idx}_{timestamp}.html"
            # Use full_html=False to minimize the HTML output
            fig.write_html(filename, include_plotlyjs='cdn', full_html=False)
            log_print(f"🌐 Chart #{chart_idx} saved to {filename}")
            return filename
        else:
            log_print(f"⚠️ No figure found in chart code #{chart_idx}.")
            return None
            
    except Exception as e:
        log_print(f"❌ Error rendering chart #{chart_idx}: {e}")
        return None
# [Rest of the file remains the same]
# Main execution flow
if __name__ == "__main__":
    if len(sys.argv) < 6:
        log_print("ℹ️ Using environment variables for DB config (CLI args not provided).")
        host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        question = input("❓ Enter your natural language question: ").strip()

        if not all([host, db_name, user, password, question]):
            log_print("❌ Missing DB environment variables or question input.")
            sys.exit(1)
    else:
        host = sys.argv[1]
        db_name = sys.argv[2]
        user = sys.argv[3]
        password = sys.argv[4]
        question = sys.argv[5]


        conn = None # Initialize connection to None
    try:
        # Database connection
        # Using mysql+mysqlconnector for MySQL databases
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}")
        conn = engine.connect()
        log_print("\n✅ Connected to the database successfully.\n")

        # Get schema
        log_print("\n📚 Available Tables and Schemas:")
        schema = ""
        try:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
        except Exception as e:
            log_print(f"❌ Error fetching tables: {e}")
            tables = []

        if not tables:
            log_print("⚠️ No tables found in the database or error fetching tables.")
            sys.exit(1)

        for t in tables:
            log_print(f"\n📌 Table: {t}")
            try:
                desc = conn.execute(text(f"DESCRIBE `{t}`")).fetchall()
                for col in desc:
                    log_print(f"  - {col[0]} ({col[1]})")
                schema += f"\nTable: {t}\n"
                for col in desc:
                    schema += f"- {col[0]} ({col[1]})\n"
            except Exception as e:
                log_print(f"❌ Error describing table `{t}`: {e}")
                schema += f"\nTable: {t}\n- Error fetching schema\n"

        # Generate SQL query using Groq
        log_print("\n--- Generating SQL Query ---")
        sql_prompt = f"""
You are a SQL assistant.
Given this database schema (MySQL):
{schema}
Generate a MySQL query that answers:
"{question}"
Only return the SQL code. Do not include any explanation or markdown outside the code block.
"""
        sql_response = client.chat.completions.create(
            model="llama3-70b-8192", # Using the specified model
            messages=[{"role": "user", "content": sql_prompt}]
        )
        sql_query = sql_response.choices[0].message.content.strip()
        # Clean up the response to get only the SQL query
        if "```" in sql_query:
            sql_query = sql_query.split("```")[1].replace("sql", "").strip()

        log_print("\n🧠 Generated SQL Query:\n")
        log_print(sql_query)

        # Execute query
        df = pd.DataFrame() # Initialize empty DataFrame
        try:
            df = pd.read_sql(sql_query, con=engine)
            log_print("\n📊 Query Result (first 5 rows):")
            if df.empty:
                log_print("⚠️ No data returned by the query.")
            else:
                log_print(df.head().to_string(index=False))
                # Save query results to HTML file
                df.to_html(QUERY_OUTPUT_FILE, index=False)
                log_print(f"\n💾 Query results saved to {QUERY_OUTPUT_FILE}")
        except Exception as e:
            log_print(f"❌ Error executing SQL query: {e}")

        # Generate visualizations if DataFrame is not empty
        if not df.empty:
            log_print("\n--- Generating Visualizations ---")
            viz_prompt = f"""
You are a data visualization expert.
Given this DataFrame (first 5 rows for schema reference):
{df.head().to_string(index=False)}

Suggest up to 3 different chart ideas using seaborn or plotly.
Each chart should have its own complete Python code block.
Assume 'df' is already defined.

IMPORTANT:
1. DO NOT include fig.show() or plt.show() in the code
2. For Plotly charts, assign the figure to a variable named 'fig'
3. For Matplotlib/Seaborn charts, use plt but don't call show()

Only return Python code blocks. Do not include any explanation or markdown outside the code blocks.
"""
            viz_response = client.chat.completions.create(
                model="llama3-70b-8192", # Using the specified model
                messages=[{"role": "user", "content": viz_prompt}]
            )

            viz_raw = viz_response.choices[0].message.content.strip()
            code_blocks = re.findall(r"```(?:python)?(.*?)```", viz_raw, re.DOTALL)

            if not code_blocks:
                log_print("⚠️ No chart code blocks found in the visualization suggestion.")
            else:
                log_print(f"\n📈 {len(code_blocks)} Suggested Chart(s):\n")
                for i, viz_code in enumerate(code_blocks, 1):
                    # Basic fixes for common LLM chart generation issues
                    viz_code = viz_code.strip()
                    if "plt.show()" not in viz_code and "fig.show()" not in viz_code:
                        if "plt." in viz_code:
                            viz_code += "\nplt.show()"
                        elif "px." in viz_code:
                            viz_code += "\nfig.show()"

                    log_print(f"\n🔢 Chart #{i} Code:\n{viz_code}")
                    log_print(f"\n🔍 Rendering Chart #{i}...")
                    render_chart(viz_code, df, i)
        else:
            log_print("\n⚠️ Skipping visualization generation as the query returned no data.")

        # Optional ML Training if DataFrame is not empty and has enough rows
        if not df.empty and len(df) > 5: # Only suggest ML if we have enough data
            log_print("\n--- Generating ML Model Suggestions ---")
            ml_prompt = f"""
You are a machine learning expert.
Given this sample DataFrame (first 5 rows for schema reference):
{df.head().to_string(index=False)}

Suggest up to 3 possible ML tasks (classification or regression).
Each suggestion should include:
- Target column (must be present in the DataFrame)
- Features to use (must be present in the DataFrame, exclude target)
- Type of ML (classification or regression)
- One sentence reason

Respond only with valid JSON. Do not include any explanation or markdown outside the JSON.

Format:
[
  {{
    "target": "target_column",
    "features": ["feat1", "feat2"],
    "type": "classification",
    "reason": "Short reason"
  }}
]
"""
            try:
                ml_response = client.chat.completions.create(
                    model="llama3-70b-8192", # Using the specified model
                    messages=[{"role": "user", "content": ml_prompt}]
                )
                ml_ideas_raw = ml_response.choices[0].message.content.strip()
                # Attempt to parse JSON, handle cases where LLM might add ```json
                if ml_ideas_raw.startswith("```json"):
                    ml_ideas_raw = ml_ideas_raw[7:]
                if ml_ideas_raw.endswith("```"):
                    ml_ideas_raw = ml_ideas_raw[:-3]

                ml_ideas = json.loads(ml_ideas_raw)

                if ml_ideas:
                    log_print("\n🔮 Suggested Machine Learning Ideas:\n")
                    ml_suggestion_content = ""
                    for idx, idea in enumerate(ml_ideas, 1):
                        suggestion = (f"{idx}. 🎯 Target: {idea.get('target', 'N/A')}\n"
                                    f"   🛠️ Features: {idea.get('features', 'N/A')}\n"
                                    f"   🧠 Type: {idea.get('type', 'N/A')}\n"
                                    f"   📋 Reason: {idea.get('reason', 'N/A')}\n")
                        log_print(suggestion)
                        ml_suggestion_content += suggestion + "\n"

                    # Save ML suggestions to file
                    with open(ML_SUGGESTION_FILE, "w", encoding="utf-8") as f:
                        f.write(ml_suggestion_content)
                    log_print(f"\n💾 ML suggestions saved to {ML_SUGGESTION_FILE}")

                    # Automatically train the first suggestion if valid
                    if ml_ideas:
                        selected = ml_ideas[0]
                        # Validate if target and features exist in the DataFrame before training
                        if selected.get('target') in df.columns and \
                           all(f in df.columns for f in selected.get('features', [])):
                            log_print(f"\n🤖 Automatically training model for: {selected.get('target')}")
                            train_and_save_model(df, selected["target"], selected["features"], selected["type"])
                        else:
                            log_print("⚠️ First ML suggestion skipped due to invalid target or features in DataFrame.")
                else:
                    log_print("⚠️ No ML model suggestions found.")
            except json.JSONDecodeError:
                log_print(f"❌ ML suggestion error: Could not parse JSON response from LLM. Raw: {ml_ideas_raw}")
            except Exception as e:
                log_print(f"⚠️ An error occurred during ML suggestion: {e}")
        else:
            log_print("\n⚠️ Skipping ML model suggestion as the query returned no data or insufficient rows (<100).")

    except Exception as e:
        log_print(f"\n❌ An unhandled error occurred: {e}")
    finally:
        if conn:
            conn.close()
            log_print("\n🔒 Database connection closed.")

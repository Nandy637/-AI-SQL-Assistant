from flask import Flask, render_template, request, jsonify
import subprocess
import os
import threading
import time

app = Flask(__name__)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("models", exist_ok=True) # Ensure models directory also exists

# Global variable to track the processing status and thread
# Using a dictionary to hold the thread object and a flag
processing_status = {"thread": None, "is_processing": False}

def run_script_in_background(host, db, user, password, question):
    """
    Function to run the sql-query-viz-ml-generator.py script.
    This will be executed in a separate thread.
    """
    global processing_status
    try:
        # Clear previous output before starting the new process
        with open("static/output.txt", "w", encoding="utf-8") as f:
            f.write("Starting processing...\n")

        # Execute the script
        # The script itself redirects its stdout/stderr to static/output.txt
        # So we don't need to capture stdout/stderr here.
        subprocess.run(
            [
                "python",
                "sql-query-viz-ml-generator.py",
                host,
                db,
                user,
                password,
                question
            ],
            check=True, # Raise CalledProcessError for non-zero exit codes
            stdout=subprocess.DEVNULL, # Suppress stdout to avoid double output
            stderr=subprocess.DEVNULL  # Suppress stderr
        )
        with open("static/output.txt", "a", encoding="utf-8") as f:
            f.write("\n--- Processing Complete ---✅\n")
    except subprocess.CalledProcessError as e:
        with open("static/output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Script Error ---❌\n")
            f.write(f"Script exited with error: {e}\n")
    except Exception as e:
        with open("static/output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- Unexpected Error ---❌\n")
            f.write(f"An unexpected error occurred: {str(e)}\n")
    finally:
        processing_status["is_processing"] = False
        processing_status["thread"] = None # Clear the thread reference

@app.route("/", methods=["GET", "POST"])
def index():
    print("==> / route hit: method =", request.method)
    error = None

    if request.method == "GET":
        print("==> GET request received, rendering form only.")
        return render_template("index.html", processing=processing_status["is_processing"])

    # === POST ===
    print("==> POST request received.")

    if processing_status["is_processing"]:
        error = "A request is already being processed. Please wait."
        print("==> Processing already in progress.")
        return render_template("index.html", error=error, processing=True)

    try:
        host = request.form["host"]
        db = request.form["db"]
        user = request.form["user"]
        password = request.form["password"]
        question = request.form["question"]

        print("==> Received form data:", host, db, user, "[password hidden]", question)

        processing_status["is_processing"] = True

        script_thread = threading.Thread(
            target=run_script_in_background,
            args=(host, db, user, password, question)
        )
        script_thread.daemon = True
        script_thread.start()
        processing_status["thread"] = script_thread

        return render_template("index.html", processing=True)

    except Exception as e:
        error = f"Error initiating request: {str(e)}"
        print("==> Error in POST:", error)
        processing_status["is_processing"] = False
        processing_status["thread"] = None
        return render_template("index.html", error=error, processing=False)
@app.route("/get_output")
def get_output():
    
    """
    Endpoint to fetch the current output and processing status.
    """
    output_content = ""
    try:
        with open("static/output.txt", "r", encoding="utf-8") as f:
            output_content = f.read()
    except FileNotFoundError:
        output_content = "No output file found yet."
    except Exception as e:
        output_content = f"Error reading output file: {str(e)}"

    # Check if the background thread is still alive
    is_currently_processing = processing_status["is_processing"] and \
                              processing_status["thread"] and \
                              processing_status["thread"].is_alive()

    # If the thread is no longer alive but the flag is still true (e.g., script finished quickly),
    # ensure the flag is updated. This handles cases where the thread finishes between checks.
    if processing_status["is_processing"] and \
       processing_status["thread"] and \
       not processing_status["thread"].is_alive():
        processing_status["is_processing"] = False
        processing_status["thread"] = None


    return jsonify(output=output_content, processing=is_currently_processing)

if __name__ == "__main__":
    # Ensure static directory is created on app startup if it doesn't exist
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)



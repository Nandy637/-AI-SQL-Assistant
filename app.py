from flask import Flask, render_template, request, jsonify
import subprocess
import os
import threading
import time
import glob

app = Flask(__name__)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variable to track the processing status and thread
processing_status = {"thread": None, "is_processing": False}

def run_script_in_background(host, db, user, password, question):
    """
    Function to run the sql-query-viz-ml-generator.py script.
    This will be executed in a separate thread.
    """
    global processing_status
    try:
        # Clear previous output and charts before starting
        with open("static/output.txt", "w", encoding="utf-8") as f:
            f.write("Starting processing...\n")
        
        # Clean old chart files
        for f in glob.glob("static/chart_*.html"):
            os.remove(f)
        for f in glob.glob("static/chart_*.png"):
            os.remove(f)

        # Execute the script
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
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
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
        processing_status["thread"] = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", processing=processing_status["is_processing"])

    if processing_status["is_processing"]:
        return render_template("index.html", error="A request is already being processed. Please wait.", processing=True)

    try:
        host = request.form["host"]
        db = request.form["db"]
        user = request.form["user"]
        password = request.form["password"]
        question = request.form["question"]

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
        processing_status["is_processing"] = False
        processing_status["thread"] = None
        return render_template("index.html", error=f"Error initiating request: {str(e)}", processing=False)

@app.route("/get_output")
def get_output():
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

    if processing_status["is_processing"] and \
       processing_status["thread"] and \
       not processing_status["thread"].is_alive():
        processing_status["is_processing"] = False
        processing_status["thread"] = None

    return jsonify(output=output_content, processing=is_currently_processing)

@app.route('/get_charts')
def get_charts():
    chart_files = []
    # Get HTML charts (Plotly)
    chart_files.extend(glob.glob("static/chart_*.html"))
    # Get PNG charts (Matplotlib/Seaborn)
    chart_files.extend(glob.glob("static/chart_*.png"))
    
    # Sort by creation time (newest first)
    chart_files.sort(key=os.path.getmtime, reverse=True)
    
    # Return just the filenames
    return jsonify(charts=[os.path.basename(f) for f in chart_files])
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, use_reloader=False)  # ← Prevent double-execution

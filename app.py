from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    # Extract arguments from the form or use default values
    root_dir = request.form.get('root_dir', '/app')
    gtFine = request.form.get('gtFine', 'test_images/gtFine')
    images = request.form.get('images', 'test_images/images')
    output_dir = request.form.get('output_dir', '/app/output')
    model_path = request.form.get('model_path', '/app/model.pth')

    # Command to run the model script directly
    command = [
        'python', 'src/main.py',
        '--root_dir', root_dir,
        '--gtFine', gtFine,
        '--images', images,
        '--output_dir', output_dir,
        '--model_path', model_path
    ]

    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Extract Mean IoU from stdout
    mean_iou = None
    for line in stdout.decode().splitlines():
        if "Mean IoU" in line:
            mean_iou = line.split(":")[-1].strip()

    if mean_iou is None:
        mean_iou = "Error: Could not calculate Mean IoU. Check model script output."

    return render_template('index.html', mean_iou=mean_iou)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

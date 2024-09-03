from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_model', methods=['POST'])
def run_model():
    # Command to run the model script inside the Docker container
    command = [
        'docker', 'run', '--rm',
        '-v', '/home/gawad/Cityscapes_project/test_images:/app/test_images',
        '-v', '/home/gawad/Cityscapes_project/output_csvs:/app/output_csvs',
        '-v', '/home/gawad/Cityscapes_project/model_best_epoch.pth:/app/model.pth',
        'gawad1/cityscapes:latest',
        '--root_dir', '/app',
        '--gtFine', 'test_images/gtFine',
        '--images', 'test_images/images',
        '--test_csv', 'output_csvs/test_data.csv',
        '--model_path', '/app/model.pth'
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

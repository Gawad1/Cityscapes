# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Copy only the requirements file first
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ /app/src/
COPY app.py /app/
COPY templates/ /app/templates/

# Expose the Flask port
EXPOSE 5000

# Run the application
ENTRYPOINT ["python", "app.py"]


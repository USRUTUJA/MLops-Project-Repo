# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files to the container
COPY train_model.py /app/

# Install required Python packages
RUN pip install boto3 pandas tensorflow scikit-learn

# Set the default command to execute the training script
CMD ["python", "train_model.py"]

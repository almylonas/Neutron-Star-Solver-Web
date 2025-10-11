# Use a stable Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /

# Copy dependencies first
COPY requirements.txt .

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Run your app
CMD ["python", "main.py"]

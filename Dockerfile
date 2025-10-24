# Use an official lightweight Python image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command
CMD ["bash"]

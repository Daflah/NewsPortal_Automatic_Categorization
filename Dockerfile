# Baca ini terlebih dahulu sebelum memulai deployment
FROM python:3.9-slim

# Set user untuk security
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements first untuk better caching
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY --chown=user . .

# Expose port 7860 (required oleh Hugging Face Spaces)
EXPOSE 7860

# Run aplikasi dengan gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "web_app.app:app"]

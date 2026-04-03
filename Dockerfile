# 1. Python ka image lein
FROM python:3.11

# 2. Aik folder banayein jahan code rahega
WORKDIR /code

# 3. Requirements file copy karke libraries install karein
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Apni poori app ka code copy karein
COPY . .

# 5. FastAPI ko Hugging Face ke default port 7860 par chalayein
# Yahan 'app.main:app' ka matlab hai: app folder ke andar main.py file mein 'app' variable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

# Python ka image use karein
FROM python:3.10

# Working directory set karein (Direct app folder mein)
WORKDIR /code/app

# Requirements install karein
# Pehle file copy karein (Check karein requirements.txt kahan hai)
COPY requirements.txt /code/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt

# Saara code copy karein (App folder ke andar)
COPY . /code/app/

# FastAPI ko port 7860 par chalayein
# Ab hum app folder ke andar hain, isliye seedha "main:app" likhein
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

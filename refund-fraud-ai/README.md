AI-Based Detection of Refund Fraud Using Generated Images

Detects AI-generated images used for fake refund claims in food delivery apps.

Tech:
- Python
- TensorFlow
- OpenCV
- CNN


step:1 Open Project Folder

cd "J:\AI-Based Detection of Refund Fraud Using Generated Images in Food Delivery Platforms\refund-fraud-ai"

step:2 Activate Virtual Environment

python -m venv venv

venv\Scripts\activate

step:3 Install Dependencies (One Time)
pip install -r requirements.txt
pip install python-multipart

step:4 Train the Model (Optional)
Run only if dataset is updated:

python model\train.py

step:5 Start Backend API

uvicorn backend.api:app --reload
Backend runs at:
http://127.0.0.1:8000

step:6 Test API Using Swagger UI

http://127.0.0.1:8000/docs

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. تحميل الموديل المدرب
model = load_model('my_project/industrial_lstm_rul_model.h5')

# 2. تحميل البيانات
# هنفترض إننا بنستخدم بيانات الـ test اللي جهزناها
test_data = pd.read_csv('my_project/test_processed.csv')

# 3. إعداد البيانات للـ LSTM (نفس الـ Windowing اللي عملناه في الكولاب)
def create_test_sequences(data, sequence_length=50):
    X = []
    # هنا بتفترضي إن الداتا فيها أعمدة الحساسات جاهزة
    features = data.drop(columns=['unit_number', 'time_in_cycles'], errors='ignore').values
    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
    return np.array(X)

SEQUENCE_LENGTH = 50
X_test = create_test_sequences(test_data, SEQUENCE_LENGTH)

# 4. التنبؤ
predictions = model.predict(X_test)

# 5. عرض النتائج
print("--- Prediction Complete ---")
print("First 10 predictions:")
print(predictions[:10].flatten())

# اختياري: حفظ النتائج في ملف
pd.DataFrame(predictions, columns=['Predicted_RUL']).to_csv('predictions.csv', index=False)
print("Results saved to predictions.csv")
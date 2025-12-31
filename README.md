# TriBuxi - Patient Adherence Prediction System

Hệ thống dự đoán mức độ tuân thủ điều trị của bệnh nhân sử dụng Random Forest Classifier.

## 📋 Mô tả

Dự án này sử dụng Machine Learning để dự đoán xem bệnh nhân có tuân thủ điều trị hay không dựa trên các đặc điểm:
- Tuổi (AGE)
- Tổng số tiền yêu cầu bồi thường hàng năm (ANNUALCLAIMAMOUNT)
- Tổng số đơn vị (UNITSTOTAL)

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- pip

### Cài đặt thư viện

```bash
pip install streamlit pandas scikit-learn joblib matplotlib pillow
```

### Chạy ứng dụng
```bash
streamlit run app.py
```

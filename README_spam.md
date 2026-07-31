# تشخیص پیامک اسپم با Naive Bayes

مدلی برای دسته‌بندی پیامک‌ها به دو کلاس `spam` و `ham` (عادی)، مبتنی بر پردازش متن کلاسیک و الگوریتم Naive Bayes.

## دیتاست

دیتاست [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) شامل پیامک‌های برچسب‌خورده (`label`, `message`) با جداکننده‌ی tab.

## پیش‌پردازش متن

تابع `text_process` روی هر پیامک این مراحل رو انجام می‌ده:
1. تبدیل متن به حروف کوچک
2. حذف علائم نگارشی (punctuation)
3. حذف stopwords زبان انگلیسی (با NLTK)

## پایپ‌لاین مدل

متن پیامک‌ها با یک پایپ‌لاین سه‌مرحله‌ای پردازش و طبقه‌بندی می‌شود:

```
CountVectorizer (Bag of Words با text_process سفارشی)
      ↓
TfidfTransformer (وزن‌دهی TF-IDF)
      ↓
MultinomialNB (طبقه‌بند Naive Bayes)
```

این سه مرحله با `sklearn.pipeline.Pipeline` به‌صورت یکپارچه اجرا می‌شوند.

## پیش‌بینی تکی (Singleton Prediction)

تابع `singleton_prediction` یک پیامک دلخواه می‌گیرد و برچسب (`spam`/`ham`) به‌همراه احتمال هر کلاس را چاپ می‌کند. برای نمونه، پیامک‌های تبلیغاتی/کلاه‌برداری (مثل برنده‌شدن جایزه یا هشدار جعلی حساب بانکی) به‌درستی spam تشخیص داده می‌شوند.

## نحوه اجرا

```bash
pip install pandas nltk scikit-learn
python -c "import nltk; nltk.download('stopwords')"
python spam_detector.py
```

## ساختار پروژه

```
sms-spam-detection/
├── README.md
├── spam_detector.py
└── SMSSpamCollection
```

## نکات توسعه‌ی بعدی

- ذخیره‌ی مدل نهایی با `joblib` یا `pickle` برای استفاده در production (در کد فعلی کامنت شده و آماده‌ی فعال‌سازی است)
- ارزیابی روی داده‌ی جدا از train با `train_test_split` و گزارش دقیق‌تر با `classification_report`

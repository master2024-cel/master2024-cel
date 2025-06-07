import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st

# تحميل البيانات
df = pd.read_csv("jugements_maroc.csv")

# معالجة نصوص الوقائع + نوع القضية
df["input_text"] = df["type_cause"] + " " + df["faits"]

# تحويل الحكم إلى أرقام
le = LabelEncoder()
y = le.fit_transform(df["jugement"])

# بناء النموذج
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)
pipeline.fit(df["input_text"], y)

# واجهة المستخدم
st.title("🔮 تطبيق توقع منطوق الأحكام القضائية المغربية (بسيط)")
st.write("أدخل نوع القضية والوقائع ثم اضغط على 'توقع الحكم'")

# اختيار نوع القضية
type_cause = st.selectbox("📂 نوع القضية", ["كراء", "طلاق", "عقار", "جنائي"])

# إدخال الوقائع
faits_input = st.text_area("✍️ وقائع القضية")

# زر التوقع
if st.button("🔮 توقع الحكم"):
    if faits_input.strip() == "":
        st.warning("⚠️ من فضلك أدخل وقائع القضية أولاً.")
    else:
        input_text = type_cause + " " + faits_input
        prediction = pipeline.predict([input_text])
        predicted_label = le.inverse_transform(prediction)[0]
        st.success(f"🖨️ الحكم المتوقع: **{predicted_label}**")

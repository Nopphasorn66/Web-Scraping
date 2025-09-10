import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ===============================
# ฟังก์ชันโหลดและเตรียมข้อมูลภาพยนตร์ (จำลอง)
# ===============================
@st.cache_data
def load_and_prepare_data():
    """
    สร้างและเตรียมข้อมูลภาพยนตร์แบบจำลองสำหรับการจัดกลุ่ม
    """
    # ข้อมูลภาพยนตร์แบบจำลอง (ชื่อ, คะแนน Tomatometer, คะแนน Popcornmeter)
    data = {
        'title': [
            'Dune: Part Two', 'Godzilla x Kong', 'Kingdom of the Planet of the Apes',
            'Inside Out 2', 'Furiosa', 'The Fall Guy', 'Civil War',
            'Ghostbusters: Frozen Empire', 'Kung Fu Panda 4', 'A Quiet Place: Day One',
            'IF', 'Madame Web', 'The Garfield Movie', 'Abigail', 'Challengers',
            'Imaginary', 'The Watchers', 'The Lord of the Rings', 'Top Gun: Maverick',
            'Barbie', 'Oppenheimer', 'The Batman', 'Pulp Fiction', 'The Shawshank Redemption',
            'Morbius', 'The Room'
        ],
        'tom_score': [93, 89, 88, 91, 90, 81, 82, 43, 71, 89, 49, 12, 38, 86, 89, 29, 31, 94, 96, 88, 93, 85, 92, 91, 15, 26],
        'pop_score': [95, 87, 85, 94, 82, 85, 68, 87, 86, 75, 50, 57, 49, 90, 82, 31, 38, 97, 99, 83, 91, 87, 96, 98, 71, 51]
    }
    df = pd.DataFrame(data)
    
    # แปลงคะแนนเป็นค่าทศนิยมสำหรับใช้ในโมเดล K-Means
    df['tom_score_clean'] = df['tom_score'].astype(float) / 100
    df['pop_score_clean'] = df['pop_score'].astype(float) / 100
    
    return df

# โหลดข้อมูลเข้ามาใช้งาน
df = load_and_prepare_data()

# ===============================
# ส่วน UI ของ Streamlit
# ===============================
st.title("ระบบจัดกลุ่มภาพยนตร์ตามคะแนน")
st.markdown("### ค้นพบหมวดหมู่ภาพยนตร์ที่น่าสนใจจากคะแนน Tomatometer และ Popcornmeter")

# ส่วนรับค่าจากผู้ใช้ (จำนวนกลุ่ม)
with st.sidebar:
    st.subheader("ตั้งค่าการจัดกลุ่ม")
    n_clusters = st.slider(
        'จำนวนกลุ่มที่ต้องการ',
        min_value=2,
        max_value=5,
        value=4,
        step=1
    )
    st.markdown("---")
    st.info("💡 **คำแนะนำ:** ลองปรับจำนวนกลุ่มเพื่อดูว่าการจัดประเภทภาพยนตร์แตกต่างกันอย่างไร")


# ===============================
# ส่วนของการจัดกลุ่ม K-Means และแสดงผล
# ===============================
if not df.empty:
    # เลือกคุณสมบัติที่ใช้ในการจัดกลุ่ม
    features = df[['tom_score_clean', 'pop_score_clean']]

    # สร้างและรันโมเดล K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    st.subheader("ผลการจัดกลุ่มภาพยนตร์")

    # พลอตผลลัพธ์การจัดกลุ่ม
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['tom_score'], df['pop_score'], c=df['cluster'], cmap='viridis', s=100)
    
    # เพิ่มชื่อภาพยนตร์ในกราฟ
    for i, row in df.iterrows():
        ax.annotate(row['title'], (row['tom_score'] + 1, row['pop_score'] + 1), fontsize=8)

    # พลอต centroid ของแต่ละกลุ่ม
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0] * 100, centroids[:, 1] * 100, marker='X', s=200, color='red', label='Cluster Centroids')
    
    ax.set_title('การจัดกลุ่มภาพยนตร์ด้วย K-Means')
    ax.set_xlabel('คะแนน Tomatometer (%)')
    ax.set_ylabel('คะแนน Popcornmeter (%)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # แสดงรายละเอียดของแต่ละกลุ่ม
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i].copy()
        
        # คำนวณค่าเฉลี่ยของคะแนนในแต่ละกลุ่ม
        avg_tom = cluster_df['tom_score'].mean()
        avg_pop = cluster_df['pop_score'].mean()
        
        # ตั้งชื่อกลุ่มตามค่าเฉลี่ยคะแนน
        if avg_tom >= 70 and avg_pop >= 70:
            cluster_name = f"กลุ่มที่ {i+1}: **ความสำเร็จจากนักวิจารณ์และผู้ชม**"
        elif avg_tom >= 70 and avg_pop < 70:
            cluster_name = f"กลุ่มที่ {i+1}: **ภาพยนตร์ขวัญใจนักวิจารณ์**"
        elif avg_tom < 70 and avg_pop >= 70:
            cluster_name = f"กลุ่มที่ {i+1}: **ขวัญใจผู้ชม, นักวิจารณ์ไม่ปลื้ม**"
        else:
            cluster_name = f"กลุ่มที่ {i+1}: **ความล้มเหลวโดยรวม**"

        with st.expander(f"{cluster_name} ({len(cluster_df)} เรื่อง)"):
            st.dataframe(cluster_df[['title', 'tom_score', 'pop_score']].reset_index(drop=True))

# ```
# eof

# ### การใช้งานโค้ด

# 1.  **บันทึกไฟล์:** บันทึกโค้ดด้านบนเป็นไฟล์ชื่อ `movie_clustering_app.py`
# 2.  **ติดตั้งไลบรารี:** ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไลบรารีที่จำเป็นแล้ว โดยรันคำสั่งใน Terminal:
#     ```sh
#     pip install streamlit pandas scikit-learn matplotlib
#     ```
# 3.  **รันแอป:** เปิด Terminal ในโฟลเดอร์ที่คุณบันทึกไฟล์ไว้ แล้วรันคำสั่ง:
#     ```sh
#     streamlit run movie_clustering_app.py
    

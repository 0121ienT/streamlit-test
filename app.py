import streamlit as st
import pandas as pd
import pickle
import json
import warnings

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Tải mô hình đã huấn luyện (vẫn từ .pkl)
try:
    # <<< THAY ĐỔI: Đảm bảo tên tệp này khớp với tệp bạn đã tạo (ví dụ: stress_model.pkl)
    with open('stress_trained.sav', 'rb') as f_model:
        model = pickle.load(f_model)
except FileNotFoundError:
    # <<< THAY ĐỔI: Cập nhật thông báo lỗi cho nhất quán
    st.error("Lỗi: Không tìm thấy tệp mô hình 'stress_model.pkl'.")
    st.error("Hãy đảm bảo bạn đã chạy tệp 'create_model.py' hoặc 'train_model.py' để tạo ra tệp này.")
    st.stop()

# --- TẢI THÔNG TIN TỪ FILE .JSON ---
try:
    with open('data_info.json', 'r', encoding='utf-8') as f_info:
        data_info = json.load(f_info) # Sử dụng json.load
    
    feature_names = data_info['feature_names']
    stats = data_info['stats']
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy tệp thông tin 'data_info.json'.")
    st.error("Hãy đảm bảo bạn đã tạo tệp này thủ công hoặc chạy tệp 'train_model.py'.")
    st.stop()
except json.JSONDecodeError:
    st.error("Lỗi: Tệp 'data_info.json' không phải là tệp JSON hợp lệ.")
    st.stop()

# Định nghĩa nhãn cho các mức độ stress
STRESS_MAP = {
    "0": "Thấp (Low)",
    "1": "Trung bình (Medium)",
    "2": "Cao (High)"
}

# <<< THÊM MỚI: Dictionary chứa các lời giải thích
STRESS_EXPLANATIONS = {
    "0": "Tình trạng tinh thần của bạn rất tốt. Mức độ stress thấp cho thấy bạn đang kiểm soát tốt các yếu tố áp lực trong cuộc sống. Hãy tiếp tục duy trì thói quen sinh hoạt và suy nghĩ tích cực!",
    "1": "Bạn đang có dấu hiệu stress ở mức độ vừa phải. Đây có thể là phản ứng bình thường trước các áp lực, nhưng bạn nên chú ý. Hãy dành thời gian thư giãn, xem xét lại khối lượng công việc/học tập và chia sẻ với bạn bè.",
    "2": "Mức độ stress của bạn đang ở mức cao. Đây là một cảnh báo quan trọng. Stress cao kéo dài có thể ảnh hưởng nghiêm trọng đến sức khỏe thể chất và tinh thần. Bạn nên giảm tải công việc ngay lập tức, tìm kiếm sự giúp đỡ từ chuyên gia hoặc người thân."
}

# --- Giao diện ứng dụng Streamlit ---
st.set_page_config(page_title="Dự đoán Mức độ Stress", layout="wide")
st.title("Ứng dụng Dự đoán Mức độ Stress 🩺")
st.write("Sử dụng các thanh trượt bên dưới để nhập vào các chỉ số của bạn và nhấn 'Dự đoán' để xem kết quả.")

# Tạo 2 cột để giao diện đỡ dài
col1, col2 = st.columns(2)

# Dictionary để lưu trữ input của người dùng
input_data = {}

# (Giữ nguyên phần tạo sliders)
mid_point = len(feature_names) // 2
features_col1 = feature_names[:mid_point]
features_col2 = feature_names[mid_point:]

with col1:
    st.header("Các chỉ số (Phần 1)")
    for feature in features_col1:
        min_val = int(stats[feature]['min'])
        max_val = int(stats[feature]['max'])
        mean_val = int(stats[feature]['mean'])
        
        input_data[feature] = st.slider(
            label=feature.replace("_", " ").capitalize(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=1
        )

with col2:
    st.header("Các chỉ số (Phần 2)")
    for feature in features_col2:
        min_val = int(stats[feature]['min'])
        max_val = int(stats[feature]['max'])
        mean_val = int(stats[feature]['mean'])
        
        input_data[feature] = st.slider(
            label=feature.replace("_", " ").capitalize(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=1
        )

# Nút dự đoán
st.divider()
if st.button("Dự đoán Mức độ Stress", type="primary", use_container_width=True):
    # Chuyển dữ liệu input thành DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names] 

    # Chạy mô hình dự đoán
    try:
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[0]
        
        # <<< --- KHỐI LOGIC MỚI THAY THẾ --- >>>
        
        # Lấy lớp được dự đoán (ví dụ: 0, 1, hoặc 2)
        predicted_class = prediction[0]
        
        # Lấy nhãn tương ứng (ví dụ: "Trung bình (Medium)")
        result_label = STRESS_MAP.get(str(predicted_class), "Không xác định")
        
        # Lấy xác suất cao nhất (xác suất của lớp được dự đoán)
        max_probability = probabilities[predicted_class]
        
        # Lấy lời giải thích tương ứng
        explanation = STRESS_EXPLANATIONS.get(str(predicted_class), "Không có lời giải thích cho kết quả này.")
        
        # --- Hiển thị kết quả ---
        
        # 1. Hiển thị kết quả chính
        st.subheader(f"Kết quả Dự đoán: Mức độ Stress là '{result_label}'")
        
        # 2. Hiển thị độ chắc chắn (xác suất)
        percent_value = f"{max_probability * 100:.2f}%"
        st.metric(label="Độ chắc chắn của mô hình", value=percent_value)
        
        # 3. Hiển thị lời giải thích
        st.subheader("Lời giải thích & Khuyến nghị:")
        st.info(explanation)
        
        # <<< --- KẾT THÚC KHỐI LOGIC MỚI --- >>>

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
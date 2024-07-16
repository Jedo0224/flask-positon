from flask import Flask, request, render_template, redirect, url_for , jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os


app = Flask(__name__)

# 파일 업로드 설정
UPLOAD_FOLDER = 'C:\\Users\\qkrwo\\uploads\\beacon_data\\'
MODEL_PATH='C:\\Users\\qkrwo\\Downloads\\Indoor-Positioning-main\\models\\rf_model.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# 파일 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 기본 라우트
@app.route('/')
def home():
    return "Flask Server For indoor-positioning"

@app.route('/train', methods=['POST'])
def train_model():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
    if os.path.exists(file_path):
        # 기존 모델 파일 삭제
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        
        result = train_and_evaluate_model(file_path)
        return jsonify(result)
    else:
        return jsonify({'status': 'error', 'message': 'CSV file not found'}), 404

# 모델 학습 및 평가 함수
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle as pkl

def train_and_evaluate_model(file_path):
    data = pd.read_csv(file_path)
    data = data.fillna(0)  # 결측값을 0으로 대체
    target = data['Room']
    features = data.drop(columns=['Room'])
    
    # 데이터 표준화
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # PCA를 사용한 차원 축소
    pca = PCA(n_components=0.95)  # 설명 분산이 95%인 주성분 선택
    features = pca.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # 모델 선택 및 하이퍼파라미터 튜닝
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 교차 검증
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    
    best_model.fit(X_train, y_train)
    
    # 모델 저장
    with open('C:\\Users\\qkrwo\\Downloads\\Indoor-Positioning-main\\models\\rf_model.pkl', 'wb') as file:
        pkl.dump(best_model, file)
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {'accuracy': accuracy, 'report': report, 'cv_scores': cv_scores}

# 결과 출력
result = train_and_evaluate_model(UPLOAD_FOLDER+"output.csv")
print("Accuracy:", result['accuracy'])
print("Classification Report:", result['report'])
print("Cross-validation Scores:", result['cv_scores'])


# # 예측 엔드포인트
# @app.route('/predict', methods=['POST'])
# def predict():
#     Sensors = request.json
#     prediction = predict_room(Sensors)
#     return jsonify({'prediction': prediction})

# 예측 함수

@app.route('/predict', methods=['POST'])
def predict_room():
    Sensors = request.json
    with open('C:\\Users\\qkrwo\\Downloads\\Indoor-Positioning-main\\models\\rf_model.pkl', 'rb') as file:
        rfc = pkl.load(file)
    app.logger.info("Received data: %s", Sensors)
     # 데이터프레임으로 변환 및 정렬
    input_df = pd.DataFrame([Sensors])
    input_df = input_df.fillna(0)  # NaN 값을 0으로 대체
    input_df = input_df.reindex(columns=rfc.feature_names_in_, fill_value=0)  # 특성 이름 순서 맞추기
    
    prediction = rfc.predict(input_df)[0]
    # 예측
    return jsonify({'status': 'success', 'prediction': str(prediction)})


    
@app.route('/predict', methods=['POST'])
def predict():
    Sensors = request.json
    app.logger.info("Received data: %s", Sensors)
    # 데이터를 로그에 출력
    print("Received data: ", Sensors)
    # 추후 예측 기능 추가 가능
    return jsonify({'status': 'success', 'data': Sensors})

if __name__ == '__main__':
    app.run(debug=True)



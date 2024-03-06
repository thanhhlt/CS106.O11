# Import các thư viện cần thiết
import argparse
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import dump, load
import time
from Lo_r import LogisticRegression

# Hàm chuẩn bị dữ liệu
def data_preprocessing(df):
    df['sex'] = df['sex'].apply(lambda s: (s=='Male' and 1) or (s=='Female' and 2))

    # df['race'] = df['race'].apply(lambda r: (r=='African-American' and 1) or (r=='Asian' and 2) 
    #                             or (r=='Caucasian' and 3) or (r=='Hispanic' and 4) 
    #                             or (r=='Native American' and 5) or (r=='Other' and 6))

    # df['c_charge_degree'] = df['c_charge_degree'].apply(lambda c: (c=='F' and 1) or (c=='M' and 2))
    
    df.drop(['race', 'c_charge_degree'], axis=1, inplace=True)
    
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']
    
    std = StandardScaler()
    for col in X.columns:
        X[col] = std.fit_transform(X[col].to_numpy().reshape(-1, 1))
    
    return df, X, y

# Hàm giúp lựa chọn siêu tham số
def hyperparameter_tuning(X_train, y_train, X_dev, y_dev, param_grid):
    result = pd.DataFrame(columns=['param', 'f1_score', 'time'])
    for l in param_grid['learning_rate']:
        for m in param_grid['max_iter']:
            for f in param_grid['fit_intercept']:
                start = time.time()
                model = LogisticRegression(learning_rate=l, max_iter=m, fit_intercept=f)
                model.train_logistic_regression(X_train, y_train)
                end = time.time()
                y_pred = model.predict(X_dev)
                result.loc[len(result.index)] = [[l, m, f], f1_score(y_dev, y_pred), end-start]
                
    print(result)
    best_param = result.iloc[result['f1_score'].idxmax(), 0]        
    return best_param

# Hàm để chạy quá trình huấn luyện
def run_train(train_dir, dev_dir, model_dir):
    # Tạo thư mục cho mô hình nếu nó chưa tồn tại
    os.makedirs(model_dir, exist_ok=True)

    # Đường dẫn đến các tệp dữ liệu
    train_file = os.path.join(train_dir, 'train.json')
    dev_file = os.path.join(dev_dir, 'dev.json')

    # Đọc dữ liệu huấn luyện và phát triển
    train_data = pd.read_json(train_file, lines=True)
    dev_data = pd.read_json(dev_file, lines=True)

    # Chuẩn bị dữ liệu cho quá trình huấn luyện
    train_data, X_train, y_train = data_preprocessing(train_data)
    dev_data, X_dev, y_dev = data_preprocessing(dev_data)

    # Tạo và huấn luyện mô hình
    param_grid = {'learning_rate' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], 'max_iter' : [200], 'fit_intercept' : [False]}
    param = hyperparameter_tuning(X_train, y_train, X_dev, y_dev, param_grid)
    
    model = LogisticRegression(param[0], param[1], param[2])
    model.train_logistic_regression(X_train, y_train)

    # Lưu mô hình
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    dump(model, model_path)


# Hàm để chạy quá trình dự đoán
def run_predict(model_dir, input_dir, output_path):
    # Đường dẫn đến mô hình và dữ liệu đầu vào
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    input_file = os.path.join(input_dir, 'test.json')

    # Tải mô hình
    model = load(model_path)

    # Đọc dữ liệu kiểm tra
    test_data = pd.read_json(input_file, lines=True)

    # Chuẩn bị dữ liệu kiểm tra
    X_test = test_data
    X_test['sex'] = X_test['sex'].apply(lambda s: (s=='Male' and 1) or (s=='Female' and 2))
    X_test.drop(['race', 'c_charge_degree'], axis=1, inplace=True)
    
    std = StandardScaler()
    for col in X_test.columns:
        X_test[col].fillna(X_test[col].mean(), inplace=True)
        X_test[col] = std.fit_transform(X_test[col].to_numpy().reshape(-1, 1))

    # Thực hiện dự đoán
    predictions = model.predict(X_test)

    # Lưu kết quả dự đoán
    pd.DataFrame(predictions, columns=['two_year_recid']).to_json(output_path, orient='records', lines=True)


# Hàm chính để xử lý lệnh từ dòng lệnh
def main():
    # Tạo một parser cho các lệnh từ dòng lệnh
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Tạo parser cho lệnh 'train'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train_dir', type=str)
    parser_train.add_argument('--dev_dir', type=str)
    parser_train.add_argument('--model_dir', type=str)

    # Tạo parser cho lệnh 'predict'
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_dir', type=str)
    parser_predict.add_argument('--input_dir', type=str)
    parser_predict.add_argument('--output_path', type=str)

    # Xử lý các đối số nhập vào
    args = parser.parse_args()

    # Chọn hành động dựa trên lệnh
    if args.command == 'train':
        run_train(args.train_dir, args.dev_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()
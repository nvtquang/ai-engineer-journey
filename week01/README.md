# Tuần 1 — Nền tảng ML: NumPy, EDA, hồi quy tuyến tính, gradient descent, logistic regression

## Notebook

| Ngày | Notebook | Nội dung |
|---|---|---|
| 1 | [day1_numpy.ipynb](day1_numpy.ipynb) | Khái niệm ML và các bài tập NumPy cơ bản: tạo mảng, indexing, reshape, broadcasting. |
| 2 | [day2_eda.ipynb](day2_eda.ipynb) | EDA Titanic bằng Pandas: giá trị thiếu, tỉ lệ sống theo nhóm, tạo Title / FamilySize / IsAlone / HasCabin. |
| 3 | [day3_linreg.ipynb](day3_linreg.ipynb) | Linear regression bằng phương trình chuẩn, tự viết metric và StandardScaler, polynomial features, phân tích phần dư (California Housing). |
| 4 | [day4_gd.ipynb](day4_gd.ipynb) | Gradient descent: gradient check, batch / SGD / mini-batch, ảnh hưởng của learning rate, momentum, vì sao phải chuẩn hóa. |
| 5 | [day5_logreg.ipynb](day5_logreg.ipynb) | Logistic regression và softmax regression tự viết: sigmoid/BCE ổn định số, ngưỡng và ma trận nhầm lẫn, chữ số viết tay. |
| 6 | [day6_titanic.ipynb](day6_titanic.ipynb) | Titanic end-to-end: preprocessor không rò rỉ, baseline, LogisticRegressionGD so với sklearn, phân tích lỗi, ablation, nộp Kaggle. |

## Kết quả Titanic

Validation: chia train.csv 80/20, stratify theo Survived, `random_state=42` (val có 179 hành khách, 1 người = 0,56%).

| Mô hình | Val accuracy | Kaggle public |
|---|---|---|
| Luôn dự đoán 0 (chết) | 0,6145 | — |
| Quy tắc giới tính (nữ = 1, nam = 0) | 0,7765 | ≈ 0,766 (baseline của Kaggle) |
| LogisticRegressionGD tự viết (η = 0,1, batch 32, 200 epoch) | **0,8436** | ___ |
| sklearn `LogisticRegression` (L2, C = 1) | 0,8436 | — |

GD tự viết và sklearn trùng 98,9% dự đoán trên val, và hệ số tương quan 0,997.

## Ablation

Bỏ từng nhóm đặc trưng, fit lại preprocessor trên tr, train lại với cùng cấu hình:

| Bỏ nhóm | Số đặc trưng | Val accuracy | Chênh lệch |
|---|---|---|---|
| (đầy đủ) | 20 | 0,8436 | — |
| Title | 15 | 0,8101 | −0,0335 (−6 người) |
| FamilySize + IsAlone | 18 | 0,8436 | 0 |
| HasCabin | 19 | 0,8492 | +0,0056 (+1 người) |
| Fare | 19 | 0,8436 | 0 |
| Embarked | 17 | 0,8380 | −0,0056 (−1 người) |

Title quan trọng nhất (Master = bé trai, một tương tác giới tính × tuổi mà mô hình tuyến tính không tự học được). Các chênh lệch ±1 người nằm trong mức nhiễu của val.

## 3 điều học được

1. **Chống rò rỉ là chuyện tổ chức code:** tách rõ `fit` (học thống kê, chỉ chạy trên train) và `transform` (chỉ dùng lại thống kê). Dấu hiệu kiểm tra nhanh: sau chuẩn hóa, val *không* có mean/std đúng bằng 0/1.
2. **Luôn có baseline trước khi train:** quy tắc giới tính một dòng code đã đạt 0,78. Mô hình chỉ hơn khoảng 12 người, và hầu hết khác biệt nằm ở vài nhóm (bé trai Master, cô gái hạng 3 gia đình đông).
3. **Với đặc trưng cộng tuyến, dự đoán ổn định nhưng hệ số thì không:** sklearn không regularize cho cùng dự đoán với GD nhưng hệ số rất khác (tương quan 0,40), vì one-hot đủ cột + bias có vô số nghiệm. Early stopping và L2 đều chọn nghiệm "nhỏ".

## Điều vẫn chưa hiểu rõ

- Val chỉ có 179 mẫu, chênh 1–2 người là nhiễu. Vậy làm sao chọn learning rate và đặc trưng một cách đáng tin? Cross-validation có giải quyết được không, và nếu dùng nó để chọn cấu hình nhiều lần thì có bị "overfit theo val" không?

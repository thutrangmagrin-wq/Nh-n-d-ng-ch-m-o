HỆ THỐNG NHẬN DẠNG CON VẬT BẰNG MẠNG NEURAL NHÂN TẠO
(Animal AI Recognition Studio)

Thông tin dự án
Ngôn ngữ lập trình: Python 3.12
Thư viện sử dụng: NumPy, Pillow
Loại bài toán: Phân loại nhị phân (Binary Classification)
Đối tượng phân loại: Mèo (Cat) & Chó (Dog)
Đường dẫn: C:\Users\Admin\Animal_AI

MỤC LỤC
Tổng quan dự án
Cấu trúc thư mục dự án
Lý thuyết nền tảng — Mạng Neural Nhân Tạo
Chi tiết từng thuật toán AI
4.1 Tiền xử lý ảnh (Image Preprocessing)
4.2 Hàm kích hoạt Sigmoid
4.3 Forward Propagation — Lan truyền xuôi
4.4 Binary Cross-Entropy Loss — Hàm mất mát
4.5 Backpropagation — Lan truyền ngược
4.6 Stochastic Gradient Descent — Bộ tối ưu hóa
4.7 Online Learning — Học trực tuyến theo phản hồi người dùng
Phân tích từng file mã nguồn
Luồng xử lý tổng thể
Giao diện Web và API Server
Phân tích ưu/nhược điểm và hướng phát triển
Kết luận
1. TỔNG QUAN DỰ ÁN
Dự án Animal AI Recognition Studio là một hệ thống nhận dạng hình ảnh con vật được xây dựng hoàn toàn từ đầu (from scratch) — không sử dụng bất kỳ thư viện học sâu nào như TensorFlow, PyTorch hay Keras. Toàn bộ thuật toán AI — từ mạng Neural, thuật toán lan truyền ngược đến tối ưu hóa trọng số — đều được lập trình thủ công bằng Python thuần và NumPy.

Đặc điểm nổi bật:
Tính năng	Mô tả
Kiến trúc	Mạng Neural 2 tầng (1 ẩn + 1 đầu ra)
Phân loại	Nhị phân: Mèo (1) ↔ Chó (0)
Học tập	Batch Gradient Descent toàn cục + Online Learning từng mẫu
Phản hồi người dùng	Human-in-the-loop: học từ phản hồi đúng/sai
Lưu trữ mô hình	Trọng số được lưu vào model.npz (NumPy binary)
Giao diện	Web Studio (HTML5/JS), hỗ trợ camera, điện thoại
Camera	WebRTC (webcam PC) + HTML5 Capture (điện thoại)
2. CẤU TRÚC THƯ MỤC DỰ ÁN

Animal_AI/
├── activation.py        ← Hàm kích hoạt Sigmoid và đạo hàm
├── loss.py              ← Hàm mất mát Binary Cross-Entropy
├── optimizer.py         ← Bộ tối ưu hóa SGD (Gradient Descent)
├── neuron.py            ← Định nghĩa một Neuron đơn lẻ
├── neural_network.py    ← Lắp ghép mạng Neural hoàn chỉnh
├── image_processor.py   ← Đọc và tiền xử lý ảnh
├── train.py             ← Script huấn luyện CLI (dòng lệnh)
├── predict.py           ← Script nhận dạng CLI (dòng lệnh)
├── app_gui.py           ← Máy chủ Web + Giao diện đồ họa
├── model.npz            ← File lưu trọng số đã huấn luyện
└── dataset/
    ├── cat/             ← Ảnh mèo (nhãn = 1)
    └── dog/             ← Ảnh chó  (nhãn = 0)
Mô hình phân tầng rõ ràng: Mỗi file đảm nhận đúng một trách nhiệm duy nhất theo nguyên tắc thiết kế Single Responsibility Principle (SRP) — điều này giúp dễ đọc, dễ bảo trì và dễ mở rộng.

3. LÝ THUYẾT NỀN TẢNG — MẠNG NEURAL NHÂN TẠO
3.1 Neuron nhân tạo là gì?
Lấy cảm hứng từ não người, một neuron nhân tạo nhận vào các giá trị đầu vào 
x
1
,
x
2
,
…
,
x
n
x 
1
​
 ,x 
2
​
 ,…,x 
n
​
 , nhân mỗi giá trị với một trọng số (weight) 
w
i
w 
i
​
  tương ứng, cộng thêm một hệ số lệch (bias) 
b
b, rồi đưa kết quả qua một hàm kích hoạt để tạo ra đầu ra:

z
=
∑
i
=
1
n
w
i
⋅
x
i
+
b
=
w
T
x
+
b
z= 
i=1
∑
n
​
 w 
i
​
 ⋅x 
i
​
 +b=w 
T
 x+b
a
=
σ
(
z
)
a=σ(z)
Trong đó 
σ
σ là hàm kích hoạt Sigmoid.

3.2 Kiến trúc mạng trong dự án

  ĐẦU VÀO (3072 chiều)
  ┌─────────────────────────┐
  │ Pixel 1, 2, ..., 3072  │
  └──────────┬──────────────┘
             │ (kết nối đầy đủ - Fully Connected)
  ┌──────────▼──────────────┐
  │  TẦNG ẨN (64 Neurons)  │  ← sigmoid activation
  │  h₁, h₂, ..., h₆₄     │
  └──────────┬──────────────┘
             │ (64 kết nối)
  ┌──────────▼──────────────┐
  │  TẦNG ĐẦU RA (1 Neuron)│  ← sigmoid activation
  │  Output: 0.0 ~ 1.0     │
  └─────────────────────────┘
       ↓             ↓
   ≥ 0.5 = MÈO   < 0.5 = CHÓ
Tổng số tham số (trọng số) của mô hình:

Tầng ẩn: 
3072
×
64
+
64
=
196
,
672
3072×64+64=196,672 tham số
Tầng đầu ra: 
64
×
1
+
1
=
65
64×1+1=65 tham số
Tổng cộng: 196,737 tham số có thể học được
4. CHI TIẾT TỪNG THUẬT TOÁN AI
4.1 Tiền Xử Lý Ảnh (Image Preprocessing)
File: 
image_processor.py

Trước khi đưa ảnh vào mạng Neural, ảnh cần được chuẩn hóa về cùng định dạng:

Bước 1: Chuyển sang RGB
python

img = Image.open(image_path).convert('RGB')
Đảm bảo mọi ảnh đều có đúng 3 kênh màu (Red, Green, Blue). Ảnh đen trắng hoặc có kênh Alpha (RGBA) đều được chuyển đổi.

Bước 2: Resize về 32×32 pixel
python

img = img.resize((32, 32), Image.BILINEAR)
Kích thước cố định: Mạng Neural yêu cầu vector đầu vào có kích thước không đổi.
Nội suy Bilinear: Tính giá trị pixel mới bằng trung bình trọng số của 4 pixel lân cận, cho kết quả mịn hơn so với nội suy Nearest Neighbor.
p
n
e
w
=
(
1
−
t
)
(
1
−
u
)
⋅
p
00
+
t
(
1
−
u
)
⋅
p
10
+
(
1
−
t
)
u
⋅
p
01
+
t
u
⋅
p
11
p 
new
​
 =(1−t)(1−u)⋅p 
00
​
 +t(1−u)⋅p 
10
​
 +(1−t)u⋅p 
01
​
 +tu⋅p 
11
​
 
Bước 3: Chuẩn hóa (Normalization)
python

arr = np.asarray(img, dtype=np.float32) / 255.0
Giá trị pixel gốc nằm trong khoảng 
[
0
,
255
]
[0,255]. Chia cho 
255
255 để đưa về 
[
0.0
,
1.0
]
[0.0,1.0].

Tại sao phải chuẩn hóa?
Nếu giữ nguyên giá trị 
[
0
,
255
]
[0,255], gradient lan truyền ngược sẽ cực kỳ lớn (vì tích 
w
⋅
x
w⋅x rất lớn), dẫn đến vanishing/exploding gradient — mô hình không thể học được.

Bước 4: Trải phẳng (Flatten)
python

X.append(img_arr.flatten())  # shape: (32×32×3,) = (3072,)
Ảnh 
32
×
32
×
3
32×32×3 được trải thẳng thành vector 1 chiều gồm 3,072 phần tử. Đây là vector đầu vào 
x
x cho mạng Neural.

4.2 Hàm Kích Hoạt Sigmoid
File: 
activation.py

σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
​
 
python

@staticmethod
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
Tại sao chọn Sigmoid?
Đầu ra trong khoảng 
(
0
,
1
)
(0,1): Phù hợp để diễn giải là xác suất — xuất ra 0.87 có nghĩa là "87% là mèo".
Phân loại nhị phân: Sigmoid + ngưỡng 0.5 là lựa chọn kinh điển cho bài toán 2 nhãn.
Vi phân được: Có đạo hàm đẹp, phục vụ lan truyền ngược.
Đạo hàm của Sigmoid:
σ
′
(
z
)
=
σ
(
z
)
⋅
(
1
−
σ
(
z
)
)
σ 
′
 (z)=σ(z)⋅(1−σ(z))
python

@staticmethod
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sig = Activation.sigmoid(x)
    return sig * (1 - sig)
Đạo hàm này được dùng trực tiếp trong bước lan truyền ngược (Backpropagation).

4.3 Forward Propagation — Lan Truyền Xuôi
File: 
neural_network.py

Lan truyền xuôi là quá trình tính toán dự đoán từ ảnh đầu vào đến đầu ra cuối cùng.

Tầng ẩn (64 neurons song song):
Với mỗi neuron 
j
j trong tầng ẩn:

z
j
(
1
)
=
w
j
(
1
)
T
⋅
x
+
b
j
(
1
)
z 
j
(1)
​
 =w 
j
(1)T
​
 ⋅x+b 
j
(1)
​
  
h
j
=
σ
(
z
j
(
1
)
)
h 
j
​
 =σ(z 
j
(1)
​
 )

Sau khi qua 64 neurons, ta có vector ẩn 
h
=
[
h
1
,
h
2
,
…
,
h
64
]
h=[h 
1
​
 ,h 
2
​
 ,…,h 
64
​
 ].

python

hidden_activations = []
for neuron in self.hidden:
    a = np.apply_along_axis(neuron.forward, 1, X)
    hidden_activations.append(a)
H = np.column_stack(hidden_activations)  # shape: (batch_size, 64)
Tầng đầu ra (1 neuron):
z
(
2
)
=
w
(
2
)
T
⋅
h
+
b
(
2
)
z 
(2)
 =w 
(2)T
 ⋅h+b 
(2)
  
y
^
=
σ
(
z
(
2
)
)
∈
(
0
,
1
)
y
^
​
 =σ(z 
(2)
 )∈(0,1)

python

out = np.apply_along_axis(self.output.forward, 1, H)
return out.reshape(-1, 1)  # shape: (batch_size, 1)
Quy tắc phán quyết:

y
^
≥
0.5
y
^
​
 ≥0.5 → Mèo (Cat)
y
^
<
0.5
y
^
​
 <0.5 → Chó (Dog)
4.4 Binary Cross-Entropy Loss — Hàm Mất Mát
File: 
loss.py

Sau khi có dự đoán 
y
^
y
^
​
 , ta cần đo mức độ "sai" của mô hình so với nhãn đúng 
y
y. Hàm mất mát Binary Cross-Entropy (BCE) là tiêu chuẩn cho bài toán phân loại nhị phân:

L
=
−
1
N
∑
i
=
1
N
[
y
i
⋅
log
⁡
(
y
^
i
)
+
(
1
−
y
i
)
⋅
log
⁡
(
1
−
y
^
i
)
]
L=− 
N
1
​
  
i=1
∑
N
​
 [y 
i
​
 ⋅log( 
y
^
​
  
i
​
 )+(1−y 
i
​
 )⋅log(1− 
y
^
​
  
i
​
 )]
python

eps = 1e-12
y_pred = np.clip(y_pred, eps, 1 - eps)   # Tránh log(0) = -∞
loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
return loss.mean()
Giải thích trực quan:
Nhãn thật 
y
y	Dự đoán 
y
^
y
^
​
 	Loss
1 (Mèo)	0.99	
−
log
⁡
(
0.99
)
≈
0.01
−log(0.99)≈0.01 ← Nhỏ, tốt
1 (Mèo)	0.01	
−
log
⁡
(
0.01
)
≈
4.6
−log(0.01)≈4.6 ← Lớn, tệ
0 (Chó)	0.01	
−
log
⁡
(
1
−
0.01
)
≈
0.01
−log(1−0.01)≈0.01 ← Nhỏ, tốt
0 (Chó)	0.99	
−
log
⁡
(
1
−
0.99
)
≈
4.6
−log(1−0.99)≈4.6 ← Lớn, tệ
Mục tiêu huấn luyện là tối thiểu hóa 
L
L qua từng epoch.

Tại sao dùng eps = 1e-12 (clip)?
Nếu 
y
^
=
0
y
^
​
 =0 thì 
log
⁡
(
0
)
=
−
∞
log(0)=−∞, chương trình sẽ bị tràn số (overflow). Clip giá trị về 
[
10
−
12
,
 
1
−
10
−
12
]
[10 
−12
 , 1−10 
−12
 ] ngăn điều đó.

4.5 Backpropagation — Lan Truyền Ngược
File: 
neural_network.py
 | 
neuron.py

Backpropagation là thuật toán cốt lõi giúp mạng học được từ dữ liệu. Nó dùng quy tắc dây chuyền (Chain Rule) của giải tích để tính đạo hàm của hàm mất mát 
L
L theo từng trọng số 
w
w, sau đó cập nhật trọng số theo chiều làm giảm 
L
L.

Bước 1: Tính gradient tầng đầu ra
∂
L
∂
y
^
=
−
y
y
^
+
1
−
y
1
−
y
^
∂ 
y
^
​
 
∂L
​
 =− 
y
^
​
 
y
​
 + 
1− 
y
^
​
 
1−y
​
 
python

eps = 1e-7
grad_output = -(y / (preds + eps)) + ((1 - y) / (1 - preds + eps))
grad_output = grad_output / batch_size   # Trung bình hóa theo batch
Bước 2: Chain Rule qua hàm Sigmoid (tầng đầu ra)
δ
(
2
)
=
∂
L
∂
y
^
⋅
σ
′
(
z
(
2
)
)
=
∂
L
∂
y
^
⋅
y
^
(
1
−
y
^
)
δ 
(2)
 = 
∂ 
y
^
​
 
∂L
​
 ⋅σ 
′
 (z 
(2)
 )= 
∂ 
y
^
​
 
∂L
​
 ⋅ 
y
^
​
 (1− 
y
^
​
 )
python

# Trong neuron.py
dz = self.sigmoid_derivative(a) * grad_output
Bước 3: Gradient theo trọng số đầu ra
∂
L
∂
w
j
(
2
)
=
δ
(
2
)
⋅
h
j
∂w 
j
(2)
​
 
∂L
​
 =δ 
(2)
 ⋅h 
j
​
  
∂
L
∂
b
(
2
)
=
δ
(
2
)
∂b 
(2)
 
∂L
​
 =δ 
(2)
 

python

grad_weights = dz * inputs      # inputs ở đây là H[i] (hidden activations)
grad_bias    = dz
Bước 4: Truyền ngược về tầng ẩn
∂
L
∂
h
j
=
δ
(
2
)
⋅
w
j
(
2
)
∂h 
j
​
 
∂L
​
 =δ 
(2)
 ⋅w 
j
(2)
​
 
python

grad_input = dz * self.weights   # shape: (hidden_units,)
Bước 5: Chain Rule qua tầng ẩn
Với mỗi neuron 
j
j trong tầng ẩn:

δ
j
(
1
)
=
∂
L
∂
h
j
⋅
σ
′
(
z
j
(
1
)
)
δ 
j
(1)
​
 = 
∂h 
j
​
 
∂L
​
 ⋅σ 
′
 (z 
j
(1)
​
 )
∂
L
∂
w
i
j
(
1
)
=
δ
j
(
1
)
⋅
x
i
∂w 
ij
(1)
​
 
∂L
​
 =δ 
j
(1)
​
 ⋅x 
i
​
 
python

for idx, neuron in enumerate(self.hidden):
    for i in range(batch_size):
        grad_in, grad_w, grad_b = neuron.backward(X[i], grad_H[i, idx])
        neuron.weights -= self.optimizer.lr * grad_w
        neuron.bias    -= self.optimizer.lr * grad_b
4.6 Stochastic Gradient Descent (SGD) — Bộ Tối Ưu Hóa
File: 
optimizer.py

Sau khi có gradient 
∇
w
L
∇ 
w
​
 L, SGD cập nhật trọng số theo công thức:

w
n
e
w
=
w
o
l
d
−
α
⋅
∂
L
∂
w
w 
new
​
 =w 
old
​
 −α⋅ 
∂w
∂L
​
 
Trong đó 
α
α (alpha) là tốc độ học (learning rate) — một siêu tham số (hyperparameter) người dùng cài đặt.

python

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr
Cập nhật thực tế trong mạng:

python

# Cập nhật trọng số tầng đầu ra
self.output.weights -= self.optimizer.lr * grad_w
self.output.bias    -= self.optimizer.lr * grad_b
# Cập nhật trọng số tầng ẩn
neuron.weights -= self.optimizer.lr * grad_w
neuron.bias    -= self.optimizer.lr * grad_b
Ảnh hưởng của Learning Rate 
α
α:
Giá trị 
α
α	Hành vi
α
α quá nhỏ (< 0.001)	Học rất chậm, cần rất nhiều epoch
α
α phù hợp (0.01–0.05)	Hội tụ ổn định, loss giảm đều
α
α quá lớn (> 0.5)	Dao động, loss không giảm hoặc tăng vọt
Dự án mặc định sử dụng 
α
=
0.03
α=0.03, phù hợp cho dataset nhỏ.

4.7 Online Learning — Học Trực Tuyến Theo Phản Hồi Người Dùng
File: 
app_gui.py
 — Endpoint /api/predict/correct

Đây là cơ chế Human-in-the-Loop — một trong những tính năng độc đáo nhất của dự án. Khi AI dự đoán sai, người dùng bấm "👎 Sai rồi", hệ thống sẽ:

Bước 1: Tải lại trọng số hiện tại từ model.npz
Bước 2: Chạy forward pass trên ảnh sai
Bước 3: Chạy 1 bước backward với nhãn đúng (Learning Rate cao 
α
=
0.15
α=0.15)
Bước 4: Lưu trọng số đã cập nhật ngay lập tức vào model.npz

python

# Tải mô hình từ file
data = np.load(MODEL_PATH, allow_pickle=True)
net = NeuralNetwork(input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS)
# ... khôi phục trọng số ...
# Dùng learning rate cao hơn bình thường để "ép" học nhanh
net.set_optimizer(SGD(lr=0.15))
# Chạy học trực tuyến trên 1 mẫu
preds = net.forward(x)
net.backward(x, np.array([correct_y]), preds)
# Lưu lại ngay lập tức
np.savez(MODEL_PATH, ...)
Đây là hình thức học tăng cường (Reinforcement Learning đơn giản): Mỗi phản hồi "sai" của người dùng = một tín hiệu phần thưởng âm → mô hình điều chỉnh trọng số ngay lập tức.

5. PHÂN TÍCH TỪNG FILE MÃ NGUỒN
📄 activation.py — Hàm Kích Hoạt
python

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))            # σ(z) = 1/(1+e^{-z})
    @staticmethod
    def sigmoid_derivative(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)                  # σ'(z) = σ(z)·(1-σ(z))
sigmoid            = Activation.sigmoid         # Alias tiện dụng
sigmoid_derivative = Activation.sigmoid_derivative
Vai trò: Cung cấp hàm phi tuyến để mạng có khả năng học các mẫu phức tạp. Nếu không có hàm kích hoạt, mạng nhiều tầng chỉ tương đương một mạng 1 tầng (hồi quy tuyến tính thuần).

📄 loss.py — Hàm Mất Mát
python

def binary_cross_entropy(y_true, y_pred):
    eps    = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)     # Ổn định số học
    loss   = -(y_true * np.log(y_pred)
             + (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()                          # Trả về giá trị vô hướng
def binary_cross_entropy_derivative(y_true, y_pred):
    eps    = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])
Vai trò: Là "thước đo" chất lượng dự đoán. Mục tiêu của toàn bộ huấn luyện là tối thiểu hóa giá trị này.

📄 optimizer.py — Bộ Tối Ưu Hóa
python

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr        # Learning Rate α — siêu tham số duy nhất
Vai trò: Quy định tốc độ mà mạng "bước" theo hướng gradient giảm loss. Đơn giản nhưng hiệu quả cho dataset nhỏ.

📄 neuron.py — Neuron Đơn Lẻ
python

class Neuron:
    def __init__(self, input_dim, activation=None):
        self.weights   = np.random.randn(input_dim) * 0.01   # Khởi tạo nhỏ
        self.bias      = 0.0
        self.activation = activation or self.sigmoid
    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias   # z = w·x + b
        return self.activation(z)                       # a = σ(z)
    def backward(self, inputs, grad_output):
        z          = np.dot(self.weights, inputs) + self.bias
        a          = self.activation(z)
        dz         = self.sigmoid_derivative(a) * grad_output   # Chain Rule
        grad_weights = dz * inputs          # ∂L/∂w = δ · x
        grad_bias    = dz                   # ∂L/∂b = δ
        grad_input   = dz * self.weights    # ∂L/∂x (truyền về tầng trước)
        return grad_input, grad_weights, grad_bias
Lý do khởi tạo trọng số bằng * 0.01: Nếu khởi tạo quá lớn, các neuron sẽ bão hòa Sigmoid ngay từ đầu (output ≈ 0 hoặc 1), gradient ≈ 0 → mạng không học được (vanishing gradient).

📄 neural_network.py — Mạng Neural Hoàn Chỉnh
python

class NeuralNetwork:
    def __init__(self, input_dim, hidden_units=64):
        self.hidden = [Neuron(input_dim, activation=sigmoid)
                       for _ in range(hidden_units)]       # 64 neurons ẩn
        self.output = Neuron(hidden_units, activation=sigmoid)  # 1 neuron ra
    def forward(self, X):
        H   = np.column_stack([np.apply_along_axis(n.forward, 1, X)
                                for n in self.hidden])     # (N, 64)
        out = np.apply_along_axis(self.output.forward, 1, H)
        return out.reshape(-1, 1)                          # (N, 1)
    def backward(self, X, y, preds):
        y          = y.reshape(-1, 1)                      # Fix broadcasting
        grad_out   = -(y/(preds+eps)) + ((1-y)/(1-preds+eps))
        grad_out  /= batch_size
        # ... Truyền ngược qua output neuron rồi qua từng hidden neuron
y.reshape(-1, 1) — Fix bug quan trọng: NumPy broadcasting: (N,) / (N,1) tạo ra ma trận (N,N) không mong muốn. Reshape về (N,1) đảm bảo phép chia element-wise đúng chiều.

📄 image_processor.py — Bộ Xử Lý Ảnh
python

IMAGE_SIZE = (32, 32)      # Kích thước cố định của mọi ảnh đầu vào
def load_image(path):
    img = Image.open(path).convert('RGB')           # Chuẩn hóa sang 3 kênh
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)    # Resize về 32×32
    arr = np.asarray(img, dtype=np.float32) / 255.0 # Chuẩn hóa [0,1]
    return arr                                       # shape: (32, 32, 3)
def load_dataset(dataset_dir):
    for label_name, label_value in [('cat', 1), ('dog', 0)]:
        for img_path in class_dir.glob('*.jpg'):
            X.append(load_image(img_path).flatten())  # flatten → (3072,)
            y.append(label_value)                      # 0 hoặc 1
    return np.stack(X), np.array(y, dtype=np.float32)
📄 app_gui.py — Máy Chủ Web & Giao Diện
File lớn nhất (~62KB), đảm nhận đa nhiệm:

API Endpoint	Phương thức	Chức năng
GET /	GET	Trả về toàn bộ HTML giao diện
GET /api/status	GET	Trả về trạng thái mô hình, dataset, training
GET /api/dataset/list	GET	Liệt kê file ảnh trong dataset
POST /api/train	POST	Bắt đầu huấn luyện trong luồng nền
POST /api/dataset/upload	POST	Nhận ảnh base64, lưu vào dataset
POST /api/predict	POST	Nhận ảnh base64, trả về nhãn dự đoán
POST /api/predict/correct	POST	Online Learning: sửa dự đoán sai
6. LUỒNG XỬ LÝ TỔNG THỂ
Luồng Huấn Luyện (Training Flow):

NGƯỜI DÙNG tải ảnh lên dataset
        │
        ▼
/api/dataset/upload → Lưu ảnh vào dataset/cat/ hoặc dataset/dog/
        │
NGƯỜI DÙNG bấm "Bắt đầu Huấn luyện" (chọn epochs, learning rate)
        │
        ▼
/api/train → Khởi động luồng nền (threading.Thread)
        │
        ▼
load_dataset() → Đọc tất cả ảnh → X (N×3072), y (N,)
        │
        ▼
NeuralNetwork.forward(X) → Tính ŷ (N×1)
        │
        ▼
binary_cross_entropy(y, ŷ) → Tính Loss L
        │
        ▼
NeuralNetwork.backward(X, y, ŷ) → Tính gradient ∂L/∂w
        │
        ▼
SGD: w ← w - α·∂L/∂w  (cập nhật tất cả 196,737 tham số)
        │
        ▼ (lặp lại N epochs)
        │
np.savez("model.npz") → Lưu trọng số vào file
Luồng Nhận Dạng (Prediction Flow):

NGƯỜI DÙNG kéo thả ảnh / chụp camera
        │
        ▼
base64 image → /api/predict
        │
        ▼
Giải mã base64 → PIL.Image → resize 32×32 → chuẩn hóa → flatten (3072,)
        │
        ▼
np.load("model.npz") → Khôi phục trọng số vào NeuralNetwork
        │
        ▼
NeuralNetwork.forward(x) → σ(z²) → ŷ ∈ (0,1)
        │
   ŷ ≥ 0.5?
   /         \
 Có           Không
 │             │
MÈO (Cat)    CHÓ (Dog)
        │
        ▼
Trả về: {"label": "Mèo (Cat)", "confidence": 0.87, "raw_probability": 0.87}
Luồng Học Trực Tuyến (Online Learning Flow):

AI dự đoán sai → NGƯỜI DÙNG bấm "👎 Sai rồi"
        │
        ▼
/api/predict/correct nhận ảnh + nhãn đúng
        │
        ▼
np.load("model.npz") → Khôi phục mạng
        │
        ▼
SGD(lr=0.15) → Học rate cao để "ép" học nhanh
        │
        ▼
1 vòng forward + backward trên ảnh sai đó
        │
        ▼
np.savez("model.npz") → Ghi đè trọng số mới
        │
        ▼
NGƯỜI DÙNG bấm nhận dạng lại → Kết quả đã cập nhật!
7. GIAO DIỆN WEB VÀ API SERVER
Công nghệ Frontend:
Thành phần	Công nghệ	Vai trò
Layout	HTML5 Semantic	Cấu trúc giao diện
Phong cách	CSS3 Vanilla (Dark Mode)	Giao diện tối, glassmorphism
Biểu đồ Loss	HTML5 Canvas API	Vẽ đồ thị loss theo thời gian thực
Camera PC	WebRTC getUserMedia()	Stream trực tiếp từ webcam
Camera điện thoại	HTML5 <input capture="environment">	Mở camera native trên iOS/Android
Giao tiếp API	Fetch API (async/await)	Gọi backend không đồng bộ
Polling	setTimeout đệ quy	Cập nhật tiến trình huấn luyện
Công nghệ Backend:
Thành phần	Công nghệ	Vai trò
HTTP Server	http.server.HTTPServer	Xử lý HTTP requests
Đa luồng	threading.Thread	Huấn luyện không block giao diện
Serialization	json	API request/response
Lưu model	numpy.savez / numpy.load	Lưu và tải trọng số
Xử lý ảnh	PIL.Image + base64	Decode và tiền xử lý ảnh upload
Truy cập từ Điện Thoại:
Server được bind trên 0.0.0.0 (tất cả giao diện mạng):

python

server = HTTPServer(("0.0.0.0", PORT), GUIRequestHandler)
Điện thoại và máy tính cùng mạng Wi-Fi có thể truy cập qua IP nội bộ:

http://192.168.x.x:8000
8. PHÂN TÍCH ƯU/NHƯỢC ĐIỂM VÀ HƯỚNG PHÁT TRIỂN
Ưu điểm:
Ưu điểm	Giải thích
Minh bạch tuyệt đối	Toàn bộ thuật toán viết tay, không có "hộp đen"
Giá trị học thuật cao	Học được toán học đằng sau AI
Không phụ thuộc	Chỉ cần Python + NumPy + Pillow
Học liên tục	Online Learning từ phản hồi người dùng
Đa nền tảng	Chạy trên PC và điện thoại
Nhược điểm & Hướng cải thiện:
Vấn đề	Nguyên nhân	Hướng giải quyết
Chậm khi dataset lớn	Dùng Python loop, không dùng ma trận hóa hoàn toàn	Vector hóa toàn bộ bằng NumPy matrix ops
Chỉ phân biệt 2 loài	Kiến trúc phân loại nhị phân	Nâng lên Softmax + One-hot (multi-class)
Độ chính xác giới hạn	Ảnh resize về 32×32, mất nhiều chi tiết	Tăng lên 64×64 hoặc dùng CNN
Không có Dropout	Có thể overfit nếu dataset nhỏ	Thêm Dropout layer (tắt ngẫu nhiên neurons)
SGD đơn giản	Không có momentum	Nâng lên Adam Optimizer
HTTP thay vì HTTPS	Camera WebRTC yêu cầu HTTPS trên điện thoại	Dùng ngrok hoặc self-signed SSL
Lộ trình nâng cấp đề xuất:

Phiên bản hiện tại (v1.0)
    └── MLP 2 tầng, Sigmoid, SGD, 32×32 ảnh
         │
    Phiên bản 2.0
    └── CNN: Thêm lớp Tích chập (Convolution) + Max Pooling
         │    → Nhận diện đặc trưng cục bộ (tai, mũi, mắt)
         │
    Phiên bản 3.0
    └── Multi-class: 10+ loài vật (Softmax + Cross-entropy đa lớp)
         │
    Phiên bản 4.0
    └── Transfer Learning: Dùng ResNet/MobileNet pretrained weights
9. KẾT LUẬN
Dự án Animal AI Recognition Studio thành công xây dựng một hệ thống nhận dạng hình ảnh con vật đầy đủ tính năng, trong đó mọi thuật toán AI đều được lập trình từ đầu mà không dựa vào bất kỳ framework học sâu nào.

Những gì đã đạt được:
✅ Triển khai hoàn chỉnh mạng Neural 2 tầng với 196,737 tham số có thể học
✅ Cài đặt Backpropagation thủ công qua Chain Rule
✅ Binary Cross-Entropy Loss với xử lý ổn định số học
✅ SGD Optimizer với learning rate có thể tùy chỉnh
✅ Online Learning từ phản hồi người dùng (Human-in-the-Loop)
✅ Lưu/tải trọng số bền vững qua model.npz
✅ Giao diện Web hiện đại, dark mode, biểu đồ real-time
✅ Hỗ trợ camera WebRTC (PC) và HTML5 Capture (điện thoại)
✅ Triển khai đa thiết bị qua mạng Wi-Fi nội bộ

Giá trị cốt lõi:
Dự án này không chỉ là một ứng dụng nhận dạng con vật — đây là một phòng thí nghiệm học tập AI hoàn chỉnh, nơi người dùng có thể tận tay thấy từng con số của ma trận trọng số thay đổi qua từng epoch, hiểu rõ tại sao loss giảm, và trực tiếp "dạy" cho AI học từ sai lầm thông qua phản hồi của mình.

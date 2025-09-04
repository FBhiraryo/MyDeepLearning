from numpy import loadtxt
from tensorflow.keras.models import load_model

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# 保存したモデルを読み込む
loaded_model = load_model('trained_model.h5')

# モデルを評価する（必要に応じて）
_, accuracy = loaded_model.evaluate(X, y)
print('Loaded Model Accuracy: %.2f' % (accuracy * 100))

# 新しいデータで予測を行う
new_predictions = (loaded_model.predict(X) > 0.5).astype(int)

# 最初の5つの予測結果を表示
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), new_predictions[i], y[i]))
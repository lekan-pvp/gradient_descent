import numpy as np

def sigmoid(x):
	""" Функция активации сигмоиды на входе х"""
	return 1 / (1 + np.exp(-x))

def forward_propagation(input_data, weights, bias):
	"""
	Вычисляет операцию прямого распространения перцептрона и возвращает результат после применения
	сигмоидной функции активации
	:param input_data: входные данные
	:param weights: веса
	:param bias: смещение
	:return:
	"""
	# берем скалярное произведение входных данных и весов и добавляем смещение
	return sigmoid(np.dot(input_data, weights) + bias) #уравнение перцептрона

def calculate_error(Y, Y_predicted):
	"""
	Вычисляет ошибку двоичной кросс-энтропии
	:param Y: метка
	:param Y_predicted: спрогнозируемый результат
	:return: ошибка двоичной кросс-энтропии
	"""
	return -Y * np.log(Y_predicted) - (1 - Y) * np.log(1 - Y_predicted)

def gradient(target, actual, X):
	"""
	Градиент весов и смещения
	:param target: метка
	:param actual: прогноз
	:param X: вход
	:return: градиент весов, градиент смещения
	"""
	dW = - (target - actual) * X
	db = target - actual
	return dW, db

def update_parameters(W, b, dW, db, learning_rate):
	"""
	Обновление значений весов и смещения
	:param W: вес
	:param dW: градиент веса, т.е. значение, которое корректирует вес
	:param db: градиент смещения, т.е. значение, которое корректирует смещение
	:param learning_rate: скорость обучения, по сути коэфициент изменения
	:return: новый вес, новое смещение
	"""
	W = W - dW * learning_rate
	b = b - db * learning_rate
	return W, b

def train(X, Y, weights, bias, epochs, learning_rate):
	"""
	Обучение перцептрона с использованием стохастического обновления
	:param X: Массив объектов входных данных
	:param Y: На выходе этикетки(метки)
	:param weights: Массив весов
	:param bias: Смещение
	:param epochs: Эпохи
	:param learning_rate: Скорость обучения
	:return: Новый массив весов, новое смещение
	"""
	sum_error = 0.0
	for i in range(epochs):
		for j in range(len(X)):
			Y_predicted = forward_propagation(X[j], weights.T, bias) # прогнозируемая метка
			sum_error = sum_error + calculate_error(Y[j], Y_predicted) # вычисляем ошибку
			dW, db = gradient(Y[j], Y_predicted, X[j]) # находим градиент
			weights, bias = update_parameters(weights, bias, dW, db, learning_rate) # обновляем параметры
		print("epochs: ", i, "error: ", sum_error)
		sum_error = 0 #переинициализируем сумму ошибок для следующей эпохи
	return weights, bias

# инициализируем параметры
# две точки данных
X = np.array(
   [[2.78, 2.55],
	[1.46, 2.36],
	[3.39, 4.40],
	[1.38, 1.85],
	[3.06, 3.00],
	[7.62, 2.75],
	[5.33, 2.08],
	[6.92, 1.77],
	[8.67, -0.24],
	[7.67, 3.50]])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # актуальные метки
weights = np.array([0.0, 0.0]) # веса перцептрона
bias = 0.0 # величина смещения
learning_rate = 0.1 # скорость обучения
epochs = 10 # эпохи
print("Before training")
print("weights:", weights, "bias:", bias)

weights, bias = train(X, Y, weights, bias, epochs, learning_rate) # train the function

print("\nAfter training")
print("weights:", weights, "bias:", bias)
# Predict values
predicted_labels = forward_propagation(X, weights.T, bias)
print("Target labels:  ", Y)
print("Predicted label:", (predicted_labels > 0.5) * 1)




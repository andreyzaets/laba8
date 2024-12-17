import numpy as np
import tensorflow as tf

np.random.seed(0)
X_data = np.random.rand(1000, 1).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 0.1, (1000, 1))

k = tf.Variable(tf.random.normal([1]), name="k")
b = tf.Variable(tf.zeros([1]), name="b")

def model(X):
    return k * X + b

optimizer = tf.optimizers.SGD(learning_rate=0.05)

for epoch in range(20000):
    with tf.GradientTape() as tape:
        y_pred = model(X_data)
        loss = tf.reduce_mean(tf.square(y_data - y_pred))
    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))
    if epoch % 1000 == 0:
        print(f"Епоха {epoch}: втрата={loss.numpy():.4f}, k={k.numpy()[0]:.4f},b = {b.numpy()[0]: .4f}")

print(f"\nРезультат: k={k.numpy()[0]:.4f}, b={b.numpy()[0]:.4f}")

import tensorflow as tf

# 플레이스 홀더 노드 정의
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

# 세션 (session) 을 만들고 플레이스홀더 노드를 통해 값 입력받음
sess = tf.Session()

print(sess.run(c, feed_dict={a: 1.0, b: 3.0}))
print(sess.run(c, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

# 연산 추가
d = 100 * c

print(sess.run(d, feed_dict={a: 1.0, b: 3.0}))
print(sess.run(d, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

# 세션 close
sess.close()

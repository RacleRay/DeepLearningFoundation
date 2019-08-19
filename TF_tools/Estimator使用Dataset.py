import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        # estimater要求输入是字典
        return {"x": parsed_line[:-1]}, parsed_line[-1:]

    dataset = (tf.contrib.data.TextLineDataset(file_path)
               .skip(1)
               .map(decode_csv))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    # 输入有确定值，make_one_shot_iterator
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3)

# 自定义输入函数，不能有参数，使用lambda表达式
classifier.train(input_fn=lambda : my_input_fn('../datasets/iris_training.csv', True, 100)


test_results = classifier.evaluate(
    input_fn=lambda: my_input_fn("../datasets/iris_test.csv", False, 1))

print("\nTest accuracy: %g %%" % (test_results["accuracy"]*100))
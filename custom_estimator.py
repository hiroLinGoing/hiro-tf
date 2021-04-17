import pandas as pd
import tensorflow as tf

TRAIN_URL = "/Users/train/tf/data/iris_training.csv"
TEST_URL = "/Users/train/tf/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

def load_data(y_name='Species'):
    """
    加载数据
    Returns the iris dataset as (train_x, train_y), (test_x, test_y).
    从本地路径加载数据，拆分成训练数据和标签, pop出来的是标签列 名称为Species
    """
    train = pd.read_csv(TRAIN_URL, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(TEST_URL, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """
    构建输入函数input_fn
    :param features:  由列名和特征值构成
    :param labels: 所有例子的标签值
    :param batch_size:
    :return:
    """
    # 将输入转化成Dataset格式
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #随机打乱 重复 批量获取样本
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # 输入层
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    # 隐藏层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # 预测结果
    predicted_classes = tf.argmax(logits, 1)
    # 如果是预测类型 直接返回预测结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Convert the labels to a one-hot tensor of shape (length of features, 3)
    # and with a on-value of 1 for each one-hot vector of length 3.
    onehot_labels = tf.one_hot(labels, 3, 1, 0)
    # Compute loss.
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    #评估模式 不需要再进行下一步优化，直接返回
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main():
    batch_size = 100
    train_steps = 1000

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x = dict(train_x)
    test_x = dict(test_x)

    #模型特征列，定义输入
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # 构建模型
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # 训练模型
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)

    # 评估模型
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 构建预测模型
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x, batch_size=batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(SPECIES[class_id], 100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main())

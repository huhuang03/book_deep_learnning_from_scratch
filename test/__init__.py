from util.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_text) = load_mnist(one_hot_label=False)
    print(t_train)

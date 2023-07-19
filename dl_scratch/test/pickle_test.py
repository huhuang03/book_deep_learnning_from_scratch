import pickle
import os
test_pickle = 'runtime/foo.pickle'

if not os.path.exists(os.path.dirname(test_pickle)):
    os.mkdir(os.path.dirname(test_pickle))

class Bar:
    def __init__(self):
        self.items = []

    def save(self):
        with open(test_pickle, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> 'Bar':
        with open(test_pickle, 'rb') as f:
            return pickle.load(f)

class Foo:
    def __init__(self, i):
        self.i = i

    def hello(self):
        pass



foo = Foo(100)
bar = Bar()
bar.items.append(foo)
print(bar.items)
bar.save()

bar1 = Bar.load()
print(bar1.items)

# foo.save()
# foo = Foo.load(test_pickle)
# foo.i = 200
# foo.save()
# foo = Foo.load(test_pickle)
# foo.hello()

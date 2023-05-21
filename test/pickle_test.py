import pickle
import os
test_pickle = 'runtime/foo.pickle'

if not os.path.exists(os.path.dirname(test_pickle)):
    os.mkdir(os.path.dirname(test_pickle))

class Foo:
    def __init__(self, i):
        self.i = i

    def hello(self):
        print(f'Foo hello {self.i}')

    def save(self):
        with open(test_pickle, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(f_path) -> 'Foo':
        with open(f_path, 'rb') as f:
            return pickle.load(f)


foo = Foo(100)
foo.save()
foo = Foo.load(test_pickle)
foo.i = 200
foo.save()
foo = Foo.load(test_pickle)
foo.hello()

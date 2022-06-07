from fileinput import filename
import pickle
class Address:
    def __init__(self, city, state):
        self.city = city
        self.state = state
class Person:
    def __init__(self, name, age, numbers, address, map):
        self.name = name
        self.age = age
        self.numbers = numbers
        self.address = address
        self.map = map
filename = "trial.pkl"
a = Person('Kayak', 22, [9,0,8,7,72,19], Address('Bengaluru', 'Karnataka'), {'Age': 22, 'Height': 32})
pickle.dump(a, open(filename, "wb"), pickle.HIGHEST_PROTOCOL)
print(pickle.load(open(filename, 'rb')).map)

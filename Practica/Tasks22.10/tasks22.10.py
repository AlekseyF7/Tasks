# class BankAccount:
#     def __init__(self, balance):
#         self.balance = balance
#
#     def deposit(self, amount):
#         if amount > 0:
#             self.balance += amount
#
#     def get_balance(self):
#         return self.balance
# account = BankAccount (1000)
# account.deposit (500)
# print(account.get_balance())

class Book:

    def __init__(self, title, year, author):
        self.title = title
        self.author = author
        self.year = year


    def info(self):
        return f'Title: {self.title}, year: {self.year}, author: {self.author}'

class Ebook(Book):
    def __init__(self, title, year, author, format):
        self.format = format
        super().__init__(author, year, title)

    def info(self):
        return f'Title: {self.title}, year: {self.year}, author: {self.author}, format: {self.format}'

book = Ebook('Война и мир', 1905, 'Лев Толстой', 'электронная')
print(book.info())

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        """"Сложение векторов: v1 + v2"""
        return Vector(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar):
        """Умножение на число: v * 5"""
        return Vector(self.x * scalar, self.y * scalar)
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(1, 1)
print (v1 + v2) #Vector (3, 4)
print (v1 * 3) #Vector (6, 9)
print(v2 * 10)
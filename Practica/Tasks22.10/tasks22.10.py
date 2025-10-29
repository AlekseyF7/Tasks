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

# class Vector:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def __add__(self, other):
#         """"Сложение векторов: v1 + v2"""
#         return Vector(self.x + other.x, self.y + other.y)
#     def __mul__(self, scalar):
#         """Умножение на число: v * 5"""
#         return Vector(self.x * scalar, self.y * scalar)
#     def __str__(self):
#         return f"Vector({self.x}, {self.y})"
#
# v1 = Vector(2, 3)
# v2 = Vector(1, 1)
# print (v1 + v2) #Vector (3, 4)
# print (v1 * 3) #Vector (6, 9)
# print(v2 * 10)

from datetime import datetime
from abc import ABC, abstractmethod


class Printable(ABC):
    @abstractmethod
    def print_info(self):
        pass


class Book(Printable):

    def print_info(self):
        return f'Title: {self.title}, year: {self.year}, author: {self.author}'

    def __init__(self, title, year, author):
        self.title = title
        self.year = year
        self.author = author

    def __str__(self):
        return self.print_info()

    def __eq__(self, other):
        return isinstance(other, Book) and self.title == other.title and self.year == other.year and self.author == other.author

    @classmethod
    def from_string(cls, string):
        title, author, year = string.split(',')
        return cls(title, int(year), author)

    @property
    def age(self):
        # Вычисляет 'возраст' книги с момента ее издания
        return datetime.now().year - self.year

    @age.setter
    def age(self, value):
        # Вычисляет год издания на основе 'возраста' книги
        self.year = datetime.now().year - value


class Ebook(Book):
    def __init__(self, title, year, author, format):
        self.format = format
        super().__init__(author, year, title)


book1 = Ebook('Война и мир', 1905, 'Лев Толстой', 'электронная')
book2 = Book.from_string("1984,Оруэлл,1949")
print(book1)
print(book2)
print(book1==book2)
print(book1.age)
print(book2.age)

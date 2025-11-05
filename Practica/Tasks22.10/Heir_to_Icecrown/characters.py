import random


class Character:
    """Базовый класс для всех персонажей игры"""
    def __init__(self, name: str, hp: int, min_attack: int, max_attack: int):
        self.name = name
        self.hp = hp
        self.max_hp = hp
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.alive = True

    def is_alive(self):
        """Проверка жив ли персонаж"""
        if self.alive and self.hp == 0:
            self.alive = False
        return self.alive

    def get_attack_damage(self):
        """Возвращает случайный урон в диапазоне"""
        return random.randint(self.min_attack, self.max_attack)

    def take_damage(self, damage: int):
        """Получение урона"""
        self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def heal(self, amount: int):
        """Механика восстановления здоровья (чрз max_hp чтобы не больше максимума)"""
        self.hp = min(self.hp +amount, self.max_hp)

class Knight(Character):
    """Рыцарь Альянса - за кого играет пользователь"""
    def __init__(self, name: str):
        # Основная атака: 15–25 урона
        super().__init__(name=name, hp=100, min_attack=15, max_attack=25)
        self.honor = 100 # Честь Альянса
        self.lich_sword = False # Есть ли меч у игрока

    def use_lich_sword(self):
        """Использовать силу меча Артаса"""
        if self.lich_sword:
            self.honor -= 10 # Использование снижает честь Альянса
            return random.randint(30, 40) # Возрастает урон (мощная атака 30–40 урона)
        else:
            return 0 # Нельзя использовать

    def gain_honor(self, amount: int = 5):
        """Повыить честь Альянса"""
        self.honor = max(0, self.honor + amount)

    def lose_honor(self, amount: int = 10):
        """Понизить честь Альянса"""
        self.honor = min(0, self.honor - amount)

class Undead_Warrior(Character):
    """Нежить (враг) из 1 битвы"""
    def __init__(self):
        # Атака: 12–18
        super().__init__(name='Восставший рыцарь', hp=50, min_attack=12, max_attack=18)

class Dragon(Character):
    """Сарторакс - Дракон Смерти из 2 битвы"""
    def __init__(self):
        # Обычная атака: 30–40
        super().__init__(name='Сарторакс', hp=300, min_attack=30, max_attack=40)
        self.phase = 1 # Фаза боя (в будущем)

    def roar(self) -> int:
        """Рёв — особая атака: 45–55 урона."""
        return random.randint(45, 55)





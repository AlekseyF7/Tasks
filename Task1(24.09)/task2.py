n = int(input("Введите число N: "))
if n < 2:
    print('число N должно быть больше 2!!!')
elif n == 2:
    print('2')
else:
    prime = [True] * (n+1)
    for i in range(2, n+1, 2):
        prime[i] = False

    p = 3
    while p**2 <= n:
        if prime[p]:
            for i in range(p**2, n+1, 2*p):
                prime[i] = False
        p += 2

    primes = [2]
    for i in range(2, n+1):
        if prime[i]:
            primes.append(i)
print(f"Список простых чисел в диапозоне [2, {n}] с помощью алгоритма “Решето Эратосфена”: {primes}")
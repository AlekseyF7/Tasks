def sred_znach(s: str) -> int:
    s = s.split()
    su = 0
    for i in s:
        su += int(i)
    sr = su / len(s)
    return sr

res = []

while True:
    s = str(input())
    if s == '':
        break
    res.append(sred_znach(s))

for i in range(len(res)):
    print(round(res[i], 2))
import re
import sys
import numpy as np

EPS = 1e-9

def povorot(tab, bazis, stroka, stolb):
    elem = tab[stroka, stolb]
    tab[stroka] /= elem
    for i in range(tab.shape[0]):
        if i != stroka:
            tab[i] -= tab[i, stolb] * tab[stroka]
    bazis[stroka] = stolb

def vibor_stroki(kol, svobod):
    indeksy = [i for i in range(len(kol)) if kol[i] > EPS]
    if not indeksy:
        return None
    otn = svobod[indeksy] / kol[indeksy]
    minv = otn.min()
    for i in indeksy:
        if abs(svobod[i] / kol[i] - minv) < 1e-12:
            return i
    return None

def simplex_metod(lp):
    A, b, c = lp.A, lp.b, lp.c
    m, n = A.shape
    bazis = lp.bazis.copy()

    T = np.zeros((m + 1, n + 1))
    T[:m, :n] = A
    T[:m, -1] = b
    T[-1, :n] = [-1 if t == 'isk' else 0 for t in lp.tip]

    for i, j in enumerate(bazis):
        if lp.tip[j] == 'isk':
            T[-1] += T[i]

    while True:
        r = T[-1, :-1]
        vozm = np.where(r > EPS)[0]
        if vozm.size == 0:
            break
        stolb = vozm[0]
        stroka = vibor_stroki(T[:m, stolb], T[:m, -1])
        if stroka is None:
            return "bezgranichno", None, None
        povorot(T, bazis, stroka, stolb)

    if abs(T[-1, -1]) > 1e-7:
        return "net_resheniya", None, None

    ostavit = [j for j, t in enumerate(lp.tip) if t != 'isk']
    A2, b2, c2 = T[:m, ostavit], T[:m, -1], lp.c[ostavit]
    n2 = len(ostavit)

    T2 = np.zeros((m + 1, n2 + 1))
    T2[:m, :n2] = A2
    T2[:m, -1] = b2
    T2[-1, :n2] = c2

    bazis2 = [ostavit.index(j) for j in bazis if j in ostavit]
    for i, j in enumerate(bazis2):
        T2[-1] -= T2[-1, j] * T2[i]

    while True:
        r = T2[-1, :-1]
        vozm = np.where(r > EPS)[0]
        if vozm.size == 0:
            break
        stolb = vozm[0]
        stroka = vibor_stroki(T2[:m, stolb], T2[:m, -1])
        if stroka is None:
            return "bezgranichno", None, None
        povorot(T2, bazis2, stroka, stolb)

    x = np.zeros(n2)
    for i, j in enumerate(bazis2):
        x[j] = T2[i, -1]

    f = -T2[-1, -1]
    if lp.minim:
        f = -f

    return "optimum_nayden", f, x[:lp.tip.count('osn')]

class lineal_prog:
    def __init__(self, c, A, b, bazis, tip, minim):
        self.c = c
        self.A = A
        self.b = b
        self.bazis = bazis
        self.tip = tip
        self.minim = minim

def kanonic(sense, c, A, b, rels):
    m, n = A.shape
    minim = sense == 'MIN'
    if minim:
        c = -c

    tip = ['osn'] * n
    dop = []
    bazis = [-1] * m
    k = n

    # обработка свободных переменных (могут быть отрицательными)

    free_vars = [] 
    if hasattr(kanonic, "free_vars"):
        free_vars = kanonic.free_vars

    if free_vars:
        print(f"Обнаружены свободные переменные: {[i+1 for i in free_vars]}")
        new_cols = []
        new_c = []
        for j in range(n):
            if j in free_vars:
                new_c += [c[j], -c[j]]
                new_cols += [A[:, j], -A[:, j]]
                tip += ['osn', 'osn']
            else:
                new_c.append(c[j])
                new_cols.append(A[:, j])
        A = np.column_stack(new_cols)
        c = np.array(new_c)
        n = len(c)
    # 🔹 конец добавления

    for i in range(m):
        if b[i] < -EPS:
            A[i] *= -1
            b[i] *= -1
            rels[i] = {'<=':'>=','>=':'<=','=':'='}[rels[i]]

    for i, znak in enumerate(rels):
        if znak == '<=':
            s = np.zeros(m); s[i] = 1
            dop.append(s)
            tip.append('dop')
            bazis[i] = k
            k += 1
        elif znak == '>=':
            s = np.zeros(m); s[i] = -1
            a = np.zeros(m); a[i] = 1
            dop += [s, a]
            tip += ['dop', 'isk']
            bazis[i] = k + 1
            k += 2
        else:
            a = np.zeros(m); a[i] = 1
            dop.append(a)
            tip.append('isk')
            bazis[i] = k
            k += 1

    if dop:
        A = np.hstack([A] + [x.reshape(-1, 1) for x in dop])
        c = np.concatenate([c, np.zeros(A.shape[1] - len(c))])

    return lineal_prog(c, A, b, bazis, tip, minim)

def read_file(path):
    reg = re.compile(r'([+\-]?\s*\d*(?:\.\d*)?)\s*x(\d+)')

    def razobrat(expr):
        coeffs = {}
        for m in reg.finditer(expr):
            num, idx = m.groups()
            num = num.replace(' ', '')
            val = 1.0 if num in ('', '+') else -1.0 if num == '-' else float(num)
            coeffs[int(idx) - 1] = coeffs.get(int(idx) - 1, 0.0) + val
        return coeffs

    lines = [l.strip() for l in open(path, encoding='utf-8').read().splitlines() if l.strip()]
    sense = lines[0].split()[0].upper()
    obj = razobrat(lines[0].split(None, 1)[1])

    constr = []
    free_vars = []
    for ln in lines[1:]:
        if ln.lower().startswith('free'):
            free_vars = [int(i) - 1 for i in ln.split()[1:]]
            continue
        znak = re.search(r'(<=|>=|=)', ln).group(1)
        left, right = ln.split(znak)
        coeffs = razobrat(left)
        rhs = float(right.strip())
        constr.append((coeffs, znak, rhs))

    n = max(max(d) for d, _, _ in constr + [(obj, '', 0)]) + 1
    m = len(constr)
    c = np.zeros(n)
    for j, v in obj.items():
        c[j] = v
    A = np.zeros((m, n))
    b = np.zeros(m)
    rels = []
    for i, (d, znak, rhs) in enumerate(constr):
        for j, v in d.items():
            A[i, j] = v
        b[i] = rhs
        rels.append(znak)

    kanonic.free_vars = free_vars

    return sense, c, A, b, rels


def main(argv):
    if len(argv) != 2:
        print("Чтобы запустить код введите: python simplex_main.py <problem.txt>")
        return

    try:
        sense, c, A, b, rels = read_file(argv[1])
        lp = kanonic(sense, c, A, b, rels)
        status, f, x = simplex_metod(lp)

        print("\nРезультат вычислений:")
        if status == "optimum_nayden":
            print(f"Оптимальное значение F = {f:.4f}")
            for i, val in enumerate(x, 1):
                print(f"  x{i} = {val:.4f}")
        elif status == "bezgranichno":
            print("Решение не ограничено.")
        else:
            print("Нет допустимых решений.")

    except Exception as e:
        print("Ошибка выполнения:", e)

if __name__ == "__main__":
    main(sys.argv)

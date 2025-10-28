import numpy as np
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class SimplexTwoPhase:
    def __init__(self):
        self.c_vec = None
        self.A = None
        self.b = None
        self.signs = None
        self.opt_type = None
        self.total_vars = 0
        self.free_vars = []
        self.var_map = []

    def read_input(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.opt_type = lines[0].lower()
        self.c_vec = np.array(list(map(float, lines[1].split())))
        self.total_vars = len(self.c_vec)

        free_line = lines[2].lower()
        if free_line.startswith("free"):
            items = free_line.split()[1:]
            if items:
                self.free_vars = [int(i) - 1 for i in items]
            start = 3
        else:
            self.free_vars = []
            start = 2

        signs, matrix, rhs = [], [], []
        for ln in lines[start:]:
            parts = ln.split()
            signs.append(parts[0])
            matrix.append(list(map(float, parts[1:-1])))
            rhs.append(float(parts[-1]))

        self.signs = signs
        self.A = np.array(matrix)
        self.b = np.array(rhs)

        if self.free_vars:
            self.expand_free_vars()

    def expand_free_vars(self):
        print("Обнаружены свободные переменные:", [i + 1 for i in self.free_vars])
        self.var_map = list(range(self.total_vars))
        new_c, new_cols = [], []

        for j in range(self.total_vars):
            if j in self.free_vars:
                new_c.extend([self.c_vec[j], -self.c_vec[j]])
                new_cols.extend([self.A[:, j], -self.A[:, j]])
                pos, neg = len(new_c) - 2, len(new_c) - 1
                self.var_map[j] = (pos, neg)
            else:
                new_c.append(self.c_vec[j])
                new_cols.append(self.A[:, j])

        self.c_vec = np.array(new_c)
        self.A = np.column_stack(new_cols)
        self.total_vars = len(new_c)

    def to_standard_form(self):
        m = len(self.b)
        ext = self.A.copy()
        base_vars = []
        artificial = []
        count_slack = count_surplus = count_art = 0
        idx = self.total_vars

        for i, s in enumerate(self.signs):
            col = np.zeros(m)
            if s == "<=":
                col[i] = 1
                ext = np.column_stack([ext, col])
                base_vars.append(idx)
                idx += 1
                count_slack += 1
            elif s == ">=":
                col[i] = -1
                ext = np.column_stack([ext, col])
                idx += 1
                count_surplus += 1
                col = np.zeros(m)
                col[i] = 1
                ext = np.column_stack([ext, col])
                artificial.append(idx)
                base_vars.append(idx)
                idx += 1
                count_art += 1
            elif s == "=":
                col[i] = 1
                ext = np.column_stack([ext, col])
                artificial.append(idx)
                base_vars.append(idx)
                idx += 1
                count_art += 1

        self.A_ext = ext
        self.artificial = artificial
        self.basis = base_vars
        self.num_slack = count_slack
        self.num_surplus = count_surplus
        self.num_art = count_art

        print(f"Форма: slack={count_slack}, surplus={count_surplus}, art={count_art}")

    def phase_one(self):
        m, n = self.A_ext.shape
        if self.num_art == 0:
            print("Фаза I не требуется — допустимое решение найдено.")
            self.table1 = np.zeros((m + 1, n + 1))
            self.table1[:m, :n] = self.A_ext
            self.table1[:m, n] = self.b
            return

        table = np.zeros((m + 1, n + 1))
        table[:m, :n] = self.A_ext
        table[:m, n] = self.b

        for i in self.artificial:
            table[m, i] = 1

        for i in range(m):
            if self.basis[i] in self.artificial:
                table[m, :] -= table[i, :]

        print("\n Фаза 1:")
        step = 0
        while True:
            step += 1
            col = -1
            min_val = -1e-9
            for j in range(n):
                if table[m, j] < min_val:
                    min_val = table[m, j]
                    col = j
            if col == -1:
                break

            row = -1
            ratio_min = float("inf")
            for i in range(m):
                if table[i, col] > 1e-9:
                    ratio = table[i, n] / table[i, col]
                    if ratio < ratio_min:
                        ratio_min = ratio
                        row = i
            if row == -1:
                raise ValueError("Фаза 1: неограничено")

            pivot = table[row, col]
            table[row, :] /= pivot
            for i in range(m + 1):
                if i != row:
                    table[i, :] -= table[i, col] * table[row, :]

            self.basis[row] = col
            print(f"  Итерация {step}: x{col + 1} вошла, W = {-table[m, n]:.4f}")

        W = -table[m, n]
        print(f"Фаза 1 завершена, W = {W:.6f}")
        if abs(W) > 1e-6:
            raise ValueError("Задача несовместна")
        self.table1 = table

    def phase_two(self):
        m, n = self.A_ext.shape
        table = np.zeros((m + 1, n + 1))
        table[:m, :] = self.table1[:m, :]

        c_ext = np.zeros(n)
        c_ext[:self.total_vars] = self.c_vec
        table[m, :n] = c_ext if self.opt_type == "min" else -c_ext

        for i, b in enumerate(self.basis):
            if b < n and table[m, b] != 0:
                table[m, :] -= table[m, b] * table[i, :]

        print("\n Фаза 2:")
        step = 0
        while step < 1000:
            step += 1
            col = -1
            min_val = -1e-9
            for j in range(n):
                if j in self.artificial:
                    continue
                if table[m, j] < min_val:
                    min_val = table[m, j]
                    col = j
            if col == -1:
                print(f"Оптимум найден (итерация {step})")
                break

            row = -1
            min_ratio = float("inf")
            for i in range(m):
                if table[i, col] > 1e-9:
                    val = table[i, n] / table[i, col]
                    if val < min_ratio:
                        min_ratio = val
                        row = i
            if row == -1:
                raise ValueError("Фаза 2: неограничено")

            pivot = table[row, col]
            table[row, :] /= pivot
            for i in range(m + 1):
                if i != row:
                    table[i, :] -= table[i, col] * table[row, :]

            old = self.basis[row]
            self.basis[row] = col
            z_val = -table[m, n] if self.opt_type == "min" else table[m, n]
            print(f"  Шаг {step}: x{col + 1} вошла, x{old + 1} вышла, Z = {z_val:.4f}")

        sol = np.zeros(self.total_vars)
        for i, b in enumerate(self.basis):
            if b < self.total_vars:
                sol[b] = table[i, n]

        z_val = -table[m, n] if self.opt_type == "min" else table[m, n]
        return sol, z_val

    def restore_vars(self, sol):
        if not self.free_vars:
            return sol
        restored = np.zeros(len(self.var_map))
        for j, mapping in enumerate(self.var_map):
            if isinstance(mapping, tuple):
                pos, neg = mapping
                restored[j] = sol[pos] - sol[neg]
            else:
                restored[j] = sol[mapping]
        return restored

    def solve(self):
        self.to_standard_form()
        self.phase_one()
        sol, z = self.phase_two()
        return sol, z


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "problem.txt"
    model = SimplexTwoPhase()
    model.read_input(filename)
    solution, z_value = model.solve()

    print("\nРезультат вычислений:")
    if solution is None:
        print("Нет допустимого решения.")
    else:
        result = model.restore_vars(solution)
        for i, val in enumerate(result, start=1):
            print(f"x{i} = {val:.4f}")
        print(f"Оптимальное значение Z = {z_value:.4f}")

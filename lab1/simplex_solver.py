import numpy as np


class SimplexException(Exception):
    pass


class SimplexSolver:
    def __init__(self, objective, C, constraints, tol=1e-9, max_iters=1000):
        self.objective = objective.lower()
        if self.objective not in ("max", "min"):
            raise ValueError("objective must be 'max' or 'min'")
        self.C = np.array(C, dtype=float)
        self.constraints = [(np.array(A, dtype=float), sign, float(b)) for A, sign, b in constraints]
        self.tol = tol
        self.max_iters = max_iters

        self.num_orig = len(self.C)

        self.tableau = None
        self.basic_vars = []
        self.var_names = []
        self.artificial_indices = []
        self.slack_surplus_indices = []
        self.num_rows = 0
        self.num_cols = 0

    def solve(self):
        self._build_tableau()

        if len(self.artificial_indices) > 0:
            self._phase_one()

            w_val = self._get_phase_one_value()
            if w_val > self.tol:
                raise SimplexException(f"Задача несовместна: минимальное значение суммы искусственных переменных = {w_val:.6g} > 0")

            self._remove_artificials()

        x_opt, z_opt = self._phase_two()
        return x_opt, z_opt

    def _build_tableau(self):
        processed = []
        for A, sign, b in self.constraints:
            if b < 0:
                A = -A
                b = -b
                if sign == "<=":
                    sign = ">="
                elif sign == ">=":
                    sign = "<="
            processed.append((A, sign, b))

        rows = len(processed)
        orig = self.num_orig

        slack_indices = []
        surplus_indices = []
        artificial_indices = []

        A_matrix = []
        b_vec = []

        extra_cols = []

        for i, (A, sign, b) in enumerate(processed):
            row = list(A)
            if sign == "<=":
                extra_cols.append(("slack", i))
            elif sign == ">=":
                extra_cols.append(("surplus+art", i))
            elif sign == "=":
                extra_cols.append(("art", i))
            else:
                raise ValueError("Unknown sign")
            A_matrix.append(np.array(row, dtype=float))
            b_vec.append(b)

        slack_count = sum(1 for kind, _ in extra_cols if kind == "slack")
        surplus_count = sum(1 for kind, _ in extra_cols if kind == "surplus+art")
        art_count = sum(1 for kind, _ in extra_cols if kind in ("surplus+art", "art"))

        A_full = np.hstack([np.array(A_matrix, dtype=float)])
        col_list = []
        var_names = [f"x{i+1}" for i in range(orig)]
        slack_idx_map = {}
        surplus_idx_map = {}
        art_idx_map = {}

        for idx, (kind, row_i) in enumerate(extra_cols):
            col = np.zeros(rows, dtype=float)
            if kind == "slack":
                col[row_i] = 1.0
                name = f"s_slack_{row_i+1}"
                slack_idx_map[row_i] = len(var_names)
                self.slack_surplus_indices.append(len(var_names))
            elif kind == "surplus+art":
                col_surplus = np.zeros(rows, dtype=float)
                col_surplus[row_i] = -1.0
                col_list.append(col_surplus)
                name_surplus = f"s_surplus_{row_i+1}"
                surplus_idx_map[row_i] = len(var_names)
                self.slack_surplus_indices.append(len(var_names))
                var_names.append(name_surplus)

                col_art = np.zeros(rows, dtype=float)
                col_art[row_i] = 1.0
                col = None
                col_list.append(col_art)
                name_art = f"a_art_{row_i+1}"
                art_idx_map[row_i] = len(var_names)+1
                self.artificial_indices.append(len(var_names)+1)
                var_names.append(name_art)
                continue
            elif kind == "art":
                col[row_i] = 1.0
                name = f"a_art_{row_i+1}"
                art_idx_map[row_i] = len(var_names)
                self.artificial_indices.append(len(var_names))
            else:
                raise RuntimeError("Unexpected kind")

            col_list.append(col)
            var_names.append(name)

        if col_list:
            extras = np.column_stack(col_list)
            A_full = np.hstack([A_full, extras])

        A_full = np.array(A_full, dtype=float)
        b_vec = np.array(b_vec, dtype=float)

        rows, cols = A_full.shape
        basic_vars = [-1] * rows
        for j in range(cols):
            col = A_full[:, j]
            ones = np.abs(col - 1.0) <= self.tol
            zeros = np.abs(col) <= self.tol
            if ones.sum() == 1 and zeros.sum() == rows - 1:
                i = np.where(ones)[0][0]
                if basic_vars[i] == -1:
                    basic_vars[i] = j

        for i in range(rows):
            if basic_vars[i] == -1:
                col = np.zeros(rows, dtype=float)
                col[i] = 1.0
                A_full = np.hstack([A_full, col.reshape(-1,1)])
                var_names.append(f"a_auto_{i+1}")
                idx_new = A_full.shape[1] - 1
                basic_vars[i] = idx_new
                self.artificial_indices.append(idx_new)

        self.tableau = np.zeros((rows + 1, A_full.shape[1] + 1), dtype=float)
        self.tableau[:rows, :A_full.shape[1]] = A_full
        self.tableau[:rows, -1] = b_vec

        self.basic_vars = basic_vars
        self.var_names = var_names
        self.num_rows = rows
        self.num_cols = A_full.shape[1]

        self.artificial_indices = sorted(list(set(self.artificial_indices)))

        self._A_full = A_full
        self._b = b_vec

    def _phase_one(self):
        rows = self.num_rows
        cols = self.num_cols

        obj = np.zeros(cols + 1, dtype=float)
        for ai in self.artificial_indices:
            obj[ai] = 1.0
        obj[-1] = 0.0

        self.tableau[-1, :] = obj.copy()

        for i in range(rows):
            basic = self.basic_vars[i]
            if basic in self.artificial_indices:
                self.tableau[-1, :] -= self.tableau[i, :]

        it = 0
        while True:
            it += 1
            if it > self.max_iters:
                raise SimplexException("Превышено число итераций в фазе I")
            entering = self._choose_entering_phase_one()
            if entering is None:
                break
            leaving = self._choose_leaving(entering)
            if leaving is None:
                raise SimplexException("Фаза I: задача неограничена (неожиданно).")
            self._pivot(leaving, entering)

    def _choose_entering_phase_one(self):
        row = self.tableau[-1, :-1]

        candidates = np.where(row > self.tol)[0]
        if candidates.size == 0:
            return None

        return int(candidates[np.argmax(row[candidates])])

    def _choose_entering_phase_two(self):
        row = self.tableau[-1, :-1]
        candidates = np.where(row < -self.tol)[0]
        if candidates.size == 0:
            return None

        return int(candidates[np.argmin(row[candidates])])

    def _choose_leaving(self, entering):
        col = self.tableau[:self.num_rows, entering]
        rhs = self.tableau[:self.num_rows, -1]
        positive = col > self.tol

        if not np.any(positive):
            return None

        ratios = np.full(self.num_rows, np.inf)
        ratios[positive] = rhs[positive] / col[positive]

        idx = np.argmin(ratios)
        if ratios[idx] == np.inf:
            return None
        return int(idx)

    def _pivot(self, row_i, col_j):
        pivot = self.tableau[row_i, col_j]
        if abs(pivot) < self.tol:
            raise SimplexException("Pivot слишком мал")

        self.tableau[row_i, :] = self.tableau[row_i, :] / pivot

        for i in range(self.tableau.shape[0]):
            if i == row_i:
                continue
            factor = self.tableau[i, col_j]
            if abs(factor) > self.tol:
                self.tableau[i, :] -= factor * self.tableau[row_i, :]

        self.basic_vars[row_i] = col_j

    def _get_phase_one_value(self):
        return self.tableau[-1, -1]

    def _remove_artificials(self):
        art_set = set(self.artificial_indices)
        for i in range(self.num_rows):
            basic = self.basic_vars[i]
            if basic in art_set:
                row = self.tableau[i, :self.num_cols]
                for j in range(self.num_cols):
                    if j in art_set:
                        continue
                    if abs(row[j]) > self.tol:

                        self._pivot(i, j)
                        break

        keep_cols = [j for j in range(self.num_cols) if j not in art_set]
        new_cols = len(keep_cols)
        new_table = np.zeros((self.num_rows + 1, new_cols + 1))
        for idx, j in enumerate(keep_cols):
            new_table[:, idx] = self.tableau[:, j]

        new_table[:, -1] = self.tableau[:, -1]

        idx_map = {old: new for new, old in enumerate(keep_cols)}
        new_basic = []
        for b in self.basic_vars:
            if b in idx_map:
                new_basic.append(idx_map[b])
            else:

                new_basic.append(-1)
        self.tableau = new_table
        self.num_cols = new_cols
        self.basic_vars = new_basic

        self.var_names = [self.var_names[j] for j in keep_cols]

        self.artificial_indices = []

        for i in range(self.num_rows):
            if self.basic_vars[i] == -1:
                for j in range(self.num_cols):
                    col = self.tableau[:self.num_rows, j]
                    ones = np.abs(col - 1.0) <= self.tol
                    zeros = np.abs(col) <= self.tol
                    if ones.sum() == 1 and zeros.sum() == self.num_rows - 1 and ones[i]:
                        self.basic_vars[i] = j
                        break
                if self.basic_vars[i] == -1:

                    pass

    def _phase_two(self):
        cols = self.num_cols
        rows = self.num_rows

        c_full = np.zeros(cols, dtype=float)

        for j, name in enumerate(self.var_names):
            if name.startswith("x"):
                try:
                    idx = int(name[1:]) - 1
                    if 0 <= idx < self.num_orig:
                        c_full[j] = self.C[idx]
                except:
                    pass

        if self.objective == "max":
            c_full = -c_full

        self.tableau[-1, :-1] = c_full
        self.tableau[-1, -1] = 0.0

        for i in range(rows):
            bvar = self.basic_vars[i]
            if bvar is None or bvar < 0:
                continue
            coeff = 0.0

            if bvar < len(c_full):
                coeff = c_full[bvar]
            else:
                coeff = 0.0
            if abs(coeff) > 0:
                self.tableau[-1, :] -= coeff * self.tableau[i, :]

        it = 0
        while True:
            it += 1
            if it > self.max_iters:
                raise SimplexException("Превышено число итераций в фазе II")
            entering = self._choose_entering_phase_two()
            if entering is None:
                break
            leaving = self._choose_leaving(entering)
            if leaving is None:
                raise SimplexException("Задача неограничена.")
            self._pivot(leaving, entering)

        x = np.zeros(self.num_orig, dtype=float)
        for i in range(rows):
            bv = self.basic_vars[i]
            if bv >= 0 and bv < len(self.var_names):
                name = self.var_names[bv]
                if name.startswith("x"):
                    try:
                        idx = int(name[1:]) - 1
                    except:
                        continue
                    if 0 <= idx < self.num_orig:
                        x[idx] = self.tableau[i, -1]

        z = self.tableau[-1, -1]

        return x, z

    def pretty_tableau(self):
        header = [f"{name}" for name in self.var_names] + ["RHS"]
        rows = []
        for i in range(self.num_rows):
            rows.append(np.round(self.tableau[i, :], 6))
        rows.append(np.round(self.tableau[-1, :], 6))
        return header, rows

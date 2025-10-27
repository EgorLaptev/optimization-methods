import json
from simplex_solver import SimplexSolver


def main(argv=None):
    with open('data/input.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    objective = data["objective"]
    C = data["C"]
    constraints = [
        (item["A"], item["sign"], item["b"])
        for item in data["constraints"]
    ]

    solver = SimplexSolver(objective, C, constraints)

    try:
        x_opt, z_opt = solver.solve()
        print("Оптимальное решение:")
        for i, x in enumerate(x_opt, start=1):
            print(f"x{i} = {x:.4f}")
        print(f"Z = {z_opt:.4f}")
    except SimplexException as e:
        print("Задача не имеет допустимого решения:")
        print(e)


if __name__ == "__main__":
    main()

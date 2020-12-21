from classical_solution_v4 import FourBodyProblem

three_body_problem = FourBodyProblem(1, 1500, 0.2, 1.1, 2, 1.3, [0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5], [0, 0.01], [0, -0.01], [-0.03, 0], [0.05, 0.05])

results = three_body_problem.calculate_trajectories()

print(str(results))

three_body_problem.display_trajectories(animated=False, save_animation=False)

three_body_problem.to_csv("output.csv")
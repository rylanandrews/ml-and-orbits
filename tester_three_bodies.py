from classical_solution_three_bodies import ThreeBodyProblem

three_body_problem = ThreeBodyProblem(1, 1, 5, 1, [0.5, 0], [-0.5, 0], [0, 0.5], [0, 0.01], [0, -0.01], [0, 0])

results = three_body_problem.calculate_trajectories()

print(str(results))

three_body_problem.display_trajectories(save_animation=False)

three_body_problem.to_csv("output.csv")
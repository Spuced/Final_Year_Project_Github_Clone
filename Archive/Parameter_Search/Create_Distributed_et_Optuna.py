import optuna

optuna.create_study(study_name="distributed-et", direction="maximize", storage="mysql://root@localhost/example")
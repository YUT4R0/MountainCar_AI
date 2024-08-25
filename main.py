import os
from mountain_car_dql import MountainCarDQL


def get_best_model(models):
    policies = [int(filename.split("_")[3].split(".")[0])
                for filename in models]
    best_policy = max(policies)
    best_result = f"result_mountaincar_dql_{best_policy}.pt"
    return best_result


if __name__ == '__main__':
    ALPHA = 0.01
    GAMMA = 0.9
    NET_SYNC_RATE = 50000
    REP_MEMO_SIZE = 100000
    MINI_BATCH_SIZE = 32
    DIVISIONS = 20
    STEPS_PER_EPOCH = 1000
    EPISODES = 20000

    mountaincar = MountainCarDQL(
        learning_rate=ALPHA,
        discount_factor=GAMMA,
        network_sync_rate=NET_SYNC_RATE,
        replay_memory_size=REP_MEMO_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        num_divisions=DIVISIONS,
        steps_per_epoch=STEPS_PER_EPOCH
    )

    current_directory = os.getcwd()
    files_in_directory = os.listdir(current_directory)

    model_files = [filename for filename in files_in_directory if filename.startswith(
        "result_mountaincar_dql_")]

    if not model_files:
        mountaincar.train(episodes=EPISODES, render=False)
    best_model = get_best_model(models=model_files)
    mountaincar.run(
        episodes=10,
        model_filepath=best_model,
    )

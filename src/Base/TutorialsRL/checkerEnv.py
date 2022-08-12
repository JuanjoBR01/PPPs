from stable_baselines3.common.env_checker import check_env
from snakeEnvV2 import SnekEnv

env = SnekEnv()

check_env(env)

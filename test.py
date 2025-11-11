from src.first_agent import Agent
from env.first_env import Env
from shapely.geometry import box

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import matplotlib.pyplot as plt
def run():
    
    # Get geospatial data
    north, south, east, west = 35.6873, 35.6864, 139.764, 139.7629
    origin = (west, south)
    bbox = box(west, south, east, north)  # shapely's box 

    #Create Environment
    env = Env(
        origin,
        bbox,
        num_agents=1,
        battery_capacity=500
    ) 

    # Add Monitor for logging
    log_dir = "./logs/sac_rescue/"
    wrapped_env = Monitor(env, filename=log_dir + "monitor.csv")

    env = DummyVecEnv([lambda: wrapped_env])


    #normalize env
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    #create model
    model = SAC(
        "MlpPolicy",
        wrapped_env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/sac_rescue/"
    )
    #Load the saved model
    model = SAC.load(
        "models/sac_rescue_final",
        env=wrapped_env,          # must pass the same environment or a new compatible one
        tensorboard_log="./logs/sac_rescue/"
    )
    
    print("Starting training...")
    for _ in range(50):
        
        # Train the model
        model.learn(
            total_timesteps=1_000,
            progress_bar=True
        )
        
        # Save model
        model.save("models/sac_rescue_final")
        print("Training complete! Model saved.")
        

        # Load the monitor CSV
        log_file = "./logs/sac_rescue/monitor.csv"
        df = pd.read_csv(log_file, skiprows=1)  # skip header info line

        # Dump to another csv file
        output_file = "./logs/sac_rescue/cleaned_log.csv"
        df.to_csv(output_file, mode='a', index=False, header=False)

    plot(df)
    
    wrapped_env.close()

def plot(df):
    # Compute cumulative timesteps
    df['cum_t'] = df['t'].cumsum()
    
    # Plot reward vs cumulative timesteps
    plt.plot(df['cum_t'], df['r'], label='Reward')
    plt.xlabel('Total Steps')
    plt.ylabel('Reward')
    plt.title('Reward vs Total Steps')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()

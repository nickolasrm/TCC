import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Define the CartPole environment
env = gym.make("CartPole-v0")

# Define the model
model = Sequential()
model.add(Dense(2, activation="softmax", input_dim=4))
model.compile()

observation, info = env.reset()
for step in range(200):
    # Get the action from the model
    action = np.argmax(model.predict(observation.reshape(1, -1)))
    # Perform the action and get the next observation and reward
    observation, reward, done, truncated, info = env.step(action)
    if done:
        break

import time

import gymnasium
import numpy as np
import pygame

import flappy_bird_gymnasium


def play(audio_on=True, render=True):
    env = flappy_bird_gymnasium.FlappyBirdEnvSimple()
    score = 0
    obs, _ = env.reset(seed=123)
    obs = np.expand_dims(obs, axis=0)

    done = False
    # action = 0
    while not done:
        if render:
            env.render()

            # observations

            last_pipe_horizontal_pos = obs[0][0]
            last_top_pipe_vertical_pos = obs[0][1]
            last_bottom_pipe_vertical_pos = obs[0][2]
            next_pipe_horizontal_pos = obs[0][3]
            next_top_pipe_vertical_pos = obs[0][4]
            next_bottom_pipe_vertical_pos = obs[0][5]
            next_next_pipe_horizontal_pos = obs[0][6]
            next_next_top_pipe_vertical_pos = obs[0][7]
            next_next_bottom_pipe_vertical_pos = obs[0][8]
            player_vertical_pos = obs[0][9]
            player_vertical_velocity = obs[0][10]
            player_rotation = obs[0][11]

            # next_pipe_center = (next_top_pipe_vertical_pos + next_bottom_pipe_vertical_pos) / 2
            # next_next_pipe_center = (next_next_top_pipe_vertical_pos + next_next_bottom_pipe_vertical_pos) / 2
            last_pipe_center = (last_top_pipe_vertical_pos + last_bottom_pipe_vertical_pos) / 2

            if player_vertical_pos > last_pipe_center:
                action = 1
            else:
                action = 0
                # if last_pipe_center > next_pipe_center:
                #     action = 0

        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        # Processing:
        obs, reward, done, _, info = env.step(action)
        obs = np.expand_dims(obs, axis=0)

        score += reward
        print(f"Obs: {obs}\n" f"Score: {score}\n")

        if render:
            time.sleep(1 / 30)

        if done:
            if render:
                env.render()
                time.sleep(0.5)
            break

    env.close()
    # assert obs.shape == (12,)
    assert info["score"] == 0
    np.testing.assert_allclose(score, 8.99999999999998)


def test_play():
    play(audio_on=False, render=False)


if __name__ == "__main__":
    play()

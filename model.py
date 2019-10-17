import tensorflow as tf
import numpy as np
import BeamNG_Env
import matplotlib.pyplot as plt
from skimage import transform


# Function for resizing image
def resize(image):
    # Greyscale Image
    x = np.mean(image, -1)
    # Normalize Pixel Values
    x = x / 255
    x = transform.resize(x, [200, 200])
    return (x)


# Apply discount to episode rewards & normalize
def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount


def rollout(batch_size):
    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []
    state = resize(env.reset())
    episode_num = 0
    action_repeat = 1
    reward = 0

    while True:

        # Run State Through Policy & Calculate Action
        feed = {X: state.reshape(1, 200, 200, 1)}
        action = sess.run(calc_action, feed_dict=feed)
        action = action[0][0]

        # Perform Action
        for i in range(action_repeat):
            state2, reward2, done = env.step(choice[action])
            reward += reward2
            if done:
                break

        # Store Results
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        # Update Current State
        reward = 0
        state = resize(state2)
        #here the image the A.I. receives can be showed
        #plt.imshow(state)
        #plt.show()
        if done:
            # Track Discounted Rewards
            rewardsFeed.append(rewards)
            discountedRewards.append(discount(rewards, gamma, normalize_r))

            if len(np.concatenate(rewardsFeed)) > batch_size:
                break

            # Reset Environment
            rewards = []
            state = resize(env.reset())
            episode_num += 1

    return np.stack(states), np.stack(actions), np.concatenate(rewardsFeed), np.concatenate(
        discountedRewards), episode_num

# This Method returns the Best suited Steering angle for the image input


def get_command(image):
    # Environment Parameters
    n_actions = 7
    n_epochs = 1000
    n = 0
    average = []
    step = 1
    batch_size = 5000

    # Define actions / steering angles
    choice = [0, 1, 2, 3, 4, 5, 6]

    # Hyper Parameters
    alpha = 1e-4
    gamma = 0.99
    normalize_r = True
    save_path = 'models/BeamDQN.ckpt'
    value_scale = 0.5
    entropy_scale = 0.00
    gradient_clip = 40

    # Conv Layers
    convs = [16, 32]
    kerns = [8, 8]
    strides = [4, 4]
    pads = 'valid'
    fc = 256
    activ = tf.nn.elu

    # Tensorflow Variables
    X = tf.placeholder(tf.float32, (None, 200, 200, 1), name='X')
    Y = tf.placeholder(tf.int32, (None,), name='actions')
    R = tf.placeholder(tf.float32, (None,), name='reward')
    N = tf.placeholder(tf.float32, (None), name='episodes')
    D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

    # Policy Network
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=convs[0],
        kernel_size=kerns[0],
        strides=strides[0],
        padding=pads,
        activation=activ,
        name='conv1')

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=convs[1],
        kernel_size=kerns[1],
        strides=strides[1],
        padding=pads,
        activation=activ,
        name='conv2')

    flat = tf.layers.flatten(conv2)

    dense = tf.layers.dense(
        inputs=flat,
        units=fc,
        activation=activ,
        name='fc')

    logits = tf.layers.dense(
        inputs=dense,
        units=n_actions,
        name='logits')

    value = tf.layers.dense(
        inputs=dense,
        units=1,
        name='value')

    calc_action = tf.multinomial(logits, 1)
    aprob = tf.nn.softmax(logits)
    action_logprob = tf.nn.log_softmax(logits)

    mean_reward = tf.divide(tf.reduce_sum(R), N)

    # Define Losses
    pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
    entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
    loss = pg_loss + value_loss - entropy_loss

    # Create Optimizer
    optimizer = tf.train.AdamOptimizer(alpha)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)  # gradient clipping
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars)

    # Initialize Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load model if exists
    saver = tf.train.Saver(tf.global_variables())
    load_was_success = True
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)
    except:
        print("No saved model to load. Starting new session")
        load_was_success = False
    else:
        print("Loaded Model: {}".format(load_path))
        saver = tf.train.Saver(tf.global_variables())
        step = int(load_path.split('-')[-1]) + 1

    image = resize(image)

    prob, val = sess.run([aprob, value], feed_dict={X: image.reshape(1, 200, 200, 1)})

    values = (prob[0][0], prob[0][1], prob[0][2], prob[0][3], prob[0][4], prob[0][5], prob[0][6])
    command = (values.index(max(values)))

    sess.close()
    tf.reset_default_graph()

    steering_angles = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
    return steering_angles[command]

# Run this to train the AI with BeamNG


if __name__ == '__main__':
    # Environment Parameters
    n_actions = 7
    n_epochs = 1000
    n = 0
    average = []
    step = 1
    batch_size = 5000

    # Define actions / steering angles
    choice = [0, 1, 2, 3, 4, 5, 6]

    # Hyper Parameters
    alpha = 1e-4
    gamma = 0.99
    normalize_r = True
    save_path = 'models/BeamDQN.ckpt'
    value_scale = 0.5
    entropy_scale = 0.00
    gradient_clip = 40

    env = BeamNG_Env.BeamHandler()

    # Conv Layers
    convs = [16, 32]
    kerns = [8, 8]
    strides = [4, 4]
    pads = 'valid'
    fc = 256
    activ = tf.nn.elu

    # Tensorflow Variables
    X = tf.placeholder(tf.float32, (None, 200, 200, 1), name='X')
    Y = tf.placeholder(tf.int32, (None,), name='actions')
    R = tf.placeholder(tf.float32, (None,), name='reward')
    N = tf.placeholder(tf.float32, (None), name='episodes')
    D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

    # Policy Network
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=convs[0],
        kernel_size=kerns[0],
        strides=strides[0],
        padding=pads,
        activation=activ,
        name='conv1')

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=convs[1],
        kernel_size=kerns[1],
        strides=strides[1],
        padding=pads,
        activation=activ,
        name='conv2')

    flat = tf.layers.flatten(conv2)

    dense = tf.layers.dense(
        inputs=flat,
        units=fc,
        activation=activ,
        name='fc')

    logits = tf.layers.dense(
        inputs=dense,
        units=n_actions,
        name='logits')

    value = tf.layers.dense(
        inputs=dense,
        units=1,
        name='value')

    calc_action = tf.multinomial(logits, 1)
    aprob = tf.nn.softmax(logits)
    action_logprob = tf.nn.log_softmax(logits)

    print(tf.trainable_variables())

    mean_reward = tf.divide(tf.reduce_sum(R), N)

    # Define Losses
    pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
    entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
    loss = pg_loss + value_loss - entropy_loss

    # Create Optimizer
    optimizer = tf.train.AdamOptimizer(alpha)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)  # gradient clipping
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars)

    # Initialize Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Load model if exists
    saver = tf.train.Saver(tf.global_variables())
    load_was_success = True
    try:
        save_dir = '/'.join(save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        saver.restore(sess, load_path)
    except:
        print("No saved model to load. Starting new session")
        load_was_success = False
    else:
        print("Loaded Model: {}".format(load_path))
        saver = tf.train.Saver(tf.global_variables())
        step = int(load_path.split('-')[-1]) + 1

    while step < n_epochs + 1:
        # Gather Training Data
        print('Epoch', step)
        s, a, r, d_r, n = rollout(batch_size)
        mean_reward = np.sum(r) / n
        average.append(mean_reward)
        print('Training Episodes: {}  Average Reward: {:4.2f}  Total Average: {:4.2f}'.format(n, mean_reward,
                                                                                              np.mean(average)))

        # Update Network
        sess.run(train_op, feed_dict={X: s.reshape(len(s), 200, 200, 1), Y: a, D_R: d_r})

        # Save Model
        if step % 3 == 0:
            print("SAVED MODEL")
            saver.save(sess, save_path, global_step=step)

        step += 1

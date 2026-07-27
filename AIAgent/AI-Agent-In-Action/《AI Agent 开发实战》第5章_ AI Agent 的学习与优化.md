# 第5章: AI Agent 的学习与优化

在本章中，我们将深入探讨AI Agent的学习和优化技术。这些技术使Agent能够从经验中学习，不断改进其性能，并适应复杂的环境。我们将重点关注强化学习、进化算法、元学习以及多智能体学习等方法。

## 5.1 强化学习在 Agent 中的应用

强化学习是一种通过与环境交互来学习最优策略的方法。它在AI Agent开发中扮演着关键角色，特别是在需要序列决策的任务中。

### 5.1.1 Q-learning 算法实现

Q-learning是一种经典的无模型强化学习算法，它通过学习状态-动作值函数（Q函数）来优化Agent的策略。

关键概念：
1. Q值：表示在给定状态下采取特定动作的预期累积奖励
2. 探索与利用：平衡尝试新动作和利用已知好动作
3. 时序差分学习：使用当前估计来更新先前的估计

代码示例：Q-learning实现
```python
import numpy as np
import gym

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

# 创建环境和Agent
env = gym.make('FrozenLake-v1')
agent = QLearningAgent(env.observation_space.n, env.action_space.n)

# 训练循环
num_episodes = 10000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

    if episode % 1000 == 0:
        print(f"Episode {episode} completed")

# 测试训练好的Agent
num_test_episodes = 100
total_rewards = 0

for _ in range(num_test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    
    total_rewards += episode_reward

average_reward = total_rewards / num_test_episodes
print(f"Average reward over {num_test_episodes} episodes: {average_reward}")
```

### 5.1.2 深度 Q 网络 (DQN)

深度Q网络（DQN）是Q-learning的一个扩展，它使用深度神经网络来近似Q函数，使其能够处理高维状态空间。

关键改进：
1. 经验回放：存储和重用过去的经验
2. 目标网络：使用单独的网络来生成目标Q值，提高稳定性
3. 卷积神经网络：处理图像等高维输入

代码示例：使用PyTorch实现DQN
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 创建环境和Agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 训练循环
num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    if episode % 10 == 0:
        agent.update_target_model()
        print(f"Episode: {episode}/{num_episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")

# 测试训练好的Agent
num_test_episodes = 10
for episode in range(num_test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    
    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        time += 1
        
    print(f"Test Episode: {episode + 1}/{num_test_episodes}, Score: {time}")

env.close()
```

### 5.1.3 策略梯度方法

策略梯度方法直接优化策略，而不是通过值函数间接优化。这使得它们特别适合于连续动作空间或复杂策略。

关键概念：
1. 策略函数：直接从状态映射到动作概率分布
2. 目标函数：通常是期望累积奖励
3. 梯度上升：通过梯度上升来优化策略参数

代码示例：使用PyTorch实现REINFORCE算法（一种简单的策略梯度方法）
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class REINFORCE:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum([r * (self.gamma ** i) for i, r in enumerate(rewards[t:])])
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

# 创建环境和Agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = REINFORCE(state_size, action_size)

# 训练循环
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    
    while True:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state
        
        if done:
            agent.update(rewards, log_probs)
            episode_reward = sum(rewards)
            print(f"Episode {episode}, Total Reward: {episode_reward}")
            break

# 测试训练好的Agent
num_test_episodes = 10
for episode in range(num_test_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        action, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    
    print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
```

这些强化学习方法为AI Agent提供了强大的学习能力，使其能够在复杂的环境中做出智能决策。在实际应用中，我们通常需要根据具体问题的特性选择合适的算法，并进行必要的调整和优化。

## 5.2 进化算法优化

进化算法是一类受生物进化启发的优化方法，它们通过模拟自然选择和遗传过程来搜索最优解。在AI Agent开发中，进化算法可以用于优化Agent的行为策略或神经网络结构。

### 5.2.1 遗传算法在 Agent 行为优化中的应用

遗传算法通过模拟生物进化过程来优化解决方案。在AI Agent中，我们可以使用遗传算法来优化Agent的行为策略。

关键步骤：
1. 编码：将Agent的策略参数编码为"基因"
2. 评估：根据Agent的性能评估每个个体的适应度
3. 选择：选择表现较好的个体进行繁殖
4. 交叉和变异：生成新的个体
5. 重复步骤2-4，直到达到终止条件

代码示例：使用遗传算法优化简单Agent的行为
```python
import numpy as np
import gym

class SimpleAgent:
    def __init__(self, weights):
        self.weights = weights

    def act(self, observation):
        return 1 if np.dot(observation, self.weights) > 0 else 0

def evaluate_agent(agent, env, n_episodes=100):
    total_reward = 0
    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def crossover(parent1, parent2):
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            individual[i] += np.random.normal(0, 0.1)
    return individual

def genetic_algorithm(env, population_size=100, generations=50):
    observation_space = env.observation_space.shape[0]
    population = [np.random.randn(observation_space) for _ in range(population_size)]

    for generation in range(generations):
        # 评估
        fitness_scores = [evaluate_agent(SimpleAgent(individual), env) for individual in population]

        # 选择
        parents = [population[i] for i in np.argsort(fitness_scores)[-10:]]

        # 生成新一代
        new_population = parents.copy()
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        best_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    best_individual = population[np.argmax(fitness_scores)]
    return SimpleAgent(best_individual)

# 创建环境和运行遗传算法
env = gym.make('CartPole-v1')
best_agent = genetic_algorithm(env)

# 测试最佳Agent
test_reward = evaluate_agent(best_agent, env, n_episodes=100)
print(f"Average Test Reward: {test_reward}")

# 可视化最佳Agent的表现
for _ in range(5):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = best_agent.act(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode Reward: {total_reward}")

env.close()
```

### 5.2.2 神经进化方法

神经进化是一种将进化算法应用于神经网络优化的方法。它可以用来优化网络的权重、结构，甚至是学习规则。

关键技术：
1. NEAT (NeuroEvolution of Augmenting Topologies)：同时优化网络结构和权重
2. HyperNEAT：使用间接编码来生成大规模神经网络
3. ES (Evolution Strategies)：一种可扩展的黑盒优化方法

代码示例：使用简化版的神经进化方法优化神经网络控制器
```python
import numpy as np
import gym
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class NeuroEvolutionAgent:
    def __init__(self, network):
        self.network = network

    def act(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation)
            action_probs = self.network(observation)
            action = torch.argmax(action_probs).item()
        return action

def evaluate_agent(agent, env, n_episodes=100):
    total_reward = 0
    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def create_offspring(parent1, parent2, mutation_rate=0.01):
    child = NeuralNetwork(parent1.fc1.in_features, parent1.fc1.out_features, parent1.fc2.out_features)
    
    # Crossover
    for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
        mask = torch.rand(child_param.data.size()) < 0.5
        child_param.data[mask] = parent1_param.data[mask]
        child_param.data[~mask] = parent2_param.data[~mask]
    
    # Mutation
    for param in child.parameters():
        if torch.rand(1) < mutation_rate:
            param.data += torch.randn(param.data.size()) * 0.1
    
    return child

def neuroevolution(env, population_size=100, generations=50):
    input_size = env.observation_space.shape[0]
    hidden_size = 16
    output_size = env.action_space.n

    population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]

    for generation in range(generations):
        # 评估
        fitness_scores = [evaluate_agent(NeuroEvolutionAgent(network), env) for network in population]

        # 选择
        parents = [population[i] for i in np.argsort(fitness_scores)[-10:]]

        # 生成新一代
        new_population = parents.copy()
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = create_offspring(parent1, parent2)
            new_population.append(child)

        population = new_population

        best_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    best_network = population[np.argmax(fitness_scores)]
    return NeuroEvolutionAgent(best_network)

# 创建环境和运行神经进化算法
env = gym.make('CartPole-v1')
best_agent = neuroevolution(env)

# 测试最佳Agent
test_reward = evaluate_agent(best_agent, env, n_episodes=100)
print(f"Average Test Reward: {test_reward}")

# 可视化最佳Agent的表现
for _ in range(5):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = best_agent.act(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Episode Reward: {total_reward}")

env.close()
```

这些进化算法为AI Agent的优化提供了另一种强大的方法。它们特别适用于难以定义明确目标函数的问题，或者在传统强化学习方法难以应用的场景中。

## 5.3 元学习与快速适应

元学习，也称为"学会学习"，是一种使AI Agent能够快速适应新任务或环境的技术。这对于需要在短时间内适应新情况的Agent特别有用。

### 5.3.1 少样本学习技术

少样本学习旨在使模型能够从极少量的样本中学习新任务。这在实际应用中非常重要，因为我们通常无法为每个新任务收集大量数据。

关键方法：
1. 原型网络 (Prototypical Networks)
2. 匹配网络 (Matching Networks)
3. 关系网络 (Relation Networks)

代码示例：使用原型网络进行少样本图像分类
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.datasets import Omniglot
from torch.utils.data import DataLoader, Subset

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

def euclidean_distance(x, y):
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)).pow(2), dim=2)

def train_prototypical(model, optimizer, train_loader, device, n_way, n_support, n_query, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 准备支持集和查询集
            x, _ = batch
            x = x.to(device)
            k = n_way * n_support
            x_support, x_query = x[:k], x[k:]

            # 计算原型
            z_support = model(x_support)
            z_support = z_support.reshape(n_way, n_support, -1).mean(dim=1)
            
            # 计算查询样本的嵌入
            z_query = model(x_query)
            
            # 计算距离和损失
            distances = euclidean_distance(z_query, z_support)
            log_p_y = F.log_softmax(-distances, dim=1)
            
            target_inds = torch.arange(0, n_way).to(device)
            target_inds = target_inds.repeat(n_query)
            
            loss = F.nll_loss(log_p_y, target_inds)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(log_p_y.data, 1)
            total_acc += (predicted == target_inds).sum().item() / n_query
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# 设置参数
n_way = 5
n_support = 5
n_query = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

train_dataset = Omniglot(root='./data', download=True, transform=transform, background=True)
train_loader = DataLoader(
    Subset(train_dataset, range(0, 10000)),  # 使用部分数据集以加快训练
    batch_size=n_way * (n_support + n_query),
    shuffle=True,
    num_workers=4
)

# 创建模型和优化器
model = PrototypicalNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_prototypical(model, optimizer, train_loader, device, n_way, n_support, n_query, epochs=10)

# 测试模型
model.eval()
test_dataset = Omniglot(root='./data', download=True, transform=transform, background=False)
test_loader = DataLoader(
    Subset(test_dataset, range(0, 2000)),  # 使用部分数据集进行测试
    batch_size=n_way * (n_support + n_query),
    shuffle=True,
    num_workers=4
)

total_acc = 0
with torch.no_grad():
    for batch in test_loader:
        x, _ = batch
        x = x.to(device)
        k = n_way * n_support
        x_support, x_query = x[:k], x[k:]

        z_support = model(x_support)
        z_support = z_support.reshape(n_way, n_support, -1).mean(dim=1)
        
        z_query = model(x_query)
        
        distances = euclidean_distance(z_query, z_support)
        log_p_y = F.log_softmax(-distances, dim=1)
        
        target_inds = torch.arange(0, n_way).to(device)
        target_inds = target_inds.repeat(n_query)
        
        _, predicted = torch.max(log_p_y.data, 1)
        total_acc += (predicted == target_inds).sum().item() / n_query

avg_acc = total_acc / len(test_loader)
print(f"Test Accuracy: {avg_acc:.4f}")
```

### 5.3.2 终身学习 Agent 设计

终身学习是指Agent能够持续学习新知识，同时保留旧知识的能力。这对于需要在动态环境中长期运行的Agent特别重要。

关键技术：
1. 弹性权重整合 (Elastic Weight Consolidation, EWC)
2. 渐进神经网络 (Progressive Neural Networks)
3. 记忆重放 (Memory Replay)

代码示例：使用弹性权重整合实现简单的终身学习Agent
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

class EWC(nn.Module):
    def __init__(self, model, fisher_estimation_sample_size):
        super(EWC, self).__init__()
        self.model = model
        self.fisher_estimation_sample_size = fisher_estimation_sample_size
        
    def estimate_fisher(self, data_loader, device):
        self.model.eval()
        fisher = {n: torch.zeros(p.shape).to(device) for n, p in self.model.named_parameters() if p.requires_grad}
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            self.model.zero_grad()
            output = self.model(input)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += p.grad.data.pow(2) / self.fisher_estimation_sample_size
        
        fisher = {n: p / self.fisher_estimation_sample_size for n, p in fisher.items()}
        return fisher
    
    def ewc_loss(self, cuda=False):
        if not hasattr(self, 'fisher'):
            return 0
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                _loss = (self.fisher[n] * (p - self.star_vars[n]).pow(2)).sum()
                loss += _loss
        return loss

class LifelongLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.ewc = EWC(self.model, fisher_estimation_sample_size=200)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.max(1)[1].item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + 0.99 * next_q_value * (1 - done)

        loss = F.mse_loss(q_value, expected_q_value.detach())
        ewc_loss = self.ewc.ewc_loss()
        total_loss = loss + 100 * ewc_loss  # EWC loss weight

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def learn_task(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
        
        # 更新 Fisher 信息和参数快照
        self.ewc.fisher = self.ewc.estimate_fisher(env, self.device)
        self.ewc.star_vars = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}

# 创建环境和Agent
env1 = gym.make('CartPole-v1')
env2 = gym.make('Acrobot-v1')
state_dim = max(env1.observation_space.shape[0], env2.observation_space.shape[0])
action_dim = max(env1.action_space.n, env2.action_space.n)
agent = LifelongLearningAgent(state_dim, action_dim)

# 学习第一个任务
print("Learning CartPole task...")
agent.learn_task(env1, num_episodes=200)

# 学习第二个任务
print("\nLearning Acrobot task...")
agent.learn_task(env2, num_episodes=200)

# 测试在两个任务上的表现
print("\nTesting on CartPole...")
env1 = gym.make('CartPole-v1')
for _ in range(10):
    state = env1.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env1.step(action)
        total_reward += reward
    print(f"CartPole Total Reward: {total_reward}")

print("\nTesting on Acrobot...")
env2 = gym.make('Acrobot-v1')
for _ in range(10):
    state = env2.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env2.step(action)
        total_reward += reward
    print(f"Acrobot Total Reward: {total_reward}")

env1.close()
env2.close()
```

## 5.4 多智能体强化学习

多智能体强化学习（MARL）涉及多个Agent在同一环境中学习和交互。这种设置可以模拟更复杂的现实世界场景，如交通系统、经济市场等。

### 5.4.1 多智能体马尔可夫决策过程

多智能体马尔可夫决策过程（MMDP）是单Agent MDP的扩展，用于描述多Agent环境。

关键概念：
1. 联合动作空间
2. 全局状态和局部观察
3. 奖励分配机制

### 5.4.2 分散式强化学习

分散式强化学习允许每个Agent独立学习其策略，而不需要访问全局信息。

关键方法：
1. 独立Q学习
2. 分散式演员-评论员（Decentralized Actor-Critic）

代码示例：使用独立Q学习的简单多Agent系统
```python
import numpy as np
import gym
import ma_gym

class IndependentQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

# 创建环境
env = gym.make('ma_gym:PredatorPrey5x5-v0')

# 创建Agent
n_agents = env.n_agents
agents = [IndependentQLearningAgent(env.observation_space[0].n, env.action_space[0].n) for _ in range(n_agents)]

# 训练循环
n_episodes = 10000
for episode in range(n_episodes):
    states = env.reset()
    total_reward = 0
    done = False

    while not done:
        actions = [agent.get_action(state) for agent, state in zip(agents, states)]
        next_states, rewards, dones, _ = env.step(actions)
        total_reward += sum(rewards)

        for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, next_states):
            agent.update(state, action, reward, next_state)

        states = next_states
        done = all(dones)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 测试训练好的Agents
n_test_episodes = 100
total_test_reward = 0

for _ in range(n_test_episodes):
    states = env.reset()
    episode_reward = 0
    done = False

    while not done:
        actions = [agent.get_action(state) for agent, state in zip(agents, states)]
        states, rewards, dones, _ = env.step(actions)
        episode_reward += sum(rewards)
        done = all(dones)

    total_test_reward += episode_reward

average_test_reward = total_test_reward / n_test_episodes
print(f"Average Test Reward: {average_test_reward}")

env.close()
```

### 5.4.3 协作探索策略

在多Agent环境中，协作探索可以帮助Agents更有效地学习。这涉及设计策略来鼓励Agents共同探索环境，而不是各自为政。

关键技术：
1. 内在激励机制
2. 通信协议
3. 共享经验池

代码示例：使用共享经验池的多Agent DQN
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import ma_gym
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SharedExperienceAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 创建环境
env = gym.make('ma_gym:PredatorPrey5x5-v0')

# 创建Agents
n_agents = env.n_agents
state_size = env.observation_space[0].shape[0]
action_size = env.action_space[0].n
agents = [SharedExperienceAgent(state_size, action_size) for _ in range(n_agents)]

# 训练循环
n_episodes = 1000
batch_size = 32

for episode in range(n_episodes):
    states = env.reset()
    total_reward = 0
    done = False

    while not done:
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        next_states, rewards, dones, _ = env.step(actions)
        total_reward += sum(rewards)

        for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states, dones):
            agent.remember(state, action, reward, next_state, done)

        states = next_states
        done = all(dones)

        # 所有Agent共享经验并学习
        if len(agents[0].memory) > batch_size:
            for agent in agents:
                agent.replay(batch_size)

    if episode % 10 == 0:
        for agent in agents:
            agent.update_target_model()
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 测试训练好的Agents
n_test_episodes = 100
total_test_reward = 0

for _ in range(n_test_episodes):
    states = env.reset()
    episode_reward = 0
    done = False

    while not done:
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        states, rewards, dones, _ = env.step(actions)
        episode_reward += sum(rewards)
        done = all(dones)

    total_test_reward += episode_reward

average_test_reward = total_test_reward / n_test_episodes
print(f"Average Test Reward: {average_test_reward}")

env.close()
```

这些多智能体强化学习方法为处理复杂的多Agent系统提供了强大的工具。在实际应用中，我们需要根据具体问题的特性选择合适的算法，并考虑Agent之间的协作和竞争关系。

通过本章，我们深入探讨了AI Agent的学习与优化技术，包括强化学习、进化算法、元学习和多智能体学习。这些方法为开发高性能、适应性强的AI Agent提供了丰富的工具和思路。在接下来的章节中，我们将探讨如何将这些技术应用到实际的AI Agent开发项目中。

import numpy as np
import matplotlib.pyplot as plt

# Define the number of agents and tasks
num_agents = 3
num_tasks = 10

# Define the difficulty level of each task
task_difficulties = np.random.randint(1, 11, size=num_tasks)

# Define the capability of each agent
agent_capabilities = np.random.randint(1, 6, size=num_agents)

# Helper function to calculate entropy
def entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-10))

# Implement LRP function
def layer_wise_relevance_propagation(q_values):
    # This is a simplified version of LRP
    # In a real scenario, this would be more complex and based on the network structure
    relevance_scores = np.abs(q_values) / (np.sum(np.abs(q_values)) + 1e-10)
    return relevance_scores

# Define the Fairness-Driven Explainable Learning algorithm
def fairness_driven_explainable_learning(num_iterations, alpha=0.1, gamma=0.9, epsilon=0.1, fairness_threshold=0.8, explainability_threshold=0.6):
    policy_params = np.random.rand(num_tasks, num_agents)
    fairness_scores = []
    explainability_scores = []
    reward_scores = [[] for _ in range(num_agents)]
    
    agent_fairness = [[] for _ in range(num_agents)]
    agent_explainability = [[] for _ in range(num_agents)]
    
    for iteration in range(num_iterations):
        task_allocations = [[] for _ in range(num_agents)]
        agent_rewards = [0] * num_agents
        
        # Compute relevance scores before task allocation
        relevance_scores = [layer_wise_relevance_propagation(params) for params in policy_params]
        
        for task_id in range(num_tasks):
            # Normalize policy parameters to avoid overflow in softmax
            policy_params[task_id] = (policy_params[task_id] - np.mean(policy_params[task_id])) / (np.std(policy_params[task_id]) + 1e-10)
            probs = np.exp(policy_params[task_id]) / np.sum(np.exp(policy_params[task_id]))
            agent_id = np.random.choice(range(num_agents), p=probs)
            
            task_allocations[agent_id].append(task_id)
            
            reward = 1 - (task_difficulties[task_id] - agent_capabilities[agent_id]) / 10
            reward = max(0, reward)
            agent_rewards[agent_id] += reward
        
        fairness_score = (np.sum(agent_rewards) ** 2) / (num_agents * np.sum(np.array(agent_rewards) ** 2))
        fairness_scores.append(fairness_score)
        
        explainability_score = 1 - np.mean([entropy(relevance) for relevance in relevance_scores]) / np.log2(num_agents)
        explainability_scores.append(explainability_score)
        
        for agent_id in range(num_agents):
            reward_scores[agent_id].append(agent_rewards[agent_id])
            agent_fairness[agent_id].append(agent_rewards[agent_id] / np.sum(agent_rewards))
            agent_explainability[agent_id].append(np.mean(relevance_scores[agent_id]))
        
        # Update policy parameters based on relevance and rewards
        for agent_id in range(num_agents):
            for task_id in task_allocations[agent_id]:
                policy_params[task_id][agent_id] += alpha * (agent_rewards[agent_id] * relevance_scores[task_id][agent_id])
        
        # Fairness adjustment
        if fairness_score < fairness_threshold:
            max_agent = np.argmax(agent_rewards)
            min_agent = np.argmin(agent_rewards)
            adjustment = (agent_rewards[max_agent] - agent_rewards[min_agent]) * alpha
            policy_params[:, max_agent] -= adjustment
            policy_params[:, min_agent] += adjustment
        
        # Explainability boost
        for task_id in range(num_tasks):
            top_agent = np.argmax(relevance_scores[task_id])
            policy_params[task_id] *= 0.5
            policy_params[task_id][top_agent] += 0.5
        
        # Periodically reset policy params to encourage exploration
        if iteration % 100 == 0:
            policy_params = 0.7 * policy_params + 0.3 * np.random.rand(num_tasks, num_agents)
        
        # Prune less relevant connections
        if iteration % 50 == 0:
            threshold = np.percentile(policy_params, 50)  # More aggressive pruning
            policy_params[policy_params < threshold] = 0
        
        # Normalize policy params
        policy_params = (policy_params - np.min(policy_params)) / (np.max(policy_params) - np.min(policy_params) + 1e-10)
    
    return fairness_scores, explainability_scores, reward_scores, agent_fairness, agent_explainability

# Define the base learning algorithm
def base_learning(num_iterations, alpha=0.1, gamma=0.9, epsilon=0.1):
    # Initialize policy parameters for each agent for each task
    policy_params = np.random.rand(num_tasks, num_agents)
    
    # Initialize fairness, explainability, and reward scores
    fairness_scores = []
    explainability_scores = []
    reward_scores = [[] for _ in range(num_agents)]
    
    agent_fairness = [[] for _ in range(num_agents)]
    agent_explainability = [[] for _ in range(num_agents)]
    
    for iteration in range(num_iterations):
        # Reset the task allocations and agent rewards
        task_allocations = [[] for _ in range(num_agents)]
        agent_rewards = [0] * num_agents
        
        # Agents take turns selecting tasks
        for task_id in range(num_tasks):
            # Normalize policy parameters to avoid overflow in softmax
            policy_params[task_id] = (policy_params[task_id] - np.mean(policy_params[task_id])) / (np.std(policy_params[task_id]) + 1e-10)
            probs = np.exp(policy_params[task_id]) / np.sum(np.exp(policy_params[task_id]))
            agent_id = np.random.choice(range(num_agents), p=probs)
            
            # Allocate the task to the agent
            task_allocations[agent_id].append(task_id)
            
            # Calculate the reward based on the agent's capability and task difficulty
            reward = 1 - (task_difficulties[task_id] - agent_capabilities[agent_id]) / 10
            reward = max(0, reward)
            agent_rewards[agent_id] += reward
        
        # Calculate fairness score
        fairness_score = (np.sum(agent_rewards) ** 2) / (num_agents * np.sum(np.array(agent_rewards) ** 2))
        fairness_scores.append(fairness_score)
        
        # Calculate explainability score using LRP
        relevance_scores = [layer_wise_relevance_propagation(params) for params in policy_params]
        explainability_score = 1 - np.mean([entropy(relevance) for relevance in relevance_scores]) / np.log2(num_agents)
        explainability_scores.append(explainability_score)
        
        for agent_id in range(num_agents):
            reward_scores[agent_id].append(agent_rewards[agent_id])
            agent_fairness[agent_id].append(agent_rewards[agent_id] / np.sum(agent_rewards))
            agent_explainability[agent_id].append(np.mean(relevance_scores[agent_id]))
        
        # Update policy parameters
        for agent_id in range(num_agents):
            for task_id in task_allocations[agent_id]:
                policy_params[task_id][agent_id] += alpha * agent_rewards[agent_id]
    
    return fairness_scores, explainability_scores, reward_scores, agent_fairness, agent_explainability

# Run the algorithms and plot results
# Run the Fairness-Driven Explainable Learning algorithm
num_iterations = 1000
fairness_scores_proposed, explainability_scores_proposed, reward_scores_proposed, agent_fairness_proposed, agent_explainability_proposed = fairness_driven_explainable_learning(num_iterations, alpha=0.1, gamma=0.9, epsilon=0.1, fairness_threshold=0.8, explainability_threshold=0.7)

# Run the base learning algorithm
fairness_scores_base, explainability_scores_base, reward_scores_base, agent_fairness_base, agent_explainability_base = base_learning(num_iterations, alpha=0.1, gamma=0.9, epsilon=0.1)

# Print average fairness, explainability, and reward for the system and each agent
print("Proposed Method:")
print(f"Average System Fairness: {np.mean(fairness_scores_proposed):.3f}")
print(f"Average System Explainability: {np.mean(explainability_scores_proposed):.3f}")
print(f"Average System Reward: {np.mean([np.mean(reward_scores_proposed[i]) for i in range(num_agents)]):.3f}")
for agent_id in range(num_agents):
    print(f"Agent {agent_id+1} Average Reward: {np.mean(reward_scores_proposed[agent_id]):.3f}")

print("\nBase Method:")
print(f"Average System Fairness: {np.mean(fairness_scores_base):.3f}")
print(f"Average System Explainability: {np.mean(explainability_scores_base):.3f}")
print(f"Average System Reward: {np.mean([np.mean(reward_scores_base[i]) for i in range(num_agents)]):.3f}")
for agent_id in range(num_agents):
    print(f"Agent {agent_id+1} Average Reward: {np.mean(reward_scores_base[agent_id]):.3f}")

# Smooth the reward scores using a moving average
window_size = 50
reward_scores_proposed_avg = [np.mean([reward_scores_proposed[i][j] for i in range(num_agents)]) for j in range(num_iterations)]
reward_scores_base_avg = [np.mean([reward_scores_base[i][j] for i in range(num_agents)]) for j in range(num_iterations)]

reward_scores_proposed_smooth = np.convolve(reward_scores_proposed_avg, np.ones(window_size)/window_size, mode='valid')
reward_scores_base_smooth = np.convolve(reward_scores_base_avg, np.ones(window_size)/window_size, mode='valid')

# Plotting the results
iterations = range(1, num_iterations + 1)
smooth_iterations = range(window_size, num_iterations + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(iterations, fairness_scores_proposed, marker='o', linestyle='-', label='Proposed Method')
plt.plot(iterations, fairness_scores_base, marker='x', linestyle='--', label='Base Method')
plt.xlabel('Iteration')
plt.ylabel('Fairness Score')
plt.title('Fairness Score')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(iterations, explainability_scores_proposed, marker='o', linestyle='-', label='Proposed Method')
plt.plot(iterations, explainability_scores_base, marker='x', linestyle='--', label='Base Method')
plt.xlabel('Iteration')
plt.ylabel('Explainability Score')
plt.title('Explainability Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(smooth_iterations, reward_scores_proposed_smooth, marker='o', linestyle='-', label='Proposed Method')
plt.plot(smooth_iterations, reward_scores_base_smooth, marker='x', linestyle='--', label='Base Method')
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
#plt.ylim(0, 3)
plt.title('Average Reward')
plt.legend()

plt.tight_layout()
plt.show()



# Plotting agents' reward, fairness, and explainability versus iterations in a separate figure
plt.figure(figsize=(18, 12))

for i in range(num_agents):
    plt.subplot(3, num_agents, i + 1)
    plt.plot(iterations, [reward_scores_proposed[i][j] for j in range(num_iterations)], marker='o', linestyle='-', label=f'Proposed Reward Agent {i+1}')
    plt.plot(iterations, [reward_scores_base[i][j] for j in range(num_iterations)], marker='x', linestyle='--', label=f'Base Reward Agent {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Agent {i+1} Reward')
    plt.title(f'Agent {i+1} Reward')
    plt.legend()

for i in range(num_agents):
    plt.subplot(3, num_agents, num_agents + i + 1)
    plt.plot(iterations, agent_fairness_proposed[i], marker='o', linestyle='-', label=f'Proposed Fairness Agent {i+1}')
    plt.plot(iterations, agent_fairness_base[i], marker='x', linestyle='--', label=f'Base Fairness Agent {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Agent {i+1} Fairness')
    plt.title(f'Agent {i+1} Fairness')
    plt.legend()

for i in range(num_agents):
    plt.subplot(3, num_agents, 2 * num_agents + i + 1)
    plt.plot(iterations, agent_explainability_proposed[i], marker='o', linestyle='-', label=f'Proposed Explainability Agent {i+1}')
    plt.plot(iterations, agent_explainability_base[i], marker='x', linestyle='--', label=f'Base Explainability Agent {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Agent {i+1} Explainability')
    plt.title(f'Agent {i+1} Explainability')
    plt.legend()

plt.tight_layout()
plt.show()

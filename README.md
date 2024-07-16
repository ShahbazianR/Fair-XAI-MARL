
# Fairness-Driven Explainable Learning

This repository contains the implementation of a Fairness-Driven Explainable Learning algorithm for task allocation among agents. The algorithm aims to ensure fair and explainable task allocation, improving overall system fairness and explainability compared to a base learning algorithm.

## Table of Contents
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Functions](#functions)
  - [entropy(probs)](#entropyprobs)
  - [layer_wise_relevance_(LRP)](#layer_wise_relevance_propagationq_values)
  - [proposed fairness_driven_explainable_learning](#fairness_driven_explainable_learningnum_iterations-alpha-gamma-epsilon-fairness_threshold-explainability_threshold)
  - [base_method](#base_learningnum_iterations-alpha-gamma-epsilon)
- [Results](#results)
- [Output](#example-output)
- [Visualization](#visualization)
- [License](#license)

## Dependencies

- Python 3.x
- numpy
- matplotlib

Install the required packages using pip:
\`\`\`bash
pip install numpy matplotlib
\`\`\`

## Code Structure

- \`num_agents\`: Number of agents.
- \`num_tasks\`: Number of tasks.
- \`task_difficulties\`: Array of task difficulty levels.
- \`agent_capabilities\`: Array of agent capabilities.
- \`entropy(probs)\`: Function to calculate the entropy of a probability distribution.
- \`layer_wise_relevance_propagation(q_values)\`: Simplified version of Layer-Wise Relevance Propagation (LRP) to calculate relevance scores.
- \`fairness_driven_explainable_learning(num_iterations, alpha, gamma, epsilon, fairness_threshold, explainability_threshold)\`: Implementation of the proposed Fairness-Driven Explainable Learning algorithm.
- \`base_learning(num_iterations, alpha, gamma, epsilon)\`: Implementation of the base learning algorithm.
- Main script to run the algorithms and plot results.

## Usage

To run the algorithms and visualize the results, simply execute the script:
\`\`\`bash
python script_name.py
\`\`\`

## Parameters

- \`num_iterations\`: Number of iterations for the learning process.
- \`alpha\`: Learning rate.
- \`gamma\`: Discount factor for future rewards.
- \`epsilon\`: Exploration rate for policy update.
- \`fairness_threshold\`: Threshold for fairness adjustment in the proposed method.
- \`explainability_threshold\`: Threshold for explainability boost in the proposed method.

## Outputs

The script outputs the average fairness, explainability, and reward scores for the proposed and base methods. It also plots the following:
1. Fairness Score vs. Iterations
2. Explainability Score vs. Iterations
3. Average Reward vs. Iterations (smoothed)
4. Agent-specific reward, fairness, and explainability vs. Iterations

## Functions

### entropy(probs)

Calculates the entropy of a probability distribution.

- **Input:** \`probs\` - Array of probabilities.
- **Output:** Entropy value.

### layer_wise_relevance_propagation(q_values)

Calculates the relevance scores using a simplified version of Layer-Wise Relevance Propagation (LRP).

- **Input:** \`q_values\` - Array of Q-values.
- **Output:** Array of relevance scores.

### fairness_driven_explainable_learning(num_iterations, alpha, gamma, epsilon, fairness_threshold, explainability_threshold)

Implements the Fairness-Driven Explainable Learning algorithm.

- **Inputs:**
  - \`num_iterations\` - Number of iterations.
  - \`alpha\` - Learning rate.
  - \`gamma\` - Discount factor.
  - \`epsilon\` - Exploration rate.
  - \`fairness_threshold\` - Fairness threshold for adjustment.
  - \`explainability_threshold\` - Explainability threshold for boosting.
- **Outputs:**
  - \`fairness_scores\` - List of fairness scores over iterations.
  - \`explainability_scores\` - List of explainability scores over iterations.
  - \`reward_scores\` - List of reward scores over iterations for each agent.
  - \`agent_fairness\` - List of fairness scores for each agent over iterations.
  - \`agent_explainability\` - List of explainability scores for each agent over iterations.

### base_learning(num_iterations, alpha, gamma, epsilon)

Implements the base learning algorithm.

- **Inputs:**
  - \`num_iterations\` - Number of iterations.
  - \`alpha\` - Learning rate.
  - \`gamma\` - Discount factor.
  - \`epsilon\` - Exploration rate.

## Results

After running the script, the results will be printed and plotted, showing the performance of the proposed method compared to the base method in terms of fairness, explainability, and reward.

##  Output (for 3 nodes and 10 tasks)
\`\`\`plaintext
Proposed Method:
Average System Fairness: 0.749
Average System Explainability: 0.440
Average System Reward: 2.115
Agent 1 Average Reward: 1.247
Agent 2 Average Reward: 1.402
Agent 3 Average Reward: 3.695

Base Method:
Average System Fairness: 0.538
Average System Explainability: 0.067
Average System Reward: 2.376
Agent 1 Average Reward: 0.913
Agent 2 Average Reward: 0.662
Agent 3 Average Reward: 5.553
\`\`\`
The figures for this simulation is also given. In the paper we publish the results for 10 nodes and 100 tasks. 
![pic1-git-3n](https://github.com/user-attachments/assets/24a0bbf6-8961-4f02-b77e-058bb500422d)
![git-rewards](https://github.com/user-attachments/assets/3b029204-1d35-4c67-a91b-6a4fadbddd63)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation
@inproceedings{mahmood2024fairness,
  title={Fairness-Driven Explainable Learning in Multi-Agent Reinforcement Learning},
  author={Tariq Mahmood and Reza Shahbazian and Irina Trubitsyna},
  booktitle={The 1st International Workshop on “Adjustable Autonomy and Physical Embodied Intelligence” - European Conference on Artificial Intelligence (ECAI)},
  year={2024},
  address={Santiago de Compostela, Spain},
  note={Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).}
}



# Tutorials

### Table of Contents
<ul>
<li><a href="#setup">Setup</a><br>
<li><a href="#t1">Tutorial 1</a>
<li><a href="#t2">Tutorial 2</a>
<li><a href="#t3">Tutorial 3</a>
</ul>

<h2 id="#setup">Setup</h2>
<ol>
<li> [OPTIONAL] Install Anaconda. Download from <a href="https://www.anaconda.com/">anaconda.com</a>. Try watch this <a href="https://www.youtube.com/watch?v=YJC6ldI3hWk">tutorial</a> if you are getting issues in installation.
<li>[OPTIONAL] After successful installation of anaconda, create a new anaconda environment. In a terminal/command prompt enter

```shell
conda create --name pansim python=3.8
conda activate pansim
```

You can replace pansim with a name of your choice. <b>Continue using this terminal/command prompt window for the following steps.</b>

<li>Clone the repository this is done by entering the following in a terminal/command prompt window.

With HTTPS: 
```shell
git clone https://github.com/cs395t-ethical-ai/PandemicSimulatorTutorial.git
```
With SSH
```shell
git clone git@github.com:cs395t-ethical-ai/PandemicSimulatorTutorial.git
```

<li>Continue in the same terminal/command prompt. Change current directory and run setup.py

```shell
cd PandemicSimulatorTutorial
python -m pip install -e .
```
</ol>

Congratulations you finished the setup. Time to start the first Tutorial.

<h2 id="#t1">Tutorial 1</h2>

Welcome to the first tutorial.<br>
After this tutorial, you should get an idea of what this repository is about. You will manually control the stage of response to a simulated pandemic. After that by studying the observations try to create a response policy using if-else statements. 

<ol>
<li><b>Skim the <a href="https://arxiv.org/abs/2010.10560">Original Paper</a>.</b> <br>This will help you understand the background of this problem and the environment.
<li><b>Run scripts/tutorials/7_run_pandemic_gym_env.py</b>. <br>Do this to understand how to use the environment. 
<li><b>Run scripts/tutorials/8_manual_control.py</b> <br> This allows you to manually set the stage at each point. Use this to understand how the stages affect the case dynamics.
<li><b>Run scripts/tutorials/9_example_policy_if_else.py</b> <br> This is an example policy implemented using an if-else statement based on the observation.
<li><b>Read <a href="#xobs">Explaining the Observation and Action</a></b><br> (below). This will help you understand the observation and make a custom policy for the next part of the tutorial.
<li><b>Run scripts/tutorials/10_custom_policy_if_else.py </b><br> This is another example of a policy with an if-else statement. Try to replace the statements and create your own policy.
<li><b>Run scripts/tutorials/11_custom_policy_test.py </b><br> Take your policy from scripts/tutorials/custom_policy_if_else.py and run this script to evaluate your policy over multiple episodes.

<br><br>
<h3 id="#xobs">Explaining the Observation and Action</h3>

<b>Observation Table</b>

---
|Observation|Explanation|How to Access|
| ----------- | -----------| ----------- |
|Critical Population (Testing)|Number of people in Critical condition according to current testing policy|obs.global_testing_summary[...,0]|
|Dead Population (Testing)|Number of people in dead according to current testing policy|obs.global_testing_summary[...,1]|
|Infected Population (Testing)|Number of people in Infected condition according to current testing policy|obs.global_testing_summary[...,2]|
|None Population (Testing)|Number of people in not in any other condition according to current testing policy|obs.global_testing_summary[...,3]|
|Recovered Population (Testing)|Number of people in Recovered condition according to current testing policy|obs.global_testing_summary[...,4]|
|Critical Population (Actual)| Actual number of people in Critical condition|obs.global_infection_summary[...,0]|
|Dead Population (Actual)|Actual number of people in dead |obs.global_infection_summary[...,1]|
|Infected Population (Actual)|Actual number of people in Infected condition |obs.global_infection_summary[...,2]|
|None Population (Actual)|Actual number of people in not in any other condition |obs.global_infection_summary[...,3]|
|Recovered Population (Actual)|Actual number of people in Recovered condition |obs.global_infection_summary[...,4]|
|Current Stage|Stage of Response at Current timestep|obs.stage[...,0]|
|Infection Flag|Whether Number of Infected People according to current testing policy exceeds threshold specified according to current simulator configuration (10 by default)|obs.infection_above_threshold[...,0]|
|Current Day|Current Day of simulation|obs.time_day[...,0]|
|Current Unlocked Non Essential Business Locations|List of ids of businesses that are Unlocked (By default additional businesses are not used in the simulator, so this is unused) |obs.unlocked_non_essential_business_locations[...,0]|
---

<b>Action  Table</b>

---
|Stages|Stay home if sick, Practice good hygiene| Wear facial coverings| Social distancing| Avoid gathering size (Risk: number)|Locked locations|
| ----------- | -----------| ----------- | ----------- | -----------| ----------- |
|Stage 0| False|False|None|None|None|
|Stage 1| True|False|None|Low: 50, High: 25|None|
|Stage 2| True|True|0.3|Low: 25, High: 10|School, Hair Salon|
|Stage 3| True|True|0.5|Low: 0, High: 0 |School, Hair Salon|
|Stage 4| True|True|0.7|Low: 0, High: 0 |School, Hair Salon, Office, Retail Store|

---

<br><br>

</ol>

<h2 id="#t2">Tutorial 2</h2>

In this tutorial we will work on modifying the observation. An observation consists of the variables your policy is allowed to use to make decisions. Let's consider the pandemic response problem: here, we are trying to design a policy that would provide an optimal decision/resposne to the current state of the pandemic. In real life, the authorities who provide such a response may not have complete knowledge of the state of the environment---for instance, an accurate count of how many people truly have the disease, rather than the proportion of people who tested positive. If we could give this crucial information to the authorities, their decision-making could improve dramatically.

This tutorial shows how to augment the observation with additional variables, so that we may create policies that explicitly consider this additional information. Specifically, we will create a flag indicating whether the current population has more critical cases than a certain threshold and add it to the observation.

ALL LINE NUMBERS IN THE FOLLOWING REFER TO THE INITIAL STATE OF THE CODE. 
PLEASE NOTE LINE NUMBER VALUES MAY CHANGE IF ADDITIONAL CHARACTERS OR LINES ARE ADDED WHILE CODING. 


Steps
<ol>
<li><b>Switch to the Tutorial 2 branch via the following lines (run from the top level of the repository)</a></b>
    
```shell
git fetch --all
git checkout tut2
```

<li><b>Review <a href="#xobs">Explaining the Observation and Action</a></b>
<li><b> Add Critical FLag </b><br>
To modify the observation we can change the code at two levels. We can either change the structure of the simulator state or we can modify the final observation. For this tutorial we will modify the simulator state. The code for simulator state is in python/pandemic_simulator/environment/interfaces/sim_state.py

For demonstration, we will be creating a new flag variable for the environment. This flag will check if the number of patients who are critical exceeds a threshold. The name of the variable should be `critical_above_threshold`. This variable is most similar to the existing `infection_above_threshold` variable. 

<ol>
<li><b>Modify simulator state template to add new flag.</b><br>
 Open python/pandemic_simulator/environment/interfaces/sim_state.py
(Line 52) Add code for the new flag in the specified area. This change will allow the sim_state to hold the new flag. 

<li><b>Modify simulator config to add the threshold, as it is a characteristic of the simulator</b><br>
 Open python/pandemic_simulator/environment/simulator_opts.py 
(Line 48) Add code for a threshold to use for the flag and assign a default value of 10. The threshold is not a variable in the PandemicSimulator class
as it is constant throughout the simulation and is a characteristic of the simulator.

<li><b>Write code to incorporate the flag in the simulator</b><br>
Open python/pandemic_simulator/environment/pandemic_sim.py 
(Line 63,93,121) Here, we will have to modify the init function to add and initialize the new threshold variable. This is to enable users to add a value for our new flag when initializing the 
Pandemic Simulator State.

(Line 168) Change from_config function to pass in the new threshold variable. 

(Line 325) Next we will have to modify the step function in pandemic_sim.py. The step function updates the simulator state after each timestep (hour); please update the state to reflect the `critical_above_threshold` value. 

(Line 416) Then set a default value for the `critical_above_threshold` flag in the reset function.

<li><b>Modify the observation for flag</b>
Open python/pandemic_simulator/environment/interfaces/pandemic_observation.py
(Line 25) Add code to initialize flag in observation.
(Line 44) Add code to attribute for flag in function.
(Line 78) Add code to update flag in observation.
</ol>

<li> [OPTIONAL] The simulator is installed as a Python package. Since we have modified the underlying package code, 
    we need to reinstall the package in order for the changes to the code to be realized. From the top level of the PandemicSimulatorTutorial repository, run the following line again: <br>

```shell
python -m pip install -e .
```
</ol>

<h2 id="#t3">Tutorial 3</h2>


In this tutorial, we will walk you through a basic example on how to use deep reinforcement learning to learn a policy for the Pandemic Simulator problem. More specifically, you will install some tools for deep reinforcement learning (RL), run some baseline policies, train a policy via the DQN algorithm, and learn to view Tensorboard logs. 

Please start this tutorial at least 1 day in advance of the deadline, as training the deep RL policy will take around 8-12 hours depending on your machine.

 1. Setup steps: 
    - Be sure to activate your conda environment!
    - Switch to the `tut3` branch using the below command. You may need to discard your changes from `tut2`, or `git stash` them.
	```
	git fetch --all
	git checkout tut3
	```
    - Since some new features have been added to the code inside the `python/pandemic_simulator` directory, re-install install the package from the top level of the PandemicSimulatorTutorial repository via the following command:
	```shell
	python3 -m pip install -e .
	```
    - Next, install the package `tianshou`. [Tianshou](https://github.com/thu-ml/tianshou) is a package that provides modular implementations of various well-known deep RL algorithms in PyTorch.  The basic architecture of the Tianshou codebase is described [here](https://tianshou.readthedocs.io/en/latest/tutorials/concepts.html).  
    ```shell
    pip install tianshou
    ```   
    - Installing Tianshou should also automatically install PyTorch and Tensorboard as dependencies. Check that this is the case by running the following commands. If this is not the case, please install these packages on your own.
    ```shell
    pip show torch
    pip show tensorboard
    ```
    - **Note regarding CPU vs GPU versions of PyTorch: the bottleneck of learning on the PandemicSim is the simulator speed, which relies on the CPU, so using the GPU for learning may not speed things up.
2. In Tutorials 1 and 2, you interacted with the `PandemicGymEnv`, which is defined in the file, `python/pandemic_simulator/environment/pandemic_env.py`. To allow the deep RL algorithms work with the PandemicSimulator, we have provided the `PandemicGymEnv3Act`, which wraps the original `PandemicGymEnv`. Please open this file and observe the modifications described. The important differences (and non-differences!) between the two environments are listed below using the vocabulary of RL:
    - *Observation space*: 
	    - The original environment used the `PandemicObservation` data class to wrap the observation. The new environment flattens the information contained within this data clas into a vector.
	    -  The new environment defines an `observation_space` attribute, to specify the size of the observation vector. 
	    - The new environment also removes two types of data from the observation to reduce the state space of the problem. First, `global_testing_summary` has been removed because we already include the true infection state in the observation as `global_infection_summary`, and `global_testing_summary` is a noisy version of the true infection state. This is to create an easier learning problem for tutorial purposes. Note that the original PandemicSim paper gave the policy network the `global_testing_summary` , and only gave the `global_infection_summary` as input to the critic network --- a more realistic approach. Second,  `unlocked_non_essential_business_locations` has been removed, because it was an unused variable. 
	    - Finally, observation data is now normalized to lie between 0 and 1. 
    - *Action space*: in the original environment, the actions that a policy could take were the stage numbers, from 0 to 4. For consistency with the way learning was implemented in the original PandemicSim paper, in the new environment, the actions are now {-1, 0, 1}. The action of -1 means "decrease the stage", 0 means "keep the stage the same", and 1 means "increase the stage". This is a *design choice* that you may wish to experiment with in your final projects.
    - *Transition function*: No change
    - *Reward function*: No change
    - *Done function*: in both the original and new environments, a done function (`done_fn`) must be passed to the environment upon initialization, to determine when each episode terminates. In the new environment, the episode will terminate after 120 steps (days), although it may terminate earlier due to the done function.
    
3. Tutorial script `12_eval_baseline_policies.py` evaluates some simple baseline policies on the new environment to give you some more intuition about the PandemicSim. It also shows how to initialize the environment with reward and done functions---which has not been shown in previous tutorials. 
    - Open tutorial script `12_eval_baseline_policies.py` and read the descriptions of the baseline policies under the `if __name__=='__main__':` statement. Which policy would you expect to do the best? The worst?
    - Run tutorial script 12 via the below command. **Please screenshot the terminal output and include in your submission.**
	 ```shell
	 python 12_eval_baseline_policies.py
	 ```
	 
4. In tutorial script `13_train_dqn.py`, we will learn a policy via the deep reinforcement learning algorithm, DQN. DQN is one of the first examples of a successful deep reinforcement learning algorithm. [OPTIONAL] Skim the original [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).
    - Please open the tutorial script, and observe the code structure. In general, most modifiable parameters have been placed in the `get_args()` function.
    - Run the RL training script, `13_train_dqn.py` (this could take anywhere from ~8-12 hours depending on your machine configurations). 
    - After an hour or two of training, you should confirm that logging files are being saved in the newly generated `log` folder. When you observe Tensorboard events files, you may proceed to the last step. 

5. In this step, you will learn how to use Tensorboard to view the log files. 
    - Enter the directory, `scripts/tutorials/log` in the terminal.
    - Run the following command: 
    ```shell
    tensorboard --logdir . --port=6007
    ```
    - Open your favorite browser, and go to the link, `localhost:6007` (do not prefix this link with anything such as `www`, `https://`, etc.). 
    - You should see a orange dashboard with auto-generated graphs of the policy's learning curves. the learning losses, etc, similar to the image in the tutorial [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). If you do not see line graphs, make sure to select `Scalars`  at the top of the dashboard. **Once training has completed, please screenshot this dashboard for your submission.** 
    
[Disclaimer] The original PandemicSim paper used the SAC algorithm---a state-of-the-art deep RL algorithm--- to learn a policy, while here, we use DQN, a classic algorithm that is relatively simpler to understand. The learning script provided in this tutorial does not represent the best possible learning setup, and many optimizations could be made to improve the learning results. 

For example, one trick used by the authors of the PandemicSim paper is the following: at the beginning of each episode, the simulator is allowed to run until the number of infections hit a value of 5. This is called a "warmup phase" in the original paper. The data collected during this warmup phase is ignored by the learning process, by not placing this data into the experience replay buffer. Afterwards, they set the initial stage to 4 (lockdown), and allow the RL policy decide the stage thereafter until the episode is done.

You may wish to try out some tricks of your own, or even different learning algorithms, for your final project. 


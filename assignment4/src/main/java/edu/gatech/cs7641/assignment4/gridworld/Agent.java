package edu.gatech.cs7641.assignment4.gridworld;

import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import static edu.gatech.cs7641.assignment4.gridworld.Config.*;


class Agent {

    static final String VALUE_ITERATION = "Value Iteration";
    static final String POLICY_ITERATION = "Policy Iteration";
    static final String Q_LEARNER = "Q-Learning";

    // constants that all planning algorithms will utilize
    private static final SimpleHashableStateFactory HASH_FACTORY = new SimpleHashableStateFactory();

    /**
     * Main planner building method which decides which planner is being requested then calls
     * the appropriate helper method
     * @param domain - SADomain the planner will process
     * @return Planner
     */
    static Planner buildAgent(SADomain domain, int maxIters) {

        switch (AGENT_NAME) {
            case VALUE_ITERATION:
                return buildValueIteration(domain, maxIters);

            case POLICY_ITERATION:
                return buildPolicyIteration(domain, maxIters);

            case Q_LEARNER:
                return buildQLearner(domain, maxIters);

            default:
                throw new IllegalArgumentException("Invalid planner requested: " + AGENT_NAME);
        }
    }

    /**
     * Builds a value iteration planner
     * @param domain - SADomain to process
     * @param maxIters - int of max iterations
     * @return ValueIteration Planner
     */
    private static Planner buildValueIteration(SADomain domain, int maxIters) {
        return new ValueIteration(domain, GAMMA, HASH_FACTORY, MAX_DELTA, maxIters);
    }

    /**
     * Builds a policy iteration planner
     * @param domain - SADomain to process
     * @param maxIters - int of max iterations
     * @return PolicyIteration Planner
     */
    private static Planner buildPolicyIteration(SADomain domain, int maxIters) {
        return new PolicyIteration(domain, GAMMA, HASH_FACTORY, MAX_DELTA, maxIters, maxIters);
    }

    /**
     * Builds a Q Learning agent
     * @param domain - SADomain to process
     * @param maxIters - int of max iterations
     * @return QLearning Planner
     */
    static QLearning buildQLearner(SADomain domain, int maxIters) {
        QLearning agent = new QLearning(domain, GAMMA, HASH_FACTORY, INIT_Q_VAL, ALPHA, maxIters);
        agent.initializeForPlanning(1);
        return agent;
    }
}
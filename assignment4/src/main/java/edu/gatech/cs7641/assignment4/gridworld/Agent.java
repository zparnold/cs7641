package edu.gatech.cs7641.assignment4.gridworld;

import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import static edu.gatech.cs7641.assignment4.gridworld.Config.*;
import static edu.gatech.cs7641.assignment4.gridworld.Domain.MOUNTAIN_CAR;


class Agent {

    static final String VALUE_ITERATION = "Value Iteration";
    static final String POLICY_ITERATION = "Policy Iteration";
    static final String Q_LEARNER = "Q-Learning";

    // constants that all planning algorithms will utilize
    private static final SimpleHashableStateFactory HASH_FACTORY = new SimpleHashableStateFactory();

    /**
     * Main planner building method which decides which planner is being requested then calls
     * the appropriate helper method
     *
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

    static Planner buildAgent(Domain domain) {
        switch (AGENT_NAME) {
            case POLICY_ITERATION:
                return buildPolicyIteration(domain);

            default:
                throw new IllegalArgumentException("Invalid planner requested: " + AGENT_NAME);
        }
    }

    /**
     * Builds a value iteration planner
     *
     * @param domain   - SADomain to process
     * @param maxIters - int of max iterations
     * @return ValueIteration Planner
     */
    private static Planner buildValueIteration(SADomain domain, int maxIters) {
        return new ValueIteration(domain, GAMMA, HASH_FACTORY, MAX_DELTA, maxIters);
    }

    /**
     * Builds a policy iteration planner
     *
     * @param domain   - SADomain to process
     * @param maxIters - int of max iterations
     * @return PolicyIteration Planner
     */
    private static Planner buildPolicyIteration(SADomain domain, int maxIters) {
        return new PolicyIteration(domain, GAMMA, HASH_FACTORY, MAX_DELTA, maxIters, maxIters);

    }

    private static Planner buildPolicyIteration(Domain domain){
        StateGenerator rStateGen = new MCRandomStateGenerator(domain.getMountainCar().physParams);
        SARSCollector collector = new SARSCollector.UniformRandomSARSCollector(domain.getSADomain());
        SARSData dataset = collector.collectNInstances(rStateGen, domain.getSADomain().getModel(), 5000, 20, null);

        NormalizedVariableFeatures features = new NormalizedVariableFeatures()
                .variableDomain("x", new VariableDomain(domain.getMountainCar().physParams.xmin, domain.getMountainCar().physParams.xmax))
                .variableDomain("v", new VariableDomain(domain.getMountainCar().physParams.vmin, domain.getMountainCar().physParams.vmax));
        FourierBasis fb = new FourierBasis(features, 4);
        return new LSPI(domain.getSADomain(), GAMMA, new DenseCrossProductFeatures(fb, 3), dataset);
    }

    private static Planner buildValueIteration(Domain domain){
        SarsaLam s = new SarsaLam(domain.getSADomain(), GAMMA, HASH_FACTORY, INIT_Q_VAL, ALPHA, 30, 0.1);
        s.initializeForPlanning(1);
        return s;
    }

    /**
     * Builds a Q Learning agent
     *
     * @param domain   - SADomain to process
     * @param maxIters - int of max iterations
     * @return QLearning Planner
     */
    static QLearning buildQLearner(SADomain domain, int maxIters) {
        QLearning agent = new QLearning(domain, GAMMA, HASH_FACTORY, INIT_Q_VAL, ALPHA, maxIters);
        agent.initializeForPlanning(1);
        return agent;
    }
}
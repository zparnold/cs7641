package edu.gatech.cs7641.assignment4.gridworld;


class Config {

    /*
    Contains all configurations utilized within this project.  Any changes necessary to which
    agent to train, which domain to build, the particular parameters for both of those items
    as well as any other specifics will be stored in this class as constants
     */

    // Main settings
    static final int NUM_TRIALS = 1000;
    static final String DOMAIN_NAME = Domain.GRID_WORLD_LARGE;
    static final String AGENT_NAME = Agent.Q_LEARNER;
    static final boolean VISUALIZE = true;
    static final boolean PLOT = true;
    static final boolean Q_LEARN_PLOT = true;

    // Agent settings
    static final double GAMMA = 0.999;
    static final double MAX_DELTA = 0.0001;
    static final double INIT_Q_VAL = 0.5;
    static final double ALPHA = 0.3;

    // Domain settings
    static final double SUCCESS_RATE = 0.8;
    static final double GOAL_REWARD = 5.0;
    static final double MOVE_REWARD = -0.1;
}
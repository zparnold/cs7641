package edu.gatech.cs7641.assignment4.gridworld;

import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.Visualizer;

import java.util.List;
import java.util.Random;


class Domain {

    static final String GRID_WORLD_SMALL = "Small Grid Domain";
    static final String GRID_WORLD_LARGE = "Large Grid Domain";
    static final String MOUNTAIN_CAR = "Mountain Car";

    public MountainCar getMountainCar() {
        return mountainCar;
    }

    private MountainCar mountainCar;
    private GridWorldDomain gridWorld;
    private SADomain oosaDomain;
    private State initState;
    private int maxSteps;

    Domain(String domainName) throws InstantiationException {

        switch (domainName) {
            case GRID_WORLD_SMALL:
                this.oosaDomain = createGWSmall();
                break;

            case GRID_WORLD_LARGE:
                this.oosaDomain = createGWLarge();
                break;

            case MOUNTAIN_CAR:
                this.oosaDomain = createMountainCar();
                break;

            default:
                throw new InstantiationException("Invalid domain selected: " + domainName);
        }

    }

    /**
     * @return OOSADomain - the domain that was generated on construction
     */
    SADomain getSADomain() {
        return this.oosaDomain;
    }

    /**
     * Creates a visualizer for the domain, good for testing and taking a screen shot of what
     * it looks like in real time
     */
    void visualizeDomain() {
        if (this.gridWorld != null){
            // create visualizer and explorer
            Visualizer v = GridWorldVisualizer.getVisualizer(this.gridWorld.getMap());
            VisualExplorer exp = new VisualExplorer(this.oosaDomain, v, this.initState);

            // set control keys to use w-s-a-d
            exp.addKeyAction("w", GridWorldDomain.ACTION_NORTH, "");
            exp.addKeyAction("s", GridWorldDomain.ACTION_SOUTH, "");
            exp.addKeyAction("a", GridWorldDomain.ACTION_WEST, "");
            exp.addKeyAction("d", GridWorldDomain.ACTION_EAST, "");

            exp.initGUI();
        }

        if (this.mountainCar != null) {
            Visualizer v = MountainCarVisualizer.getVisualizer(mountainCar);
            VisualActionObserver vob = new VisualActionObserver(v);
            vob.initGUI();
        }


    }

    /**
     * Creates a value function visualization for the given agent which showcases the route
     * planning the agent discovered
     * @param agent - Planner agent to simulate
     */
    void visualizeValueFunction(Planner agent) {
        if (gridWorld != null){
            List<State> allStates = StateReachability.getReachableStates(
                    this.initState,
                    this.oosaDomain,
                    agent.getHashingFactory()
            );

            ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization (
                    allStates,
                    this.gridWorld.getWidth(),
                    this.gridWorld.getHeight(),
                    (ValueFunction) agent,
                    agent.planFromState(this.initState())
            );

            gui.initGUI();
        }
    }

    /**
     * @return State - initial and final state for the domain
     */
    State initState() {
        return this.initState;
    }

    int maxSteps() {
        return this.maxSteps;
    }

    /**
     * Helper that builds the 11 x 11 gridworld state that is default for BURLAP.  Uses the default
     * 4 room setup with two entry/exit points in each room.  The agent starts in one room diagonal
     * from the end target.  Tweak rewards and successes here.
     * @return OOSADomain
     */
    private SADomain createGWSmall() {
        int mapSize = 101;

        // create 11x11 grid world using Burlap's default 4 room setup and set the success rate
        // of each move to the value determined
        this.gridWorld = new GridWorldDomain(mapSize, mapSize);
        this.gridWorld.setMapToFourRooms();
        this.gridWorld.setProbSucceedTransitionDynamics(Config.SUCCESS_RATE);

        GridWorldTerminalFunction gwtf = new GridWorldTerminalFunction(
                this.gridWorld.getWidth() - 1,
                this.gridWorld.getHeight() - 1
        );

        GridWorldRewardFunction gwrf = buildRewardFunction();

        this.gridWorld.setTf(gwtf);
        this.gridWorld.setRf(gwrf);

        // setup initial state, goal location, and generator for state
        this.initState = new GridWorldState(
                new GridAgent(0, 0),
                new GridLocation(mapSize - 1, mapSize - 1, "X")
        );
        this.maxSteps = 2000;

        // finally generate the domain to use
        return this.gridWorld.generateDomain();
    }

    /**
     * Builds a large grid world with more complex paths and hazards.  Grid world will model the
     * following example:
     *
     *      {X,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0},
     *      {0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0},
     *      {0,0,1,1,1,0,1,0,1,L,1,0,1,0,S,1,1,0,1,0,0},
     *      {0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0},
     *      {1,1,1,0,1,0,1,0,1,0,1,1,1,1,S,1,1,0,1,0,0},
     *      {0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
     *      {0,0,1,1,1,0,1,S,1,0,1,0,1,0,1,1,1,0,1,0,0},
     *      {0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0},
     *      {0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1},
     *      {0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0},
     *      {1,1,1,0,1,0,1,M,1,1,1,0,1,0,M,1,1,0,1,0,0},
     *      {0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0},
     *      {0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,S,0,0},
     *      {0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0},
     *      {1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1},
     *      {0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0},
     *      {0,0,1,0,1,0,1,1,1,0,1,L,0,0,1,0,1,1,1,0,0},
     *      {0,0,0,0,0,0,1,S,0,0,0,0,1,0,1,0,1,0,0,0,0},
     *      {0,0,1,0,1,1,1,1,0,1,1,0,1,0,1,0,1,0,1,0,0},
     *      {0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0},
     *      {0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,G}
     *
     * @return OOSADomain
     */
    private SADomain createGWLarge() {

      int[][]  map = new int[][] {
                {0,0,1,0,0,0,0,0,1,0,0},
                {0,0,1,0,0,0,0,0,1,0,0},
                {0,0,1,0,0,1,1,0,0,0,0},
                {0,0,0,0,0,0,1,0,1,0,0},
                {1,1,1,1,0,1,1,0,1,0,0},
                {0,0,1,0,0,0,0,0,0,0,0},
                {1,0,1,0,1,1,1,0,1,0,0},
                {1,0,0,0,1,0,0,0,1,0,0},
                {1,1,1,1,1,0,1,0,1,1,1},
                {0,0,1,0,0,0,1,0,0,0,0},
                {1,0,1,0,0,1,1,0,1,0,0}
        };

        this.gridWorld = new GridWorldDomain(map);
        this.gridWorld.setProbSucceedTransitionDynamics(Config.SUCCESS_RATE);

        // establish the domain terminating function
        GridWorldTerminalFunction gwtf = new GridWorldTerminalFunction(
                this.gridWorld.getWidth() - 1,
                this.gridWorld.getHeight() - 1
        );

        GridWorldRewardFunction gwrf = buildRewardFunction();

        gwrf.setReward(2, 9, -3.0 + Config.MOVE_REWARD);
        gwrf.setReward(2, 9, -1.0 + Config.MOVE_REWARD);
        gwrf.setReward(4, 9, -1.0 + Config.MOVE_REWARD);
        gwrf.setReward(6, 7, -1.0 + Config.MOVE_REWARD);
//        gwrf.setReward(10, 7, -2.0 + Config.MOVE_REWARD);
//        gwrf.setReward(10, 14, -2.0 + Config.MOVE_REWARD);
//        gwrf.setReward(12, 18, -1.0 + Config.MOVE_REWARD);
//        gwrf.setReward(16, 11, -3.0 + Config.MOVE_REWARD);
//        gwrf.setReward(17, 7, -1.0 + Config.MOVE_REWARD);

        this.gridWorld.setTf(gwtf);
        this.gridWorld.setRf(gwrf);

        // setup initial state, goal location, and generator for state
        this.initState = new GridWorldState(
                new GridAgent(0, 0),
                new GridLocation(this.gridWorld.getWidth() - 1, this.gridWorld.getHeight() - 1, "X")
        );
        this.maxSteps = 1_000;

        // finally generate the domain to use
        return this.gridWorld.generateDomain();
    }

    private SADomain createMountainCar(){
        MountainCar mcGen = new MountainCar();
        this.mountainCar = mcGen;
        TerminalFunction tf = new MountainCar.ClassicMCTF(Config.GOAL_REWARD);
        //this.mountainCar.setRf(new GoalBasedRF(tf, Config.GOAL_REWARD, Config.MOVE_REWARD));
        //this.mountainCar.setTf(tf);
        this.initState = mountainCar.valleyState();
        this.maxSteps = 10000000;
        return mcGen.generateDomain();
    }

    /**
     * Builds Grid World reward function
     * @return GridWorldRewardFunction
     */
    private GridWorldRewardFunction buildRewardFunction() {
        GridWorldRewardFunction gwrf = new GridWorldRewardFunction(
                this.gridWorld.getWidth(),
                this.gridWorld.getHeight(),
                Config.MOVE_REWARD
        );

        gwrf.setReward(
                this.gridWorld.getWidth() - 1,
                this.gridWorld.getHeight() - 1,
                Config.GOAL_REWARD
        );

        return gwrf;
    }
}
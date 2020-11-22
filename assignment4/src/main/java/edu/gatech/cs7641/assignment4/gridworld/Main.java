package edu.gatech.cs7641.assignment4.gridworld;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.planning.Planner;
import burlap.domain.singleagent.mountaincar.MCState;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.environment.extensions.EnvironmentServer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.util.ArrayList;
import java.util.List;

import static burlap.behavior.policy.PolicyUtils.rollout;
import static edu.gatech.cs7641.assignment4.gridworld.Domain.MOUNTAIN_CAR;


public class Main {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting experiment...");
        System.out.println("Running domain: " + Config.DOMAIN_NAME);
        System.out.println("Running agent: " + Config.AGENT_NAME);

        // build all necessary components to run a simulation
        Domain domain = new Domain(Config.DOMAIN_NAME);
        SimulatedEnvironment env;
        if (Config.DOMAIN_NAME == MOUNTAIN_CAR){
            env = new SimulatedEnvironment(domain.getSADomain(), new MCState(domain.getMountainCar().physParams.valleyPos(), 0));
        } else {
            env = new SimulatedEnvironment(domain.getSADomain(), domain.initState());
        }
        Planner agent = Agent.buildAgent(domain.getSADomain(), domain.maxSteps());
        System.out.println("Enviroment setup");

        // create visualizer for the domain in order to interact with
        if (Config.VISUALIZE) {
            domain.visualizeDomain();
            domain.visualizeValueFunction(agent);
        }

        // iterate based on the
        System.out.println("Starting trials...");
        List<Result> results = new ArrayList<>();
        for (int i = 0; i < Config.NUM_TRIALS; i++) {

            // process the episode and resent when complete, log the time necessary
            long startTime = System.nanoTime();
            Episode episode = rollout(agent.planFromState(domain.initState()), env, domain.maxSteps());
            env.resetEnvironment();

            // divide time down to microseconds and save environement results for plotting
            long totalTime = (System.nanoTime() - startTime) / 1_000_000;
            results.add(new Result(episode.numTimeSteps(), episode.rewardSequence, totalTime));

            if ((i + 1) % 10 == 0) {
                System.out.println("Processed trials: " + (i + 1));
            }
        }

        // log and plot results with these helper methods
        System.out.println("Calculating and showcasing results");
        logResults(results);
        if (Config.PLOT) {
            plotResults(results);
        }

        // for a learning agent BURLAP has a visualization tool built-in
        if (Config.Q_LEARN_PLOT && Config.AGENT_NAME.equals(Agent.Q_LEARNER)) {
            System.out.println("Building learning agent visuals");

            LearningAgentFactory learningFactory = new LearningAgentFactory() {
                public String getAgentName() {
                    return Config.AGENT_NAME;
                }

                public LearningAgent generateAgent() {
                    return Agent.buildQLearner(domain.getSADomain(), domain.maxSteps());
                }
            };

            LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                    Config.NUM_TRIALS, domain.maxSteps(), learningFactory);

            exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                    TrialMode.MOST_RECENT_AND_AVERAGE,
                    PerformanceMetric.STEPS_PER_EPISODE,
                    PerformanceMetric.AVERAGE_EPISODE_REWARD);

            exp.startExperiment();
        }
        System.out.println("Experiment complete.");
    }

    /**
     * Helper that logs results for a simulation
     * @param results - List of Results to aggregate
     */
    private static void logResults(List<Result> results) {
        int totalSteps = 0;
        Integer bestSteps = null;
        double totalReward = 0;
        Double bestReward = null;
        double totalTime = 0;

        for (int trial = 1; trial < results.size() + 1; trial++) {
            Result result = results.get(trial - 1);

            totalSteps += result.numSteps;
            totalReward += result.reward;
            totalTime += result.time;

            if (bestSteps == null || bestSteps > result.numSteps) {
                bestSteps = result.numSteps;
            }

            if (bestReward == null || bestReward < result.reward) {
                bestReward = result.reward;
            }
        }

        System.out.println("Results from trials:");
        System.out.println("Average number of steps:    " + totalSteps / results.size());
        System.out.println("Best number of steps:       " + bestSteps);
        System.out.println("Average reward:             " + totalReward / results.size());
        System.out.println("Best reward:                " + bestReward);
        System.out.println("Average processing time:    " + totalTime / results.size());
    }

    /**
     * Helper that plots results for a simulation
     * @param results - List of Results to aggregate
     */
    private static void plotResults(List<Result> results) {
        XYLineChart_AWT chart = new XYLineChart_AWT("Reward per trial", Config.AGENT_NAME, results);
        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);
    }

    /**
     * Results class tracks all data gathered from the environment simulation for a
     * single trial
     */
    private static class Result {

        int numSteps;
        long time;
        List<Double> rewards;
        double reward;

        Result(int numStepsTaken, List<Double> rewardSequence, long totalTime) {
            this.numSteps = numStepsTaken;
            this.rewards = rewardSequence;
            this.time = totalTime;

            // calculate the total reward earned during the trial
            this.reward = 0;
            rewardSequence.forEach(r -> this.reward += r);
        }
    }

    /**
     * Builds application frame and chart for plotting the given data set
     */
    private static class XYLineChart_AWT extends ApplicationFrame {

        XYLineChart_AWT(String title, String planner, List<Result> results) {
            super(title);

            JFreeChart xylineChart = ChartFactory.createXYLineChart(
                    title,
                    "Trial Step",
                    "Reward Value",
                    createDataset(planner, results)
            );

            ChartPanel chartPanel = new ChartPanel(xylineChart);
            chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
            final XYPlot plot = xylineChart.getXYPlot();

            XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
            plot.setRenderer(renderer);
            setContentPane(chartPanel);
        }

        private XYDataset createDataset(String planner, List<Result> results) {
            XYSeries series = new XYSeries(planner);

            for (int i = 0; i < results.size(); i++) {
                series.add(i + 1, results.get(i).reward);
            }

            return new XYSeriesCollection(series);
        }
    }
}
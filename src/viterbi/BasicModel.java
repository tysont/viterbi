package viterbi;

import java.util.*;

/**
 * BasicModel provides a straight forward, in memory representation of a {@link Model}.
 */
public class BasicModel implements Model {

    /**
     * The initial state for the model, which is often not possible to transition into.
     */
    private State initialState;
    /**
     * The list of possible states for the model.
     */
    private List<State> states;
    /**
     * The list of possible observations for the model.
     */
    private List<Observation> observations;

    /**
     * Create an empty BasicModel.
     */
    public BasicModel()
    {
        this.initialState = null;
        this.states = new ArrayList<State>();
        this.observations = new ArrayList<Observation>();
    }

    /**
     * Create a BasicModel with state.
     * @param initialState the initial state
     * @param states the list of possible states
     * @param observations the list of possible observations
     */
    public BasicModel(State initialState, List<State> states, List<Observation> observations) {

        this.initialState = initialState;
        this.states = states;
        this.observations = observations;
    }

    /**
     * Generates a probability matrix in order to find a Viterbi Path.
     * @param observationSequence a sequence of observations from which a path is desired
     * @return a probability matrix of the log likelihood of each observation, in each state
     */
    private Double[][] getProbabilityMatrix(List<Observation> observationSequence) {

        // Iterate over each observation.
        List<State> states = getTransitionStates();
        Double[][] probabilities = new Double[states.size()][observationSequence.size()];
        for (int o = 0; o < observationSequence.size(); o++) {

            Observation observationFromSequence = observationSequence.get(o);

            // Iterate over each state.
            for (int s = 0; s < states.size(); s++) {

                State toState = states.get(s);
                double maxProbability = Integer.MIN_VALUE;

                // If it's the first observation, take the max probability of the outcomes from the initial state.
                if (o == 0) {

                    double transitionProbability = initialState.getTransitionProbability(toState);
                    double emissionProbability = toState.getEmissionProbability(observationFromSequence);
                    maxProbability = getProbability(0, transitionProbability, emissionProbability);
                }

                // Otherwise take the max probability of outcomes from all prior states.
                else {
                    for (int f = 0; f < states.size(); f++) {
                        State fromState = states.get(f);
                        double priorProbability = probabilities[f][o - 1];
                        double transitionProbability = fromState.getTransitionProbability(toState);
                        double emissionProbability = toState.getEmissionProbability(observationFromSequence);
                        double probability = getProbability(priorProbability, transitionProbability, emissionProbability);

                        maxProbability = Math.max(maxProbability, probability);
                    }

                }

                // Store the result in the matrix.
                probabilities[s][o] = maxProbability;
            }
        }

        return probabilities;
    }

    /**
     * Walks through a probability matrix in reverse to identify the Viterbi Path.
     * @param probabilities the probability matrix of the log likelihood of each observation, in each state
     * @param observationSequence a sequence of observations from which a path is desired
     * @return the optimal path
     */
    private Path walkProbabilityMatrix(Double[][] probabilities, List<Observation> observationSequence) {

        // Find the best final state based on probabilities in the matrix.
        List<State> states = getTransitionStates();
        double finalProbability = Integer.MIN_VALUE;
        State toState = null;
        for (int s = 0; s < states.size(); s++) {

            double probability = probabilities[s][probabilities[0].length - 1];
            if ((toState == null) || (probability > finalProbability)) {
                toState = states.get(s);
                finalProbability = probability;
            }
        }


        // Iterate over each observation in reverse.
        Path path = new Path(observationSequence, finalProbability);
        Segment segment = null;
        for (int o = probabilities[0].length - 1; o > 0; o--) {

            // If it's the first observation or the state changed, create a new segment.
            Observation observationFromSequence = observationSequence.get(o);
            if (segment == null) {
                segment = new Segment();
                segment.setState(toState);
                segment.setLastObservation(observationFromSequence);
            }

            // Find the prior state that was used to get the current probability.
            double maxProbability = probabilities[states.indexOf(toState)][o];
            for (int f = 0; f < states.size(); f++) {
                State fromState = states.get(f);

                double priorProbability = probabilities[f][o - 1];
                double transitionProbability = fromState.getTransitionProbability(toState);
                double emissionProbability = toState.getEmissionProbability(observationFromSequence);
                double probability = getProbability(priorProbability, transitionProbability, emissionProbability);

                if (probability == maxProbability) {

                    // Once the state is found, check whether it's a state change.
                    if (!fromState.equals(toState)) {
                        segment.setFirstObservation(observationFromSequence);
                        path.addSegment(segment);
                        segment = null;
                    }

                    toState = fromState;
                    break;
                }
            }

        }

        // Add the final segment.
        segment.setFirstObservation(observationSequence.get(0));
        path.addSegment(segment);

        return path;
    }

    /**
     * Calculate the priority given a set of input parameters. Equation can be tuned depending on whether the raw
     * probability or log of the probability is desired, but an addition of the log probability is required in order
     * to prevent underflow on larger data sets.  Note that the initial call must be modified so that it's passing
     * in a prior probability of 1 (rather than 0) to use the raw probability.
     * @param priorProbability the probability of the prior state
     * @param transitionProbability the probability of transitioning to the current state
     * @param emissionProbability the probability of emitting the current observation
     * @return the probability, represented as a negative number if using the log of the probability formula
     */
    private double getProbability(double priorProbability, double transitionProbability, double emissionProbability) {
        //return priorProbability * transitionProbability * emissionProbability;
        return priorProbability + (double)Math.log(transitionProbability) + (double)Math.log(emissionProbability);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        for (State state : getStates()) {
            sb.append(i + ". " + state.getIdentifier() + " ");

            if (state.getTransitionProbabilities().size() > 0) {
                sb.append("(Transitions:");
                int j = 0;
                for (State toState : state.getTransitionProbabilities().keySet()) {
                    if (j > 0) sb.append(",");
                    sb.append(toState.getIdentifier() + "=" + state.getTransitionProbabilities().get(toState));
                    j++;
                }

                sb.append(") ");
            }

            if (state.getEmissionProbabilities().size() > 0) {
                sb.append("(Emissions:");
                int j = 0;
                for (Observation observation : state.getEmissionProbabilities().keySet()) {
                    if (j > 0) sb.append(",");
                    sb.append(observation + "=" + state.getEmissionProbabilities().get(observation));
                    j++;
                }

                sb.append(") ");
            }

            sb.append(System.lineSeparator());
            i++;
        }

        return sb.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<State> getStates() {
        return states;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setStates(List<State> states) {
        this.states = states;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addStates(State... states) {
        Collections.addAll(getStates(), states);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<State> getTransitionStates() {
        List<State> states = new ArrayList<State>();
        for (State state : getStates()) {
            for (State transitionState : state.getTransitionProbabilities().keySet()) {
                if ((state.getTransitionProbabilities().get(transitionState) > 0f) &&
                        (!states.contains(transitionState))){

                    states.add(transitionState);
                }
            }
        }

        return states;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Observation> getObservations() {
        return observations;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setObservations(List<Observation> observations) {
        this.observations = observations;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addObservations(Observation... observations) {
        Collections.addAll(getObservations(), observations);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public State getInitialState() {
        return initialState;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setInitialState(State initialState) {
        this.initialState = initialState;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Path getPath(List<Observation> observationSequence) {

        Double[][] probabilities = getProbabilityMatrix(observationSequence);
        return walkProbabilityMatrix(probabilities, observationSequence);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void train(Path path) {

        Map<List<Object>, Double> transitionProbabilities = path.getTransitionProbabilities();
        Map<List<Object>, Double> emissionProbabilities = path.getEmissionProbabilities();

        for (State state : getStates()) {
            if (state != initialState) {
                state.clear();
                for (List<Object> key : transitionProbabilities.keySet()) {
                    if (state == key.get(0)) {
                        State toState = (State)key.get(1);
                        double probability = transitionProbabilities.get(key);
                        state.addTransitionProbability(toState, probability);
                    }
                }

                for (List<Object> key : emissionProbabilities.keySet()) {
                    if (state == key.get(0)) {
                        Observation observation = (Observation)key.get(1);
                        double probability = emissionProbabilities.get(key);
                        state.addEmissionProbability(observation, probability);
                    }
                }
            }
        }
    }
}

package neuralnetwork

import (
	"math"
	"math/rand"
	"sync"
)

type Layer struct {
	NumNIn  int `json:"num_nodes_in"`
	NumNOut int `json:"num_nodes_out"`

	Weights []float64 `json:"weights"`
	Biases  []float64 `json:"biases"`

	lossGradientW, lossGradientB     []float64 `json:"-"`
	weightVelocities, biasVelocities []float64 `json:"-"`
	muW                              sync.Mutex
	muB                              sync.Mutex

	ActivationFn IActivation `json:"-"`
}

func NewLayer(numIn, numOut int, rng *rand.Rand) *Layer {
	l := &Layer{
		NumNIn: numIn, NumNOut: numOut,
	}
	l.Weights = make([]float64, numIn*numOut)
	l.Biases = make([]float64, numOut)

	l.lossGradientW = make([]float64, len(l.Weights))
	l.lossGradientB = make([]float64, numOut)

	l.weightVelocities = make([]float64, len(l.Weights))
	l.biasVelocities = make([]float64, numOut)

	l.InitializeRandomWeights(rng)
	return l
}

func (l *Layer) SetActivation(act ActivationType) {
	l.ActivationFn = GetActivationFromType(act)
}

func (l *Layer) InitializeRandomWeights(rng *rand.Rand) {
	for i := range l.Weights {
		l.Weights[i] = randomIn(rng, 0, 1) / math.Sqrt(float64(l.NumNIn))
	}
}

func (l *Layer) CalculateLearningOutputs(inputs []float64, learnData *LayerLearnData) []float64 {
	learnData.inputs = inputs

	for nodeOut := 0; nodeOut < l.NumNOut; nodeOut++ {
		weightedInput := l.Biases[nodeOut]
		for nodeIn := 0; nodeIn < l.NumNIn; nodeIn++ {
			weightedInput += inputs[nodeIn] * l.GetWeight(nodeIn, nodeOut)
		}
		learnData.weightedInputs[nodeOut] = weightedInput
	}

	for i := range learnData.activations {
		learnData.activations[i] = l.ActivationFn.Activate(learnData.weightedInputs, i)
	}

	return learnData.activations
}

func (l *Layer) CalculateOutputLayerNodeValues(layerLearnData *LayerLearnData, expectedOutputs []float64, loss ILoss) (lossDerivative float64) {
	for i := 0; i < len(layerLearnData.nodeValues); i++ {
		lossDerivative = loss.LossDerivative(layerLearnData.activations[i], expectedOutputs[i])
		activationDerivative := l.ActivationFn.Derivative(layerLearnData.weightedInputs, i)
		layerLearnData.nodeValues[i] = lossDerivative * activationDerivative
	}
	return
}

func (l *Layer) UpdateGradients(layerLearnData *LayerLearnData) {

	l.muW.Lock()
	for nodeOut := 0; nodeOut < l.NumNOut; nodeOut++ {
		nodeValue := layerLearnData.nodeValues[nodeOut]
		for nodeIn := 0; nodeIn < l.NumNIn; nodeIn++ {
			derivativeLossWrtWeight := layerLearnData.inputs[nodeIn] * nodeValue
			l.lossGradientW[l.GetFlatWeightIndex(nodeIn, nodeOut)] += derivativeLossWrtWeight
		}
	}
	l.muW.Unlock()

	l.muB.Lock()
	for nodeOut := 0; nodeOut < l.NumNOut; nodeOut++ {
		derivativeLossWrtBias := 1 * layerLearnData.nodeValues[nodeOut]
		l.lossGradientB[nodeOut] += derivativeLossWrtBias
	}
	l.muB.Unlock()
}

func (l *Layer) CalculateHiddenLayerNodeValues(layerLearnData *LayerLearnData, oldLayer *Layer, oldNodeValues []float64) {
	for newNodeIndex := 0; newNodeIndex < l.NumNOut; newNodeIndex++ {
		newNodeVal := 0.0
		for oldNodeIndex := 0; oldNodeIndex < len(oldNodeValues); oldNodeIndex++ {
			weightedInputDerivative := oldLayer.GetWeight(newNodeIndex, oldNodeIndex)
			newNodeVal += weightedInputDerivative * oldNodeValues[oldNodeIndex]
		}
		newNodeVal *= l.ActivationFn.Derivative(layerLearnData.weightedInputs, newNodeIndex)
		layerLearnData.nodeValues[newNodeIndex] = newNodeVal
	}
}

func (l *Layer) ApplyGradient(learnRate, regularization, momentum float64) {
	weightDecay := (1 - regularization*learnRate)
	for i := 0; i < len(l.Weights); i++ {
		weight := l.Weights[i]
		velocity := l.weightVelocities[i]*momentum - l.lossGradientW[i]*learnRate

		l.weightVelocities[i] = velocity
		l.Weights[i] = weight*weightDecay + velocity
		l.lossGradientW[i] = 0
	}

	for i := 0; i < len(l.Biases); i++ {
		velocity := l.biasVelocities[i]*momentum - l.lossGradientB[i]*learnRate
		l.biasVelocities[i] = velocity
		l.Biases[i] += velocity
		l.lossGradientB[i] = 0
	}
}

func (l *Layer) CalculateOutputs(inputs []float64) []float64 {
	weightedInputs := make([]float64, l.NumNOut)

	for nodeOut := 0; nodeOut < l.NumNOut; nodeOut++ {
		weightedInput := l.Biases[nodeOut]
		for nodeIn := 0; nodeIn < l.NumNIn; nodeIn++ {
			weightedInput += inputs[nodeIn] * l.GetWeight(nodeIn, nodeOut)
		}
		weightedInputs[nodeOut] = weightedInput
	}

	activations := make([]float64, l.NumNOut)
	for outputNode := 0; outputNode < l.NumNOut; outputNode++ {
		activations[outputNode] = l.ActivationFn.Activate(weightedInputs, outputNode)
	}

	return activations
}

func (l *Layer) GetFlatWeightIndex(inIndex, outIndex int) int {
	return outIndex*l.NumNIn + inIndex
}

func (l *Layer) GetWeight(nodeIn, nodeOut int) float64 {
	index := l.GetFlatWeightIndex(nodeIn, nodeOut)
	return l.Weights[index]
}

func randomIn(rng *rand.Rand, mean, standardDeviation float64) float64 {
	x1 := 1 - rng.Float64()
	x2 := 1 - rng.Float64()

	y1 := math.Sqrt(-2.0*math.Log(x1)) * math.Cos(2.0*math.Pi*x2)
	return y1*standardDeviation + mean
}

package main

import (
	"math/rand"
	"time"
)

type NNConf struct {
	LayerSizes []int `json:"-"`

	Activation    ActivationType `json:"hidden_activations"`
	OutActivation ActivationType `json:"output_activation"`
	Loss          LossType       `json:"loss"`
}

type NeuralNetwork struct {
	Layers []*Layer `json:"layers"`
	Loss   ILoss    `json:"-"`
	*History

	Config NNConf `json:"config"`
}

func NewNN(conf NNConf, history *History) *NeuralNetwork {
	nn := &NeuralNetwork{
		Config:  conf,
		History: history,
	}

	nn.Layers = make([]*Layer, len(conf.LayerSizes)-1)
	for i := 0; i < len(nn.Layers); i++ {
		nn.Layers[i] = NewLayer(conf.LayerSizes[i], conf.LayerSizes[i+1], rand.New(rand.NewSource(time.Now().UnixNano())))
	}

	nn.SetActivationFns(conf.Activation, conf.OutActivation)
	nn.SetLossFns(conf.Loss)

	return nn
}

func (nn *NeuralNetwork) SetActivationFns(act, outAct ActivationType) {
	for _, layer := range nn.Layers {
		layer.SetActivation(act)
	}
	nn.Layers[len(nn.Layers)-1].SetActivation(outAct)
}

func (nn *NeuralNetwork) SetLossFns(lossType LossType) {
	nn.Loss = GetLossFromType(lossType)
}

var batchLearnData []*NetworkLearnData = nil

func (nn *NeuralNetwork) Learn(trainingData []DataPoint, rate, regularization, momentum float64) {
	if batchLearnData == nil || len(batchLearnData) != len(trainingData) {
		batchLearnData = make([]*NetworkLearnData, len(trainingData))
		// debug(trainingData)
		for i := range batchLearnData {
			batchLearnData[i] = NewNetworkLearnData(nn.Layers)
		}
	}

	done := make(chan bool)
	for i, dataP := range trainingData {
		go func(i int, data DataPoint) {
			nn.UpdateGradients(data, batchLearnData[i])
			done <- true
		}(i, dataP)
	}

	for i := 0; i < len(trainingData); i++ {
		<-done
	}

	for _, layer := range nn.Layers {
		layer.ApplyGradient(rate/float64(len(trainingData)), regularization, momentum)
	}

}

func (nn *NeuralNetwork) UpdateGradients(data DataPoint, learnData *NetworkLearnData) {
	inputsToNextLayer := data.inputs
	for i, layer := range nn.Layers {
		inputsToNextLayer = layer.CalculateLearningOutputs(inputsToNextLayer, learnData.layerData[i])
	}

	// BackProp
	outputLayerIndex := len(nn.Layers) - 1
	outputLayer := nn.Layers[outputLayerIndex]
	outputLearnData := learnData.layerData[outputLayerIndex]

	outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, nn.Loss)
	outputLayer.UpdateGradients(outputLearnData)

	for i := outputLayerIndex - 1; i >= 0; i-- {
		layerLearnData := learnData.layerData[i]
		hiddenLayer := nn.Layers[i]

		hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, nn.Layers[i+1], learnData.layerData[i+1].nodeValues)
		hiddenLayer.UpdateGradients(layerLearnData)
	}
}

func (nn *NeuralNetwork) Classify(inputs []float64) (predictedClass int, outputs []float64) {
	outputs = nn.CalculateOutputs(inputs)
	predictedClass = MaxValueIndex(outputs)
	return predictedClass, outputs
}

func (nn *NeuralNetwork) CalculateOutputs(inputs []float64) []float64 {
	for _, layer := range nn.Layers {
		inputs = layer.CalculateOutputs(inputs)
	}
	return inputs
}

func (nn *NeuralNetwork) calculateTotalLoss(batch Batch) float64 {
	totalLoss := 0.0

	for _, dataPoint := range batch.data {
		_, outputs := nn.Classify(dataPoint.inputs)
		loss := nn.Loss.LossFunction(outputs, dataPoint.expectedOutputs)
		totalLoss += loss
	}

	return totalLoss
}

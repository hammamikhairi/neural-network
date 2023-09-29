package main

import "math"

type LossType int

const (
	MeanSquareError_T LossType = iota
	CrossEntropy_T

	BinaryCrossEntropy_T
)

type ILoss interface {
	LossFunction(predictedOutputs []float64, expectedOutputs []float64) float64
	LossDerivative(predictedOutput, expectedOutput float64) float64
}

type Loss struct{}

func GetLossFromType(lossType LossType) ILoss {
	switch lossType {
	case MeanSquareError_T:
		return MeanSquaredError{}
	case CrossEntropy_T:
		return CrossEntropy{}
	case BinaryCrossEntropy_T:
		return BinaryCrossEntropy{}
	default:
		panic("Unhandled loss type")
	}
}

type CrossEntropy struct{}

func (c CrossEntropy) LossFunction(predictedOutputs, expectedOutputs []float64) float64 {
	loss := 0.0
	for i := 0; i < len(predictedOutputs); i++ {
		x := predictedOutputs[i]
		y := expectedOutputs[i]
		v := 0.0

		if y == 1 {
			v = -math.Log(x)

		} else if y == 0 {
			v = -math.Log(1 - x)
		}

		if !math.IsNaN(v) {
			loss += v
		}
	}
	return loss
}

func (c CrossEntropy) LossDerivative(predictedOutput, expectedOutput float64) float64 {
	x := predictedOutput
	y := expectedOutput
	if x == 0 || x == 1 {
		return 0
	}
	return (-x + y) / (x * (x - 1))
}

type MeanSquaredError struct{}

func (mse MeanSquaredError) LossFunction(predictedOutputs, expectedOutputs []float64) float64 {
	var loss float64
	for i := 0; i < len(predictedOutputs); i++ {
		error := predictedOutputs[i] - expectedOutputs[i]
		loss += error * error
	}
	return 0.5 * loss
}

func (mse MeanSquaredError) LossDerivative(predictedOutput, expectedOutput float64) float64 {
	return predictedOutput - expectedOutput
}

type BinaryCrossEntropy struct{}

func (bce BinaryCrossEntropy) LossFunction(predictedOutputs, expectedOutputs []float64) float64 {
	loss := 0.0
	for i := 0; i < len(predictedOutputs); i++ {
		x := predictedOutputs[i]
		y := expectedOutputs[i]

		v := -y*math.Log(x) - (1-y)*math.Log(1-x)

		if !math.IsNaN(v) {
			loss += v
		}
	}
	return loss
}

func (bce BinaryCrossEntropy) LossDerivative(predictedOutput, expectedOutput float64) float64 {
	x := predictedOutput
	y := expectedOutput

	if x == 0 {
		return 1e10
	} else if x == 1 {
		return -1e10
	}

	return (-y/x + (1-y)/(1-x))
}

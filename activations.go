package main

import "math"

type ActivationType int

const (
	Sigmoid ActivationType = iota
	ReLU
	TanH
	SiLU
	Softmax
)

type IActivation interface {
	Activate(inputs []float64, index int) float64
	Derivative(inputs []float64, index int) float64
}

type Activation struct{}

func GetActivationFromType(activationType ActivationType) IActivation {
	switch activationType {
	case Sigmoid:
		return SigmoidActivation{}
	case TanH:
		return TanHActivation{}
	case ReLU:
		return ReLUActivation{}
	case SiLU:
		return SiLUActivation{}
	case Softmax:
		return SoftmaxActivation{}
	default:
		panic("Unhandled activation type")
	}
}

type SigmoidActivation struct{}

func (a SigmoidActivation) Activate(inputs []float64, index int) float64 {
	return 1.0 / (1 + math.Exp(-inputs[index]))
}

func (a SigmoidActivation) Derivative(inputs []float64, index int) float64 {
	o := a.Activate(inputs, index)
	return o * (1 - o)
}

type TanHActivation struct{}

func (a TanHActivation) Activate(inputs []float64, index int) float64 {
	e2 := math.Exp(2 * inputs[index])
	return (e2 - 1) / (e2 + 1)
}

func (a TanHActivation) Derivative(inputs []float64, index int) float64 {
	e2 := math.Exp(2 * inputs[index])
	t := (e2 - 1) / (e2 + 1)
	return 1 - t*t
}

type ReLUActivation struct{}

func (a ReLUActivation) Activate(inputs []float64, index int) float64 {
	return math.Max(0, inputs[index])
}

func (a ReLUActivation) Derivative(inputs []float64, index int) float64 {
	if inputs[index] > 0 {
		return 1
	}
	return 0
}

type SiLUActivation struct{}

func (a SiLUActivation) Activate(inputs []float64, index int) float64 {
	return inputs[index] / (1 + math.Exp(-inputs[index]))
}

func (a SiLUActivation) Derivative(inputs []float64, index int) float64 {
	sig := 1 / (1 + math.Exp(-inputs[index]))
	return inputs[index]*sig*(1-sig) + sig
}

type SoftmaxActivation struct{}

func (a SoftmaxActivation) Activate(inputs []float64, index int) float64 {
	expSum := float64(0)
	for _, input := range inputs {
		expSum += math.Exp(input)
	}
	return math.Exp(inputs[index]) / expSum
}

func (a SoftmaxActivation) Derivative(inputs []float64, index int) float64 {
	expSum := float64(0)
	for _, input := range inputs {
		expSum += math.Exp(input)
	}
	ex := math.Exp(inputs[index])

	return (ex*expSum - ex*ex) / (expSum * expSum)
}

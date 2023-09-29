package main

import (
	"fmt"

	. "github.com/hammamikhairi/neural-network"
)

func FashionMain() {

	const (
		DATASETS_PATH string = "/home/khairi/DataSets/"
		DATA_PATH     string = DATASETS_PATH + "Fashion/"

		NUM_INPUTS = 784
		NUM_LABELS = 10
	)
	conf := NNConf{
		LayerSizes:    []int{NUM_INPUTS, 128, 128, 128, 128, NUM_LABELS},
		Activation:    ReLU,
		OutActivation: Softmax,
		Loss:          CrossEntropy_T,
	}

	tConf := TrainerConf{
		Epochs:         10,
		TrainingSplit:  0.8,
		Rate:           0.05,
		RateDecay:      0.075,
		Momentum:       0.9,
		Regularization: 0.1,
		BatchSize:      32,
		OnEpochComplete: func(epochIndex int, evaluation *EvaluationData, EpochLoss float64) {
			fmt.Printf("Epoch %d -- %s -- Loss : %.4f\n", epochIndex, evaluation.GetAccuracyString(), EpochLoss)
		},
	}

	t := NewTrainer(tConf)
	t.NNInit(conf)
	t.LoadMNISTData(DATA_PATH)
	t.Train()

	// eval NN
	eval := t.Eval(true)
	println(eval.GetAccuracyString())

	// save history for visualization
	t.History.Save("hist.json")

	// save NN to use later
	t.SaveNN("nn.json")
}

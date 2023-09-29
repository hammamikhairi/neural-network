package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	. "github.com/hammamikhairi/neural-network"
)

func WinesMain() {

	const (
		DATASETS_PATH string = "/home/khairi/DataSets/"
		DATA_PATH     string = DATASETS_PATH + "RedWhiteWine/wine.csv"
	)

	var (
		numInputs  = 4
		NUM_LABELS = 3
	)

	conf := NNConf{
		LayerSizes:    []int{numInputs, 3, NUM_LABELS},
		Activation:    Sigmoid,
		OutActivation: Sigmoid,
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

	// Load Custom Data
	winesData := LoadCustomData(DATA_PATH, numInputs, NUM_LABELS)
	tr, val := SplitData(winesData, t.Config.TrainingSplit)
	t.LoadCustomData(tr, val)

	t.Train()

	// eval NN
	eval := t.Eval(true)
	println(eval.GetAccuracyString())

	// save history for visualization
	t.History.Save("hist.json")

	// save NN to use later
	t.SaveNN("nn.json")
}

func LoadCustomData(fileName string, numInputs, NUM_LABELS int) []DataPoint {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	dataPoints := []DataPoint{}
	for idx, record := range rawCSVData {

		// i removed the header during preproc, dont forget to change if needed
		if idx == -1 {
			continue
		}

		dp := make([]float64, numInputs)
		dpi := 0
		label := 0.0

		for i, val := range record {
			if i == 12 {
				label, _ = strconv.ParseFloat(val, 64)
				continue
			}

			dp[dpi], _ = strconv.ParseFloat(val, 64)
			dpi++
		}

		dataPoints = append(
			dataPoints,
			NewDataPoint(dp, int(label), NUM_LABELS),
		)
	}

	return dataPoints
}

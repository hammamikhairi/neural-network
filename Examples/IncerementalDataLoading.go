package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	. "github.com/hammamikhairi/neural-network"
)

func IncrementalTraining() {

	const (
		DATASETS_PATH string = "/home/khairi/DataSets/"
		DATA_PATH     string = DATASETS_PATH + "Doodles/"

		NUM_INPUTS = 784
		NUM_LABELS = 10
	)

	conf := NNConf{
		LayerSizes:    []int{NUM_INPUTS, 128, 128, 64, NUM_LABELS},
		Activation:    ReLU,
		OutActivation: Softmax,
		Loss:          CrossEntropy_T,
	}

	tConf := TrainerConf{
		Epochs:         20,
		TrainingSplit:  0.8,
		Rate:           0.05,
		RateDecay:      0.075,
		Momentum:       0.9,
		Regularization: 0.1,
		BatchSize:      32,
		OnEpochComplete: func(epochIndex int, evaluation *EvaluationData, EpochCost float64) {
			fmt.Printf("Epoch %d -- %s -- Cost : %.4f\n", epochIndex, evaluation.GetAccuracyString(), EpochCost)
		},
	}

	t := NewTrainer(tConf)
	t.NNInit(conf)

	// use a custom function to load the data into DataPoints
	dps := loadFiles(DATA_PATH, NUM_LABELS)

	ShuffleBatches(dps)
	tr, val := SplitData(dps, t.Config.TrainingSplit)
	t.LoadIncData(tr, val)

	t.IncTrain(NUM_LABELS)

	// eval NN
	eval := t.IncrementalEval(true, NUM_LABELS)
	println(eval.GetAccuracyString())

	// save history for visualization
	t.History.Save("hist-incremental.json")

	// save NN to use later
	t.SaveNN("nn-incremental.json")
}

var (
	IndexingMap map[string]int
)

func loadFiles(path string, NUM_LABELS int) []ImageFile {
	currentLabel := ""
	dps := []ImageFile{}

	IndexingMap = make(map[string]int)
	BASE_DIR := "Doodles"

	err := filepath.Walk(path,
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				currentLabel = info.Name()
				if currentLabel != BASE_DIR {
					IndexingMap[currentLabel] = len(IndexingMap)
				}
			} else {
				dp := ImageFile{
					FilePath: path,
					Label:    IndexingMap[currentLabel],
				}
				dps = append(dps, dp)
			}

			return nil
		})
	if err != nil {
		log.Println(err)
	}

	return dps
}

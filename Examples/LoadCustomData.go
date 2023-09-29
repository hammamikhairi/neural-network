package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	. "github.com/hammamikhairi/neural-network"
)

func main() {

	const (
		DATASETS_PATH string = "/home/khairi/DataSets/"
		DATA_PATH     string = DATASETS_PATH + "HandNoise/pictures/"

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
	dps := loadPictures(DATA_PATH, NUM_LABELS)

	ShuffleBatches(dps)
	tr, val := SplitData(dps, t.Config.TrainingSplit)
	t.LoadCustomData(tr, val)

	t.Train()

	// eval NN
	eval := t.Eval(true)
	println(eval.GetAccuracyString())

	// save history for visualization
	t.History.Save("hist.json")

	// save NN to use later
	t.SaveNN("nn-final-layers-go-brrrr.json")
}

func loadPictures(path string, NUM_LABELS int) []DataPoint {
	currentLabel := 0
	dps := []DataPoint{}
	err := filepath.Walk(path,
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if info.IsDir() {
				c := info.Name()
				cc, err := strconv.ParseInt(c, 10, 32)
				if err != nil {
					return nil
				}
				currentLabel = int(cc)

			} else {
				dp := LoadSinglePNGImage(path, currentLabel, NUM_LABELS)
				dps = append(dps, dp)
			}

			return nil
		})
	if err != nil {
		log.Println(err)
	}

	return dps
}

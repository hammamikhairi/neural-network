package main

import . "github.com/hammamikhairi/neural-network"

const (
	DATASETS_PATH string = "/home/khairi/DataSets/"
	DATA_PATH     string = DATASETS_PATH + "Fashion-Minst/"
)

func LoadNNFromFile() {

	var nnPath string = "./nn.json"

	t := NewTrainer(TrainerConf{})
	e := t.LoadNNFromFile(nnPath)

	if e != nil {
		panic(e)
	}

	t.LoadMNISTData(DATA_PATH)

	// eval NN
	eval := t.Eval(true)
	println(eval.GetAccuracyString())
}

package main

import . "github.com/hammamikhairi/neural-network"

func PredictOne() {

	const (
		NUM_LABELS = 10
	)

	var (
		nnPath  string = "./nn.json"
		imgPath string = "./image.png"
	)

	t := NewTrainer(TrainerConf{})
	e := t.LoadNNFromFile(nnPath)
	if e != nil {
		panic(e)
	}

	img := LoadSinglePNGImage(imgPath, 3, NUM_LABELS)
	println("Predicted label : ", t.PredictSingle(img))
}

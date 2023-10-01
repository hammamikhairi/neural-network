package main

import "sync"

func LoadBatch(files []ImageFile, numLabels int) []DataPoint {
	dps := []DataPoint{}
	for _, file := range files {
		dp := LoadSinglePNGImage(file.FilePath, file.Label, numLabels)

		dps = append(dps, dp)
	}

	return dps
}

func (t *Trainer) LoadIncData(training, validation []ImageFile) {
	println("[INFO] Loading Inc Data")
	t.incTrainingData, t.incValidationData = training, validation
	t.incTrainingBatches = CreateMiniIncBatches(t.incTrainingData, t.Config.BatchSize)
}

func (t *Trainer) IncTrain(numLabels int) {
	println("[INFO] Started Incremental Training")
	currentRate := t.Config.Rate
	totalBatches := len(t.incTrainingBatches)

	for epochIdx := 0; epochIdx < t.Config.Epochs; epochIdx++ {
		epochLoss := 0.0
		for i := 0; i < totalBatches; i++ {
			displayProgress(i, totalBatches)
			batch := LoadBatch(t.incTrainingBatches[i].files, numLabels)
			t.NN.Learn(batch, currentRate, t.Config.Regularization, t.Config.Momentum)
			epochLoss += t.NN.calculateTotalLoss(batch)
		}

		evalutation := t.IncrementalEval(true, numLabels)
		epochLoss = epochLoss / float64(totalBatches)
		t.History.Loss = append(t.History.Loss, epochLoss)
		t.History.Acc = append(t.History.Acc, evalutation.GettAccuracy())

		if t.Config.OnEpochComplete != nil {
			t.Config.OnEpochComplete(epochIdx, evalutation, epochLoss)
		}

		ShuffleBatches(t.incTrainingBatches)
		currentRate = (1.0 / (1.0 + t.Config.RateDecay*float64(epochIdx))) * t.Config.Rate
	}
}

func (t *Trainer) IncrementalEval(useEvalData bool, numLabels int) *EvaluationData {

	if useEvalData {
		return t.IncrementalEvaluate(t.incValidationData, numLabels)
	} else {
		return t.IncrementalEvaluate(t.incTrainingData, numLabels)
	}
}

func (t *Trainer) IncrementalEvaluate(data []ImageFile, numLabels int) *EvaluationData {
	evalData := NewEvaluationData(numLabels)
	evalData.total = len(data)

	var mu sync.Mutex

	for _, dp := range data {
		img := LoadSinglePNGImage(dp.FilePath, dp.Label, numLabels)
		output := t.NN.CalculateOutputs(img.inputs)
		predictedLabel := MaxValueIndex(output)

		mu.Lock()

		evalData.totalPerClass[dp.Label]++

		if predictedLabel == dp.Label {
			evalData.numCorrectPerClass[dp.Label]++
			evalData.numCorrect++
		} else {
			evalData.wronglyPredictedAs[predictedLabel]++
		}

		mu.Unlock()
	}

	return evalData
}

func CreateMiniIncBatches(data []ImageFile, batchSize int) []IncBatch {
	numBatches := len(data) / batchSize
	batches := make([]IncBatch, numBatches)

	for i := 0; i < numBatches; i++ {
		startIndex := i * batchSize
		endIndex := (i + 1) * batchSize
		batchData := data[startIndex:endIndex]
		batches[i] = IncBatch{files: batchData}
	}

	// If there are any remaining data points not fitting in a full batch
	if len(data)%batchSize != 0 {
		remainingData := data[numBatches*batchSize:]
		batches = append(batches, IncBatch{files: remainingData})
	}

	return batches
}

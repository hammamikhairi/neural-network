package neuralnetwork

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sync"
)

type Trainer struct {
	NN     *NeuralNetwork
	Config TrainerConf
	*History

	trainingData    []DataPoint
	validationData  []DataPoint
	trainingBatches []Batch

	incTrainingData    []ImageFile
	incValidationData  []ImageFile
	incTrainingBatches []IncBatch
}

type TrainerConf struct {
	OnEpochComplete func(epochIndex int, evaluation *EvaluationData, EpochLoss float64)
	TrainingSplit   float64
	BatchSize       int
	Epochs          int

	Rate, RateDecay          float64
	Momentum, Regularization float64
}

func NewTrainer(tConf TrainerConf) *Trainer {
	t := &Trainer{
		Config: tConf,
		History: &History{
			Loss: []float64{},
			Acc:  []float64{},
		},
	}

	return t
}

func (t *Trainer) NNInit(nnConf NNConf) {
	t.NN = NewNN(
		nnConf,
		t.History,
	)
}

func (t *Trainer) LoadCustomData(training, validation []DataPoint) {
	println("[INFO] Loading Data")
	t.trainingData, t.validationData = training, validation
	t.trainingBatches = CreateMiniBatches(t.trainingData, t.Config.BatchSize)
}

func (t *Trainer) LoadMNISTData(path string) {

	var (
		TRAINING_DATA   = path + "train-images-idx3-ubyte"
		TRAINING_LABELS = path + "train-labels-idx1-ubyte"
		EVAL_DATA       = path + "t10k-images-idx3-ubyte"
		EVAL_LABELS     = path + "t10k-labels-idx1-ubyte"
	)

	println("[INFO] Loading MNIST Data")
	t.trainingData = LoadMNISTData(TRAINING_DATA, TRAINING_LABELS)
	t.validationData = LoadMNISTData(EVAL_DATA, EVAL_LABELS)
	t.trainingBatches = CreateMiniBatches(t.trainingData, t.Config.BatchSize)
}

func (t *Trainer) Train() {
	println("[INFO] Started Training")
	currentRate := t.Config.Rate
	totalBatches := len(t.trainingBatches)
	for epochIdx := 0; epochIdx < t.Config.Epochs; epochIdx++ {
		epochLoss := 0.0
		for i := 0; i < totalBatches; i++ {
			displayProgress(i, totalBatches)
			t.NN.Learn(t.trainingBatches[i].data, currentRate, t.Config.Regularization, t.Config.Momentum)

			epochLoss += t.NN.calculateTotalLoss(t.trainingBatches[i].data)
		}

		evalutation := t.Eval(true)
		epochLoss = epochLoss / float64(totalBatches)
		t.History.Loss = append(t.History.Loss, epochLoss)
		t.History.Acc = append(t.History.Acc, evalutation.GettAccuracy())

		if t.Config.OnEpochComplete != nil {
			t.Config.OnEpochComplete(epochIdx, evalutation, epochLoss)
		}

		ShuffleBatches(t.trainingBatches)
		currentRate = (1.0 / (1.0 + t.Config.RateDecay*float64(epochIdx))) * t.Config.Rate
	}
}

func (t *Trainer) Eval(useEvalData bool) *EvaluationData {
	if useEvalData {
		return t.Evaluate(t.validationData)
	} else {
		return t.Evaluate(t.trainingData)
	}
}

func (t *Trainer) Evaluate(data []DataPoint) *EvaluationData {
	evalData := NewEvaluationData(len(data[0].expectedOutputs))
	evalData.total = len(data)

	var mu sync.Mutex

	for _, dp := range data {
		output := t.NN.CalculateOutputs(dp.inputs)
		predictedLabel := MaxValueIndex(output)

		mu.Lock()

		evalData.totalPerClass[dp.label]++

		if predictedLabel == dp.label {
			evalData.numCorrectPerClass[dp.label]++
			evalData.numCorrect++
		} else {
			evalData.wronglyPredictedAs[predictedLabel]++
		}

		mu.Unlock()
	}

	return evalData
}

func (t *Trainer) Classify(inputs []float64) []float64 {
	return t.NN.CalculateOutputs(inputs)
}

func (t *Trainer) PredictSingle(data DataPoint) int {
	output := t.Classify(data.inputs)
	for i, percentage := range output {
		strPercentage := fmt.Sprintf("%.4f", percentage*100)
		fmtPercentage := ""
		if len(strPercentage) == 6 {
			fmtPercentage = "0" + strPercentage
		} else {
			fmtPercentage = strPercentage
		}
		fmt.Printf("Label : %d => %s%%\n", i, fmtPercentage)
	}
	return MaxValueIndex(output)
}

func (t *Trainer) SaveNN(path string) error {
	jsonData, err := json.MarshalIndent(*t.NN, "", "  ")
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, jsonData, 0644)
	if err != nil {
		return err
	}

	return nil
}

func (t *Trainer) LoadNNFromFile(path string) error {
	jsonData, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	t.NN = &NeuralNetwork{}
	err = json.Unmarshal(jsonData, t.NN)
	if err != nil {
		return err
	}

	t.NN.SetActivationFns(t.NN.Config.Activation, t.NN.Config.OutActivation)
	t.NN.SetLossFns(t.NN.Config.Loss)

	return nil
}

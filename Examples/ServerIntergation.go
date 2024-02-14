package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	. "github.com/hammamikhairi/neural-network"
)

const (
	NUM_INPUTS = 784
	NUM_LABELS = 10

	DATASETS_PATH string = "/home/khairi/DataSets/"
	DATA_PATH     string = DATASETS_PATH + "Fashion-Minst/"
)

var (
	t *Trainer
)

func statsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// time stat
		start := time.Now()

		next.ServeHTTP(w, r)

		// Log stats
		log.Println(
			fmt.Sprintf(
				"Endpoint %s took %s", r.URL.Path, time.Since(start).String(),
			),
		)
	})
}

func main() {
	println("khqklhflqkf")
	t = NewTrainer(TrainerConf{})
	e := t.LoadNNFromFile("nn-final-layers-go-brrrr.json")
	if e != nil {
		panic(e)
	}

	http.HandleFunc("/", statsMiddleware(hola))
	http.HandleFunc("/predict", statsMiddleware(Classify))
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}

}

func hola(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")

	json.NewEncoder(w).Encode("Hello curious one")

}

func Classify(w http.ResponseWriter, r *http.Request) {

	start := time.Now()
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	pixels := r.URL.Query().Get("pixels")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	inputs := Conv(pixels)

	dataPoint := NewDataPoint(inputs, 0, NUM_LABELS)
	percentages := t.PredictionsPercentages(dataPoint)

	end := time.Since(start).String()

	w.WriteHeader(http.StatusOK)
	err := json.NewEncoder(w).Encode(
		map[string]interface{}{
			"preditions": percentages,
			"time":       end,
		},
	)

	if err != nil {
		panic(err)
	}

}

func Conv(inputString string) []float64 {
	inputs := strings.Split(inputString, ",")

	// Step 2: Convert substrings to floats
	var floatSlice []float64
	for _, input := range inputs {
		input = strings.TrimSpace(input) // Remove leading/trailing spaces
		f, err := strconv.ParseFloat(input, 64)
		if err != nil {
			log.Fatalf("Error converting %s to float: %v", input, err)
		}
		floatSlice = append(floatSlice, f)
	}

	return floatSlice
}

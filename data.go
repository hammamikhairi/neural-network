package main

import (
	"encoding/binary"
	"image"
	_ "image/png"
	"math/rand"
	"os"
	"time"
)

type DataPoint struct {
	inputs          []float64
	expectedOutputs []float64
	label           int
}

type ImageFile struct {
	FilePath string
	Label    int
}

type Batch struct {
	data []DataPoint
}

type IncBatch struct {
	files []ImageFile
}

func NewDataPoint(inputs []float64, label, numLabels int) DataPoint {
	dp := DataPoint{
		inputs: inputs,
		label:  label,
	}
	dp.expectedOutputs = dp.createOneHot(label, numLabels)
	return dp
}

func (dp *DataPoint) createOneHot(index, num int) []float64 {
	oneHot := make([]float64, num)
	oneHot[index] = 1
	return oneHot
}

func NewBatch(data []DataPoint) *Batch {
	return &Batch{data: data}
}

func CreateMiniBatches(data []DataPoint, batchSize int) []Batch {
	numBatches := len(data) / batchSize
	batches := make([]Batch, numBatches)

	for i := 0; i < numBatches; i++ {
		startIndex := i * batchSize
		endIndex := (i + 1) * batchSize
		batchData := data[startIndex:endIndex]
		batches[i] = Batch{data: batchData}
	}

	// If there are any remaining data points not fitting in a full batch
	if len(data)%batchSize != 0 {
		remainingData := data[numBatches*batchSize:]
		batches = append(batches, Batch{data: remainingData})
	}

	return batches
}

func SplitData[T any](allData []T, trainingSplit float64) ([]T, []T) {
	ShuffleBatches(allData)

	totalLength := len(allData)
	trainingLength := int(float64(totalLength) * trainingSplit)

	return allData[:trainingLength], allData[trainingLength:]
}

func ShuffleBatches[T any](batches []T) {
	rand.Seed(time.Now().UnixNano())

	for i := len(batches) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		batches[i], batches[j] = batches[j], batches[i]
	}
}

func LoadMNISTData(imPath, labPath string) []DataPoint {
	trainingImages, err := readMNISTImages(imPath)
	if err != nil {
		panic(err)
	}

	trainingLabels, err := readMNISTLabels(labPath)
	if err != nil {
		panic(err)
	}

	return convertToDataPoints(trainingImages, trainingLabels, 10)
}

func convertToDataPoints(images [][]float64, labels []int, numLabels int) []DataPoint {
	var dataPoints []DataPoint

	for i, image := range images {
		dp := NewDataPoint(image, labels[i], numLabels)
		dataPoints = append(dataPoints, dp)
	}

	return dataPoints
}

func readMNISTImages(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magicNumber uint32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}

	var numImages uint32
	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}

	var numRows, numCols uint32
	if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
		return nil, err
	}

	imageSize := numRows * numCols

	images := make([][]float64, numImages)

	for i := range images {
		imageData := make([]byte, imageSize)
		if err := binary.Read(file, binary.BigEndian, &imageData); err != nil {
			return nil, err
		}

		// Convert byte data to float64 and normalize to [0, 1]
		imageDataFloat := make([]float64, len(imageData))
		for j, val := range imageData {
			imageDataFloat[j] = float64(val) / 255.0
		}

		images[i] = imageDataFloat
	}

	return images, nil
}

func readMNISTLabels(filename string) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magicNumber uint32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}

	var numLabels uint32
	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	labels := make([]int, numLabels)

	for i := range labels {
		var label uint8
		if err := binary.Read(file, binary.BigEndian, &label); err != nil {
			return nil, err
		}
		labels[i] = int(label)
	}

	return labels, nil
}

func LoadSinglePNGImage(imPath string, label, numLabels int) DataPoint {
	pixels, err := loadImagePixels(imPath)
	if err != nil {
		panic(err)
	}
	return NewDataPoint(pixels, label, numLabels)
}

func loadImagePixels(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		print("here")
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	var flattenedImage []float64

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			grayValue := (float64(r) + float64(g) + float64(b)) / (3 * 65535) // Normalize to [0, 1]
			flattenedImage = append(flattenedImage, grayValue)
		}
	}

	return flattenedImage, nil
}
